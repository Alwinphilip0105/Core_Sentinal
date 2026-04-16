"""
Risk-Aware Assistant: PyQt6 UI driven by clipboard paste into LLMs (tray + RemediationDialog).
In-app toast notifications replace modal message boxes for errors and notices.

Usage:
  python main.py

Flow:
  - Windows: a low-level WH_KEYBOARD_LL hook suppresses Ctrl+V in LLM windows before the app sees it;
    clipboard text is scored on a background worker thread; safe/warn replays synthetic Ctrl+V;
    block opens remediation without replaying (see win_paste_hook).
  - detect_llm_window() (in win_paste_hook) gates monitoring when \"Monitor LLM only\" is on.
  - score_clipboard_with_pii runs on the paste worker thread; results are delivered to the GUI via
    Qt signals. infer.preload_guardrail_model() at startup avoids load on first paste unless
    GUARDRAIL_BLOCKING_PRELOAD=0.
  - Floating pill (RiskBubble): Grammarly-style UI; monitoring refreshed ~1s.
  - Main thread uses result[\"action\"] from the scorer:
    silent — updates pill only; suppressed Ctrl+V is replayed after scoring.
    warn — replay paste first, then RemediationDialog in review mode (amber banner; not hold).
    block — remediation without replaying first; hold mode when high/critical.
"""
from __future__ import annotations

import os as _os
_torch_lib = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.venv/Lib/site-packages/torch/lib")
if _os.path.isdir(_torch_lib) and hasattr(_os, "add_dll_directory"):
    _os.add_dll_directory(_os.path.abspath(_torch_lib))
del _os, _torch_lib




import os as _os


import copy
import json
import os
import random
import subprocess
import sys
import threading
import webbrowser
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

_guardrail_dir = Path(__file__).resolve().parent
if str(_guardrail_dir) not in sys.path:
    sys.path.insert(0, str(_guardrail_dir))

from PyQt6.QtCore import QObject, QRect, QThread, Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QCursor, QFont, QIcon, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QMessageBox,
    QSystemTrayIcon,
    QToolTip,
)

from guardrail_runtime import (
    is_guard_snoozed,
    is_monitoring_paused,
    is_recent_duplicate,
    record_recent_text,
    record_scored_clipboard_risk_score,
    set_monitoring_paused,
)
from feedback_store import get_feedback_stats, should_trigger_retrain
from font_clamp import install_qt_message_filter, normalize_application_font
import user_settings
from guardrail_logs import export_scoring_events_csv, log_clipboard_event
from infer import (
    preload_guardrail_model,
    score_clipboard_with_pii,
    should_bypass_duplicate_skip_for_text,
)
from ui_remediation_dialog import RemediationDialog
from ui_risk_bubble import RiskBubble

from toast import show_toast, toast_safe
from sentinel_sync_daemon import start_sync_thread

# Low-risk silent feedback (CharacterEvent near pill; not full panel)
_SAFE_MESSAGES: list[tuple[str, str, str]] = [
    ("😌", "All clear!", "Nothing risky detected"),
    ("😎", "Looking good!", "Safe to paste"),
    ("🙌", "Nice one!", "No PII found"),
    ("✅", "Green light!", "Paste approved"),
    ("🛡️", "Sentinel approves!", "Safe content"),
    ("😄", "You're on a roll!", "Keep it up"),
    ("👍", "All good here!", "No issues found"),
    ("🤖", "Scan complete!", "Nothing to report"),
]
_STREAK_MILESTONES: frozenset[int] = frozenset({5, 10, 25, 50, 100})

# Retrain trigger can be tuned later via env without code changes.
_RETRAIN_TRIGGER_MIN = max(
    1,
    int(os.environ.get("GUARDRAIL_RETRAIN_MIN_CORRECTIONS", "10") or "10"),
)


class _RetrainNotifier(QObject):
    """Marshals retrain completion toasts from a worker thread to the GUI thread."""

    show_success = pyqtSignal()
    show_failure = pyqtSignal(str)


_retrain_notifier: _RetrainNotifier | None = None


def _train_subprocess_timeout() -> float | None:
    """Seconds for train.py, or None = no limit. Env GUARDRAIL_TRAIN_TIMEOUT_SEC (default 4h)."""
    raw = os.environ.get("GUARDRAIL_TRAIN_TIMEOUT_SEC", "14400").strip().lower()
    if raw in ("0", "none", "unlimited", "off"):
        return None
    try:
        sec = int(raw)
        return None if sec < 0 else float(max(60, sec))
    except ValueError:
        return 14400.0


def auto_retrain() -> None:
    """Run merge_feedback_to_training.py then train.py in a background thread."""

    def _run() -> None:
        global _retrain_notifier
        timeout_env = "GUARDRAIL_TRAIN_TIMEOUT_SEC"
        default_timeout = 14400
        try:
            guardrail_root = Path(__file__).resolve().parent
            python = sys.executable
            cwd = str(guardrail_root)
            train_timeout = _train_subprocess_timeout()

            merge = guardrail_root / "merge_feedback_to_training.py"
            if merge.exists():
                r = subprocess.run(
                    [python, str(merge)],
                    cwd=cwd,
                    timeout=120,
                )
                if r.returncode != 0:
                    raise RuntimeError(
                        f"merge_feedback_to_training.py exited {r.returncode}"
                    )

            train = guardrail_root / "train.py"
            if not train.exists():
                alt = guardrail_root.parent / "train.py"
                if alt.exists():
                    train = alt
            if not train.exists():
                raise RuntimeError("train.py not found under guardrail or repo root")

            r = subprocess.run(
                [python, str(train)],
                cwd=cwd,
                timeout=train_timeout,
            )
            if r.returncode != 0:
                raise RuntimeError(f"train.py exited {r.returncode}")

            if _retrain_notifier is not None:
                _retrain_notifier.show_success.emit()
        except subprocess.TimeoutExpired as exc:
            timed_out_after = int(getattr(exc, "timeout", 0) or 0)
            if timed_out_after <= 0:
                timed_out_after = int(train_timeout or default_timeout)
            print(
                f"[retrain] TIMED OUT after {timed_out_after}s.\n"
                f"  FIX: Your {timeout_env} is set to {timed_out_after}.\n"
                f"  Options:\n"
                f"    1. Remove {timeout_env} from your environment "
                f"(default: {default_timeout}s / 4hr)\n"
                f"    2. Set {timeout_env}=14400 for 4 hours\n"
                f"    3. Set {timeout_env}=none for no time limit\n"
                f"  Check: .env file, system env vars, shell profile (~/.bashrc, ~/.zshrc)",
                flush=True,
            )
            if _retrain_notifier is not None:
                _retrain_notifier.show_failure.emit("train.py timed out (see log)")
        except Exception as e:
            print(f"[retrain] failed: {e}", flush=True)
            if _retrain_notifier is not None:
                _retrain_notifier.show_failure.emit(str(e))

    t = threading.Thread(target=_run, daemon=True)
    t.start()


def _build_circular_tray_icon() -> QIcon:
    """Create a circular Core Sentinel tray icon."""
    size = 64
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setPen(QPen(QColor(220, 220, 220, 180), 2.0))
    p.setBrush(QColor(12, 37, 34, 245))
    p.drawEllipse(3, 3, size - 6, size - 6)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(2, 195, 154, 255))
    p.drawEllipse(10, 10, size - 20, size - 20)
    p.setPen(QColor(255, 255, 255))
    f = QFont("Segoe UI", 20)
    f.setBold(True)
    p.setFont(f)
    p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, "CS")
    p.end()
    return QIcon(pm)


if sys.platform == "win32":
    from win_paste_hook import (
        configure_paste_hook,
        start_paste_hook_threads,
        uninstall_paste_hook,
    )

_bubble_instance: RiskBubble | None = None


class PasteHookBridge(QObject):
    """Cross-thread delivery from the low-level keyboard hook worker to the Qt main thread."""

    scored = pyqtSignal(str, object, str, object, object)
    replay_only = pyqtSignal()

# Optional: log to clipboard_events.jsonl (set to a path to enable)
CLIPBOARD_EVENTS_JSONL = None  # or _guardrail_dir / "logs" / "clipboard_events.jsonl"

# Set to True to print when paste is detected (or when Ctrl+V is not in an LLM window)
DEBUG_PASTE = True

_MAX_PASTE_CHARS = 50_000


def _preprocess_paste(text: str | None) -> tuple[str | None, dict]:
    """
    Returns (processed_text, metadata_dict).
    processed_text is None when analysis should be skipped entirely.
    """
    if text is None:
        return None, {"skip_reason": "too_short"}
    if not isinstance(text, str):
        text = ""
    char_count = len(text)
    word_count = len(text.split())

    if char_count < 3 or not text.strip():
        return None, {"skip_reason": "too_short"}

    if char_count > _MAX_PASTE_CHARS:
        text = text[:_MAX_PASTE_CHARS]
        metadata = {
            "truncated_at": _MAX_PASTE_CHARS,
            "original_chars": char_count,
            "warning": "paste_too_large",
            "char_count": len(text),
            "word_count": len(text.split()),
            "estimated_chunks": max(1, len(text.split()) // 70),
        }
        return text, metadata

    return text, {
        "char_count": char_count,
        "word_count": word_count,
        "estimated_chunks": max(1, word_count // 70),
    }


def _log_skipped_paste_event(meta: dict) -> None:
    """Log that inference was skipped before the model (too short / empty)."""
    if DEBUG_PASTE:
        print(f"[Paste] Skipped (preprocess): {meta}", flush=True)
    path = CLIPBOARD_EVENTS_JSONL
    if not path:
        return
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "paste_skipped_preprocess",
            "meta": meta,
        }
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass


class InferenceWorker(QThread):
    """
    Runs score_clipboard_with_pii off the GUI thread. Main thread: create, connect, start(), return.

    Named result_ready (not \"finished\") so we do not shadow QThread.finished, which fires when
    the thread actually terminates (used for deleteLater).
    """

    # object carries a plain dict; pyqtSignal(dict) maps to QVariantMap and can corrupt/crash.
    result_ready = pyqtSignal(object)
    failed = pyqtSignal()

    def __init__(self, text: str, parent: QObject | None = None):
        super().__init__(parent)
        self.text = text

    def run(self) -> None:
        try:
            result = score_clipboard_with_pii(self.text)
            # Deep copy so nested spans/triggers are not shared with infer cache (clipboard clear
            # or later scoring must not empty the list backing the remediation panel).
            payload = copy.deepcopy(result) if isinstance(result, dict) else {}
            self.result_ready.emit(payload)
        except Exception as e:
            if DEBUG_PASTE:
                print(f"[Paste] Error: {e}", flush=True)
            self.failed.emit()


def check_feedback_and_maybe_retrain(bubble: RiskBubble | None) -> None:
    """Log feedback stats; if enough pending rows, offer to retrain (merge + train) in the background."""
    stats = get_feedback_stats()
    print(
        f"[feedback] total={stats['total']} "
        f"pending={stats['pending']} "
        f"corrections={stats['corrections']}",
        flush=True,
    )

    if should_trigger_retrain(min_count=_RETRAIN_TRIGGER_MIN):
        print(
            f"[feedback] {_RETRAIN_TRIGGER_MIN}+ pending corrections — retrain recommended",
            flush=True,
        )
        pending = int(stats.get("pending") or 0)
        mb = QMessageBox(bubble)
        mb.setIcon(QMessageBox.Icon.Question)
        mb.setWindowTitle("Retrain model?")
        mb.setText(
            f"You have {pending} corrections collected.\n"
            "Retrain the model now? (takes ~2 minutes)"
        )
        btn_yes = mb.addButton("Yes", QMessageBox.ButtonRole.YesRole)
        mb.addButton("Later", QMessageBox.ButtonRole.NoRole)
        mb.exec()
        if mb.clickedButton() == btn_yes:
            auto_retrain()
    else:
        print(
            f"[feedback] {stats['pending']} pending (need {_RETRAIN_TRIGGER_MIN} to trigger retrain)",
            flush=True,
        )


class _ModelWarmupThread(QThread):
    """Pre-loads the guardrail model so the first real paste is faster."""

    def run(self) -> None:
        try:
            score_clipboard_with_pii("warm up")
            if DEBUG_PASTE:
                print("[Guardrail] Model warm-up finished.", flush=True)
        except Exception as e:
            if DEBUG_PASTE:
                print(f"[Guardrail] Model warm-up failed (non-fatal): {e}", flush=True)


class _PasteAnalysisController(QObject):
    """Main thread: serial paste queue, starts QThread workers, updates bubble/tray/dialog."""

    def __init__(self, bubble: RiskBubble, tray: QSystemTrayIcon, parent: QObject | None = None):
        super().__init__(parent)
        self._bubble = bubble
        self._tray = tray
        self._pending: deque[tuple[str, str, object, object]] = deque()
        self._busy = False
        self._active_job: tuple[str, str, object, object] | None = None
        self._blocked_text: str = ""
        self._blocked_result: dict | None = None
        self._replay_sent = False
        self._safe_paste_count = 0
        self._char_event = None

    def _show_random_safe_character(self, score: int) -> None:
        from ui_risk_bubble import CharacterEvent

        emoji, title, base_sub = random.choice(_SAFE_MESSAGES)
        subtitle = f"{base_sub} — score {score}/100"
        w = CharacterEvent(
            emoji=emoji,
            title=title,
            subtitle=subtitle,
            color="#43A047",
            duration=2000,
        )
        w._position_near_bubble(self._bubble)
        self._char_event = w

    @pyqtSlot()
    def on_hook_replay_only(self) -> None:
        """Snooze / duplicate skip / scorer error: allow the suppressed paste through."""
        self._replay_sent = False
        if sys.platform != "win32":
            return
        try:
            from win_paste_hook import replay_suppressed_paste

            if not self._replay_sent:
                self._replay_sent = True
                replay_suppressed_paste()
        except Exception:
            pass

    @pyqtSlot(str, object, str, object, object)
    def on_hook_scored(
        self,
        text: str,
        result: object,
        agent_name: str,
        url,
        cleaned_title,
    ) -> None:
        """Main thread: scoring finished after hook suppressed Ctrl+V; replay if safe/warn."""
        self._replay_sent = False
        try:
            if not isinstance(result, dict):
                result = {}
            record_recent_text(text)
            try:
                rs = int(result.get("risk_score", 0))
            except (TypeError, ValueError):
                rs = 0
            record_scored_clipboard_risk_score(
                text, rs, risk_level=str(result.get("risk") or "")
            )
            action = self._normalize_action(result)
            # Replay suppressed Ctrl+V only for silent — warn replays in _dispatch_result after scoring.
            if action == "silent" and sys.platform == "win32":
                try:
                    from win_paste_hook import replay_suppressed_paste

                    if not self._replay_sent:
                        self._replay_sent = True
                        replay_suppressed_paste()
                except Exception:
                    pass
            self._dispatch_result(result, text, agent_name, url, cleaned_title)
        except Exception as e:
            if DEBUG_PASTE:
                import traceback

                traceback.print_exc()
                print(f"[Paste] Hook dispatch error: {e}", flush=True)
            self._bubble.set_idle()
            if sys.platform == "win32":
                try:
                    from win_paste_hook import replay_suppressed_paste

                    if not self._replay_sent:
                        self._replay_sent = True
                        replay_suppressed_paste()
                except Exception:
                    pass

    def on_paste_detected(
        self,
        text: str,
        agent_name: str,
        url,
        cleaned_title,
        *,
        ignore_pause: bool = False,
    ) -> None:
        """First step before inference: length limits, skip tiny/empty pastes, optional huge-paste warning."""
        if not ignore_pause and is_monitoring_paused():
            self._bubble.set_idle()
            return
        processed, meta = _preprocess_paste(text)
        if processed is None:
            _log_skipped_paste_event(meta)
            self._bubble.set_idle()
            return
        if meta.get("warning") == "paste_too_large":
            msg = "Large paste — scanning first 50,000 characters"
            if self._tray.isVisible():
                self._tray.showMessage(
                    "Clipboard Guardrail",
                    msg,
                    QSystemTrayIcon.MessageIcon.Information,
                    6000,
                )
            else:
                QToolTip.showText(
                    QCursor.pos(),
                    msg,
                    None,
                    QRect(),
                    6000,
                )
        # Clipboard poll + Ctrl+V can both fire before the first run finishes;
        # record_recent_text only runs after inference, so dedupe here to avoid
        # duplicate scores / telemetry and a second remediation pass after close.
        if self._active_job is not None and self._active_job[0] == processed:
            return
        if any(p[0] == processed for p in self._pending):
            return
        self._pending.append((processed, agent_name, url, cleaned_title))
        self._try_start_next()

    def enqueue(self, text: str, agent_name: str, url, cleaned_title) -> None:
        """Backward-compatible alias for on_paste_detected."""
        self.on_paste_detected(text, agent_name, url, cleaned_title)

    def _normalize_action(self, result: dict) -> str:
        critical = bool(result.get("critical_secret_detected", False))
        action = result.get("action")
        if action not in ("silent", "warn", "block"):
            decision = result.get("decision", "allow")
            action = {"allow": "silent", "warn": "warn", "block": "block"}.get(decision, "silent")
            if critical:
                action = "block"
        return action

    @staticmethod
    def _should_intercept_paste_hold(result: dict) -> bool:
        """High / critical paste hold: clear clipboard and require remediation before paste."""
        if not isinstance(result, dict):
            return False
        if bool(result.get("critical_secret_detected")):
            return True
        r = str(result.get("risk", "") or "").lower()
        if r in ("critical", "high", "h"):
            return True
        try:
            rs = int(result.get("risk_score", 0) or 0)
        except (TypeError, ValueError):
            rs = 0
        return rs > 70

    def _clear_paste_hold_state(self) -> None:
        self._blocked_text = ""
        self._blocked_result = None
        self._bubble.set_hold_state(False)

    def _clear_clipboard_for_hold(self) -> None:
        try:
            import pyperclip

            pyperclip.copy("")
        except Exception:
            pass

    def _show_warn_panel(
        self,
        text: str,
        result: dict,
        agent_name: str,
        url,
        cleaned_title,
    ) -> None:
        """Open remediation in review mode: paste already replayed; panel is non-blocking."""
        try:
            sp = result.get("spans")
            spans_list = sp if isinstance(sp, list) else []
            try:
                sc = int(result.get("risk_score", 0) or 0)
            except (TypeError, ValueError):
                sc = 0
            rk = str(result.get("risk", "low") or "low")
            dialog = RemediationDialog(
                text,
                result,
                agent_name,
                self._bubble,
                hold_mode=False,
                critical_hold=False,
            )
            dialog.monitoring_toggled.connect(
                lambda paused: self.kick_queue() if not paused else None
            )
            dialog.remediation_finished.connect(lambda _ok: None)
            dialog.finished.connect(lambda _code: None)
            dialog.show_with_warn_mode(spans=spans_list, score=sc, risk=rk)
        except Exception as e:
            if DEBUG_PASTE:
                print(f"[Paste] Warn panel error: {e}", flush=True)

    def _dispatch_result(
        self,
        result: dict,
        text: str,
        agent_name: str,
        url,
        cleaned_title,
    ) -> None:
        action = self._normalize_action(result)
        self._bubble._update_streak_from_action(action)
        score = result.get("risk_score", 0)
        critical = bool(result.get("critical_secret_detected", False))
        if DEBUG_PASTE and action != "silent":
            print(
                f"[Paste] Score={score} action={action!r} critical_secret={critical!r}",
                flush=True,
            )

        if action == "silent":
            try:
                sc = int(result.get("risk_score", 0) or 0)
            except (TypeError, ValueError):
                sc = 0
            sc = max(0, min(100, sc))
            if sys.platform == "win32":
                try:
                    from win_paste_hook import replay_suppressed_paste

                    if not self._replay_sent:
                        self._replay_sent = True
                        replay_suppressed_paste()
                except Exception:
                    pass
            self._bubble.update_from_result(
                result, agent_name or "LLM", text, url=url, cleaned_title=cleaned_title
            )
            if sc <= 40:
                streak = getattr(self._bubble, "_streak", {}) or {}
                count = int(streak.get("streak_count", 0) or 0)
                if count not in _STREAK_MILESTONES:
                    self._safe_paste_count += 1
                    if self._safe_paste_count % 3 == 1:
                        QTimer.singleShot(
                            400,
                            lambda s=sc: self._show_random_safe_character(s),
                        )
                if sc > 20 and count not in _STREAK_MILESTONES:
                    toast_safe(f"Safe — score {sc}/100")
            return

        if action == "warn":
            self._bubble.update_from_result(
                result, agent_name or "LLM", text, url=url, cleaned_title=cleaned_title
            )
            if is_guard_snoozed():
                print(
                    "[Silent Guard] Snooze active; warn remediation suppressed.",
                    flush=True,
                )
                return
            if sys.platform == "win32":
                try:
                    from win_paste_hook import replay_suppressed_paste

                    if not self._replay_sent:
                        self._replay_sent = True
                        replay_suppressed_paste()
                except Exception:
                    pass
            res_copy = dict(result)
            ag = agent_name or "LLM"
            QTimer.singleShot(
                300,
                lambda t=text, r=res_copy, a=ag, u=url, ct=cleaned_title: self._show_warn_panel(
                    t, r, a, u, ct
                ),
            )
            return

        if action == "block":
            if user_settings.load().get("play_sound_on_block"):
                try:
                    import winsound

                    winsound.MessageBeep(winsound.MB_ICONHAND)
                except Exception:
                    pass
            self._bubble.update_from_result(
                result, agent_name or "LLM", text, url=url, cleaned_title=cleaned_title
            )
            if is_guard_snoozed() and not critical:
                print(
                    "[Silent Guard] Snooze active; remediation dialog suppressed "
                    "(analysis already completed).",
                    flush=True,
                )
                return
            hold_paste = self._should_intercept_paste_hold(result)
            if hold_paste:
                self._blocked_text = text
                self._blocked_result = result
                self._bubble.set_hold_state(True, critical=critical)
                dialog = RemediationDialog(
                    text,
                    result,
                    agent_name or "LLM",
                    self._bubble,
                    hold_mode=True,
                    critical_hold=critical,
                )
            else:
                dialog = RemediationDialog(text, result, agent_name or "LLM", None)
            dialog.monitoring_toggled.connect(
                lambda paused: self.kick_queue() if not paused else None
            )
            dialog.remediation_finished.connect(lambda _ok: self._clear_paste_hold_state())
            dialog.finished.connect(lambda _code: self._clear_paste_hold_state())
            if hold_paste:
                sp = result.get("spans")
                spans_list = sp if isinstance(sp, list) else []
                try:
                    sc = int(result.get("risk_score", 0) or 0)
                except (TypeError, ValueError):
                    sc = 0
                dialog.show_with_hold_mode(
                    spans=spans_list,
                    score=sc,
                    risk=str(result.get("risk", "low")),
                    critical=critical,
                    original_text=text,
                )
                QTimer.singleShot(50, self._clear_clipboard_for_hold)
            else:
                dialog.show()

    @pyqtSlot(object)
    def on_inference_done(self, result: object) -> None:
        """Main-thread slot: InferenceWorker.result_ready (dict payload)."""
        job = self._active_job
        self._active_job = None
        text = agent_name = ""
        url = cleaned_title = None
        if job is not None:
            text, agent_name, url, cleaned_title = job
        try:
            if not isinstance(result, dict):
                result = {}
            record_recent_text(text)
            try:
                rs = int(result.get("risk_score", 0))
            except (TypeError, ValueError):
                rs = 0
            record_scored_clipboard_risk_score(
                text, rs, risk_level=str(result.get("risk") or "")
            )
            self._busy = False
            self._dispatch_result(result, text, agent_name, url, cleaned_title)
        except Exception as e:
            if DEBUG_PASTE:
                import traceback

                traceback.print_exc()
                print(f"[Paste] UI dispatch error: {e}", flush=True)
            self._bubble.set_idle()
            self._busy = False
        finally:
            self._try_start_next()

    @pyqtSlot()
    def _on_inference_failed(self) -> None:
        self._active_job = None
        self._busy = False
        self._bubble.set_idle()
        self._try_start_next()

    def kick_queue(self) -> None:
        """Resume processing after unpause (or manual kick)."""
        self._try_start_next()

    def _try_start_next(self) -> None:
        if is_monitoring_paused():
            return
        if self._busy or not self._pending:
            return
        self._busy = True
        text, agent_name, url, cleaned_title = self._pending.popleft()
        self._active_job = (text, agent_name, url, cleaned_title)
        # Queued pastes: previous job may have left clear/issues UI; show analysing again.
        self._bubble.set_analysing(text)

        worker = InferenceWorker(text, self)
        worker.result_ready.connect(self.on_inference_done, Qt.ConnectionType.QueuedConnection)
        worker.failed.connect(self._on_inference_failed, Qt.ConnectionType.QueuedConnection)
        worker.finished.connect(worker.deleteLater)
        worker.start()


def on_paste_event(text: str):
    """Reserved; paste interception uses win_paste_hook on Windows."""
    del text


def _on_clipboard_preview_impl(text: str, result: dict, bubble: RiskBubble) -> None:
    """Background clipboard poll: update traffic-light preview when not in an LLM window."""
    if is_monitoring_paused():
        return
    try:
        score = int(result.get("risk_score", 0) or 0)
    except (TypeError, ValueError):
        score = 0
    action = str(result.get("action", "silent"))
    try:
        from active_window_llm import is_llm_window

        in_llm = bool(is_llm_window())
    except Exception:
        in_llm = False
    if in_llm:
        if action != "silent":
            try:
                from document_scanner import generate_paste_report

                generate_paste_report(text, result)
            except Exception:
                pass
        return
    if action == "block":
        bubble.set_clipboard_preview("high", score)
    elif action == "warn":
        bubble.set_clipboard_preview("med", score)
    else:
        bubble.set_clipboard_preview("safe", score)


def main():
    app = QApplication(sys.argv)
    install_qt_message_filter()
    app.setProperty("paste_controller", None)

    user_settings.apply_to_environment_and_runtime()
    normalize_application_font(app)

    preload_ok = False
    blocking = os.environ.get("GUARDRAIL_BLOCKING_PRELOAD", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )
    if blocking:
        try:
            preload_guardrail_model()
            preload_ok = True
            if DEBUG_PASTE:
                print("[Guardrail] Model preloaded (ready for paste).", flush=True)
        except Exception as e:
            print(f"[Guardrail] Model preload failed (will load on first paste): {e}", flush=True)

    start_sync_thread()

    global _bubble_instance
    if _bubble_instance is None:
        _bubble_instance = RiskBubble()
    bubble = _bubble_instance
    bubble.menu_action_clicked.connect(bubble.handle_menu_action)
    bubble.set_show_badge_on_issues(
        bool(user_settings.load().get("show_badge_on_issues", True))
    )

    tray = QSystemTrayIcon(app)
    tray.setIcon(_build_circular_tray_icon())
    tray.setToolTip("Core Sentinel Guardrail")
    if QSystemTrayIcon.isSystemTrayAvailable():
        tray.show()

    paste_controller = _PasteAnalysisController(bubble, tray, app)
    app.setProperty("paste_controller", paste_controller)

    if sys.platform == "win32":
        _paste_hook_bridge = PasteHookBridge(app)
        configure_paste_hook(_paste_hook_bridge.scored.emit, _paste_hook_bridge.replay_only.emit)
        _paste_hook_bridge.scored.connect(
            paste_controller.on_hook_scored, Qt.ConnectionType.QueuedConnection
        )
        _paste_hook_bridge.replay_only.connect(
            paste_controller.on_hook_replay_only, Qt.ConnectionType.QueuedConnection
        )
        start_paste_hook_threads()
        app.aboutToQuit.connect(uninstall_paste_hook)

    def _on_bubble_pause(checked: bool) -> None:
        set_monitoring_paused(checked)
        if checked:
            bubble.set_idle()
        else:
            paste_controller.kick_queue()

    def _open_sentinel_settings() -> None:
        from ui_sentinel_settings import SentinelSettingsDialog

        dlg = SentinelSettingsDialog(bubble)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        d = user_settings.load()
        bubble.set_show_badge_on_issues(bool(d.get("show_badge_on_issues", True)))

    def _view_report() -> None:
        try:
            root = Path(__file__).resolve().parent
            paste_reports = root / "logs" / "paste_reports"
            latest_html: Path | None = None
            if paste_reports.exists():
                html_files = sorted(
                    paste_reports.glob("*.html"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if html_files:
                    latest_html = html_files[0]
            if latest_html is None:
                fallback = root / "logs" / "test_report.html"
                if fallback.exists():
                    latest_html = fallback
            if latest_html is not None:
                webbrowser.open(latest_html.resolve().as_uri())
                return
            # Last fallback: generate CSV export.
            p = export_scoring_events_csv()
            webbrowser.open(p.resolve().as_uri())
        except OSError as e:
            show_toast(f"View report: {e}", parent=bubble, color="#E53935")

    def _on_context_menu(action: str) -> None:
        if action == "settings":
            _open_sentinel_settings()
        elif action == "scan_file":
            bubble._do_scan_file()
        elif action == "view_report":
            _view_report()
        elif action == "quit":
            app.quit()

    # Bubble tray/context menu is built in ui_risk_bubble.RiskBubble._show_context_menu
    # (📊 My Privacy Dashboard + separator before Quit).
    bubble.context_menu_action.connect(_on_context_menu)
    bubble.monitoring_pause_changed.connect(_on_bubble_pause)

    global _retrain_notifier
    _retrain_notifier = _RetrainNotifier(bubble)
    _retrain_notifier.show_success.connect(
        lambda: show_toast(
            "Core Sentinel has learned from your corrections",
            title="Model retrained successfully",
            parent=bubble,
        )
    )
    _retrain_notifier.show_failure.connect(
        lambda msg: show_toast(
            f"Retrain failed: {msg}",
            color="#E53935",
            duration=8000,
            parent=bubble,
            title="Retrain error",
        )
    )

    _s = QShortcut(QKeySequence("Alt+G"), bubble)
    _s.activated.connect(bubble._toggle_monitoring)
    _s = QShortcut(QKeySequence("Alt+R"), bubble)
    _s.activated.connect(bubble._do_rephrase)
    _s = QShortcut(QKeySequence("Alt+D"), bubble)
    _s.activated.connect(bubble._do_redact)
    _s = QShortcut(QKeySequence("Alt+E"), bubble)
    _s.activated.connect(bubble._do_encrypt)
    _s = QShortcut(QKeySequence("Alt+S"), bubble)
    _s.activated.connect(bubble._do_scan_file)
    _s = QShortcut(QKeySequence("Alt+,"), bubble)
    _s.activated.connect(bubble._do_open_settings)

    bubble.show()
    bubble.raise_()
    # Defer raise so the pill appears above other always-on-top tools after the event loop runs.
    QTimer.singleShot(0, bubble.bring_to_front)
    QTimer.singleShot(400, bubble.bring_to_front)

    from clipboard_monitor import ClipboardMonitor

    _clip_monitor = ClipboardMonitor()
    _clip_monitor.clipboard_changed.connect(
        lambda t, r, b=bubble: _on_clipboard_preview_impl(t, r, b),
        Qt.ConnectionType.QueuedConnection,
    )
    _clip_monitor.start()

    if not preload_ok:
        warmup = _ModelWarmupThread(app)
        warmup.finished.connect(warmup.deleteLater)
        warmup.start()

    print(
        "Risk-Aware Assistant running. "
        + (
            "Ctrl+V is intercepted in LLM windows before paste; text is scored then released or blocked."
            if sys.platform == "win32"
            else "Paste hook is Windows-only; run on Windows for full protection."
        )
    )

    def _schedule_feedback_check() -> None:
        QTimer.singleShot(3000, lambda: check_feedback_and_maybe_retrain(bubble))

    if preload_ok:
        _schedule_feedback_check()
    else:

        def _on_warmup_done() -> None:
            _schedule_feedback_check()

        warmup.finished.connect(_on_warmup_done)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
