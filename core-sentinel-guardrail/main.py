"""
Risk-Aware Assistant: PyQt6 UI driven by clipboard paste into LLMs (tray + RemediationDialog).
In-app toast notifications replace modal message boxes for errors and notices.

Usage:
  python main.py

Flow:
  - Keyboard hook runs in a background thread.
  - On Ctrl+V: is_active_window_llm() first; if not LLM, no scoring/logging/UI (see GUARDRAIL_DEBUG_WINDOW).
  - If active window is an LLM, we read clipboard; duplicate pastes (same text as one of
    the last 3) skip scoring. Otherwise a \"__paste__\" job is queued for the GUI thread.
  - score_clipboard_with_pii runs only on InferenceWorker (QThread); result_ready / failed are
    handled on the main thread (bubble + tray/dialog). By default infer.preload_guardrail_model()
    runs before the bubble is shown so the first paste pays inference only (not load). Set
    GUARDRAIL_BLOCKING_PRELOAD=0 to skip that and use background warm-up instead. Paste queue
    is polled every ~25ms. The pill spinner runs on the main thread.
  - Floating pill (RiskBubble): Grammarly-style UI; monitoring refreshed ~1s; optional Win32 anchor to focused edit.
  - While an LLM window is focused, a QTimer periodically reads the system clipboard (default every
    2s; GUARDRAIL_CLIPBOARD_POLL_MS, 0 disables) and enqueues analysis when the text changed, so
    risky copies are flagged before paste.
  - Main thread (QTimer) polls the queue and uses result[\"action\"] from the scorer:
    silent — updates pill only;
    warn — QSystemTrayIcon.showMessage or non-blocking QToolTip (no modal);
    block — non-modal RemediationDialog (slides in from right; REDACT/HASH/ENCRYPT/PROCEED + optional snooze).
"""
from __future__ import annotations

import os as _os
_torch_lib = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.venv/Lib/site-packages/torch/lib")
if _os.path.isdir(_torch_lib) and hasattr(_os, "add_dll_directory"):
    _os.add_dll_directory(_os.path.abspath(_torch_lib))
del _os, _torch_lib




import os as _os


import json
import os
import queue
import subprocess
import sys
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

_guardrail_dir = Path(__file__).resolve().parent
if str(_guardrail_dir) not in sys.path:
    sys.path.insert(0, str(_guardrail_dir))

from PyQt6.QtCore import QObject, QRect, QThread, Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QCursor, QGuiApplication, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QStyle,
    QSystemTrayIcon,
    QToolTip,
)

from active_window_llm import (
    get_foreground_app_label,
    is_active_window_llm,
    log_guardrail_active_window_banner,
)
from guardrail_runtime import (
    get_monitor_llm_only,
    is_guard_snoozed,
    is_monitoring_paused,
    is_recent_duplicate,
    record_recent_text,
    record_scored_clipboard_risk_score,
    set_monitoring_paused,
)
from feedback_store import get_feedback_stats, should_trigger_retrain
from font_clamp import normalize_application_font
import user_settings
from guardrail_logs import export_scoring_events_csv, log_clipboard_event
from infer import (
    preload_guardrail_model,
    score_clipboard_with_pii,
    should_bypass_duplicate_skip_for_text,
)
from ui_remediation_dialog import RemediationDialog
from ui_risk_bubble import RiskBubble

from toast import show_toast

_bubble_instance: RiskBubble | None = None

# Optional: log to clipboard_events.jsonl (set to a path to enable)
CLIPBOARD_EVENTS_JSONL = None  # or _guardrail_dir / "logs" / "clipboard_events.jsonl"

# Queue: ("__paste__", text, agent_name, url_or_none, cleaned_title_or_none) from listener -> main
_paste_queue = queue.Queue()
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
            payload = dict(result) if isinstance(result, dict) else {}
            self.result_ready.emit(payload)
        except Exception as e:
            if DEBUG_PASTE:
                print(f"[Paste] Error: {e}", flush=True)
            self.failed.emit()


def check_feedback_and_maybe_retrain(tray: QSystemTrayIcon | None) -> None:
    """Log feedback stats; if enough pending rows, notify user to run train.py (no auto-retrain)."""
    stats = get_feedback_stats()
    print(
        f"[feedback] total={stats['total']} "
        f"pending={stats['pending']} "
        f"corrections={stats['corrections']}",
        flush=True,
    )

    if should_trigger_retrain(min_count=30):
        print(
            "[feedback] 30+ pending corrections — retrain recommended",
            flush=True,
        )
        if tray is not None and QSystemTrayIcon.isSystemTrayAvailable():
            tray.showMessage(
                "Guardrail update available",
                f"{stats['pending']} corrections collected. Run train.py to retrain.",
                QSystemTrayIcon.MessageIcon.Information,
                5000,
            )
    else:
        print(
            f"[feedback] {stats['pending']} pending (need 30 to trigger retrain)",
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

    def _dispatch_result(
        self,
        result: dict,
        text: str,
        agent_name: str,
        url,
        cleaned_title,
    ) -> None:
        action = self._normalize_action(result)
        score = result.get("risk_score", 0)
        critical = bool(result.get("critical_secret_detected", False))
        if DEBUG_PASTE and action != "silent":
            print(
                f"[Paste] Score={score} action={action!r} critical_secret={critical!r}",
                flush=True,
            )

        if action == "silent":
            self._bubble.update_from_result(
                result, agent_name or "LLM", text, url=url, cleaned_title=cleaned_title
            )
            return

        if action == "warn":
            self._bubble.update_from_result(
                result, agent_name or "LLM", text, url=url, cleaned_title=cleaned_title
            )
            if is_guard_snoozed():
                print(
                    "[Silent Guard] Snooze active; warn tray notification suppressed.",
                    flush=True,
                )
                return
            msg = (result.get("message") or "Review clipboard before pasting.").replace("\n", " ")
            if self._tray.isVisible():
                self._tray.showMessage(
                    "Clipboard Guardrail",
                    msg[:256],
                    QSystemTrayIcon.MessageIcon.Warning,
                    8000,
                )
            else:
                QToolTip.showText(
                    QCursor.pos(),
                    msg[:500],
                    None,
                    QRect(),
                    8000,
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
            dialog = RemediationDialog(text, result, agent_name or "LLM", None)
            dialog.monitoring_toggled.connect(
                lambda paused: self.kick_queue() if not paused else None
            )
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


def _listener_thread_fn():
    """Run keyboard hook; on Ctrl+V in LLM window, score and push to queue. Always allow paste."""
    import keyboard

    try:
        import pyperclip
    except Exception:
        return

    def on_key(event):
        if getattr(event, "event_type", None) != "down":
            return True
        if getattr(event, "name", None) != "v":
            return True
        try:
            if not keyboard.is_pressed("ctrl"):
                return True
            if is_monitoring_paused():
                return True
            is_llm, agent_name, url, cleaned_title = is_active_window_llm(debug=False)
            banner_label = agent_name if is_llm else get_foreground_app_label()
            log_guardrail_active_window_banner(is_llm, banner_label)
            if get_monitor_llm_only() and not is_llm:
                return True
            text = pyperclip.paste()
            if not text or not isinstance(text, str):
                return True
            text = text.strip()
            if not text:
                return True
            if is_recent_duplicate(text) and not should_bypass_duplicate_skip_for_text(text):
                if DEBUG_PASTE:
                    print("[Silent Guard] Duplicate paste text, skipping analysis.", flush=True)
                return True
            _paste_queue.put(("__paste__", text, agent_name, url, cleaned_title))
        except Exception as e:
            if DEBUG_PASTE:
                print(f"[Paste] Error: {e}", flush=True)
        return True

    keyboard.hook(on_key)
    keyboard.wait()


def on_paste_event(text: str):
    """
    Reserved hook; paste handling uses _PasteAnalysisController.on_paste_detected from
    process_paste_queue / clipboard poll.
    """
    del text


def _start_clipboard_poll_timer(
    bubble: RiskBubble,
    paste_controller: _PasteAnalysisController,
    parent: QObject,
) -> QTimer | None:
    """
    When the focused window is an LLM, poll clipboard on an interval; enqueue scoring if text
    changed (so users get warnings before Ctrl+V). GUARDRAIL_CLIPBOARD_POLL_MS=0 disables.
    """
    raw = os.environ.get("GUARDRAIL_CLIPBOARD_POLL_MS", "2000").strip()
    try:
        interval_ms = int(raw)
    except ValueError:
        interval_ms = 2000
    if interval_ms <= 0:
        return None

    last_clip_text: list[str | None] = [None]

    def tick() -> None:
        try:
            if is_monitoring_paused():
                return
            is_llm, agent_name, url, cleaned_title = is_active_window_llm(debug=False)
            if get_monitor_llm_only() and not is_llm:
                return
            t = QGuiApplication.clipboard().text()
            if not t or not isinstance(t, str):
                return
            t = t.strip()
            if not t:
                return
            if last_clip_text[0] == t:
                return
            last_clip_text[0] = t
            if is_recent_duplicate(t) and not should_bypass_duplicate_skip_for_text(t):
                return
            bubble.set_analysing(t)
            paste_controller.on_paste_detected(t, agent_name or "", url, cleaned_title)
        except Exception as e:
            if DEBUG_PASTE:
                print(f"[Clipboard poll] {e}", flush=True)

    poll = QTimer(parent)
    poll.timeout.connect(tick)
    poll.start(interval_ms)
    return poll


def main():
    app = QApplication(sys.argv)
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

    global _bubble_instance
    if _bubble_instance is None:
        _bubble_instance = RiskBubble()
    bubble = _bubble_instance
    bubble.menu_action_clicked.connect(bubble.handle_menu_action)
    bubble.set_show_badge_on_issues(
        bool(user_settings.load().get("show_badge_on_issues", True))
    )

    tray = QSystemTrayIcon(app)
    tray.setIcon(app.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning))
    tray.setToolTip("Core Sentinel Guardrail")
    if QSystemTrayIcon.isSystemTrayAvailable():
        tray.show()

    paste_controller = _PasteAnalysisController(bubble, tray, app)
    app.setProperty("paste_controller", paste_controller)

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
            p = export_scoring_events_csv()
            subprocess.Popen(
                ["notepad.exe", str(p.resolve())],
                shell=False,
            )
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

    bubble.context_menu_action.connect(_on_context_menu)
    bubble.monitoring_pause_changed.connect(_on_bubble_pause)

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

    if not preload_ok:
        warmup = _ModelWarmupThread(app)
        warmup.finished.connect(warmup.deleteLater)
        warmup.start()

    def process_paste_queue():
        try:
            while True:
                kind, a, b, c, d = _paste_queue.get_nowait()
                if kind == "__paste__":
                    if is_monitoring_paused():
                        continue
                    # Main-thread first response to paste (listener thread only enqueues).
                    bubble.set_analysing(a)
                    paste_controller.on_paste_detected(a, b, c, d)
        except queue.Empty:
            pass

    timer = QTimer()
    timer.timeout.connect(process_paste_queue)
    timer.start(25)

    _start_clipboard_poll_timer(bubble, paste_controller, app)

    thread = threading.Thread(target=_listener_thread_fn, daemon=True)
    thread.start()

    print(
        "Risk-Aware Assistant running. Clipboard is scanned periodically while an LLM is focused; "
        "Ctrl+V still triggers analysis. Close this window to exit."
    )

    def _schedule_feedback_check() -> None:
        QTimer.singleShot(3000, lambda: check_feedback_and_maybe_retrain(tray))

    if preload_ok:
        _schedule_feedback_check()
    else:

        def _on_warmup_done() -> None:
            _schedule_feedback_check()

        warmup.finished.connect(_on_warmup_done)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
