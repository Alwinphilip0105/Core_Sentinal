"""
Background clipboard poller: scores clipboard periodically for traffic-light preview (non-blocking).
"""

from __future__ import annotations

import hashlib
import threading
import time

import pyperclip
from PyQt6.QtCore import QObject, pyqtSignal

from guardrail_runtime import is_monitoring_paused


class ClipboardMonitor(QObject):
    """Signals (text, result) to main thread — use QueuedConnection from worker thread."""

    clipboard_changed = pyqtSignal(str, dict)

    def __init__(self, bubble_ref=None):
        super().__init__()
        self._running = False
        self._last_text = ""
        self._last_hash = ""
        self._bubble = bubble_ref
        self._interval = 2.0
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="clipboard-monitor",
        )
        self._thread.start()
        print(
            "[monitor] Clipboard monitor started — polling every 2s",
            flush=True,
        )

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        while self._running:
            try:
                if is_monitoring_paused():
                    time.sleep(self._interval)
                    continue
                if self._bubble is not None and bool(getattr(self._bubble, "_analysing", False)):
                    time.sleep(self._interval)
                    continue
                try:
                    from active_window_llm import is_llm_window

                    if bool(is_llm_window()):
                        time.sleep(self._interval)
                        continue
                except Exception:
                    pass

                text = pyperclip.paste()
                if not text or len(text.strip()) < 3:
                    time.sleep(self._interval)
                    continue

                h = hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()
                if h == self._last_hash:
                    time.sleep(self._interval)
                    continue

                self._last_hash = h
                self._last_text = text

                from infer import score_clipboard_with_pii

                started = time.perf_counter()
                result = score_clipboard_with_pii(text)
                score = int(result.get("risk_score", 0) or 0)
                action = str(result.get("action", "silent"))
                elapsed_ms = (time.perf_counter() - started) * 1000.0

                print(
                    f"[monitor] Clipboard changed score={score} action={action} dt_ms={elapsed_ms:.1f}",
                    flush=True,
                )

                self.clipboard_changed.emit(text, result)

            except Exception:
                pass

            time.sleep(self._interval)
