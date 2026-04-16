"""
Windows low-level keyboard hook (WH_KEYBOARD_LL): suppress Ctrl+V before the target app sees it,
score clipboard text on a worker thread, then replay paste (safe) or block and show remediation.

Requires a dedicated thread running a Win32 message loop (GetMessage).
"""

from __future__ import annotations

import copy
import ctypes
import os
import queue
import sys
import traceback
import threading
import time
from ctypes import CFUNCTYPE, POINTER, c_int
from ctypes import wintypes as wt

_replay_lock = threading.Lock()
_suppress_next_v = False
_capture_lock = threading.Lock()
_job_inflight = False
_last_captured_text = ""

if sys.platform != "win32":
    # Stubs for non-Windows (imports only)
    _hook_handle = None
    _hook_proc = None

    def configure_paste_hook(_emit_scored, _emit_replay) -> None:
        pass

    def start_paste_hook_threads() -> None:
        pass

    def uninstall_paste_hook() -> None:
        pass

    def replay_suppressed_paste() -> bool:
        return False

else:
    windll = ctypes.windll

    # Fix 64-bit pointer overflow in hook callbacks (lParam must be c_void_p, not LPARAM).
    _user32 = windll.user32
    _user32.SetWindowsHookExW.restype = ctypes.c_void_p
    _user32.SetWindowsHookExW.argtypes = [
        c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        wt.DWORD,
    ]
    _user32.CallNextHookEx.restype = ctypes.c_long
    _user32.CallNextHookEx.argtypes = [
        ctypes.c_void_p,
        c_int,
        wt.WPARAM,
        ctypes.c_void_p,
    ]
    _user32.UnhookWindowsHookEx.restype = wt.BOOL
    _user32.UnhookWindowsHookEx.argtypes = [ctypes.c_void_p]
    _user32.GetAsyncKeyState.restype = ctypes.c_short
    _user32.GetAsyncKeyState.argtypes = [c_int]

    ULONG_PTR = ctypes.c_uint64 if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_uint32

    WH_KEYBOARD_LL = 13
    WM_KEYDOWN = 0x0100
    WM_SYSKEYDOWN = 0x0104
    WM_QUIT = 0x0012
    VK_V = 0x56
    VK_CONTROL = 0x11
    HC_ACTION = 0
    KEYEVENTF_KEYUP = 0x0002

    class KBDLLHOOKSTRUCT(ctypes.Structure):
        _fields_ = [
            ("vkCode", wt.DWORD),
            ("scanCode", wt.DWORD),
            ("flags", wt.DWORD),
            ("time", wt.DWORD),
            ("dwExtraInfo", ULONG_PTR),
        ]

    HOOKPROC = CFUNCTYPE(
        ctypes.c_long,
        c_int,
        wt.WPARAM,
        ctypes.c_void_p,
    )

    _hook_handle = None
    _hook_proc: HOOKPROC | None = None
    _hook_thread_id: int = 0

    _emit_scored = None
    _emit_replay = None

    _score_queue: queue.Queue[tuple[str, str, object, object] | None] = queue.Queue()

    _score_worker_thread: threading.Thread | None = None
    _hook_loop_thread: threading.Thread | None = None
    _stop_score_worker = threading.Event()

    def configure_paste_hook(emit_scored, emit_replay) -> None:
        """Set callbacks: emit_scored(text, result_dict, agent, url, title), emit_replay() for snooze/duplicate/fail-safe."""
        global _emit_scored, _emit_replay
        _emit_scored = emit_scored
        _emit_replay = emit_replay

    def replay_suppressed_paste() -> bool:
        """
        Replay the suppressed Ctrl+V.
        Returns True if replay was sent, False if already replaying (prevents double paste).
        """
        global _suppress_next_v

        if not _replay_lock.acquire(blocking=False):
            print("[hook] replay already in progress — skipping", flush=True)
            return False

        try:
            _suppress_next_v = True
            time.sleep(0.06)
            u32 = windll.user32
            u32.keybd_event(VK_CONTROL, 0, 0, 0)
            u32.keybd_event(VK_V, 0, 0, 0)
            u32.keybd_event(VK_V, 0, KEYEVENTF_KEYUP, 0)
            u32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)
            print("[hook] replay sent", flush=True)
            return True
        finally:
            time.sleep(0.2)
            _replay_lock.release()

    def _keyboard_hook(nCode, wParam, lParam):
        global _hook_handle, _suppress_next_v
        _lp = ctypes.c_void_p(lParam)
        if nCode < HC_ACTION:
            return _user32.CallNextHookEx(_hook_handle, nCode, wParam, _lp)

        if wParam not in (WM_KEYDOWN, WM_SYSKEYDOWN):
            return _user32.CallNextHookEx(_hook_handle, nCode, wParam, _lp)

        kb = ctypes.cast(_lp, POINTER(KBDLLHOOKSTRUCT)).contents
        ctrl_held = (_user32.GetAsyncKeyState(VK_CONTROL) & 0x8000) != 0

        if kb.vkCode != VK_V or not ctrl_held:
            return _user32.CallNextHookEx(_hook_handle, nCode, wParam, _lp)

        if _suppress_next_v:
            _suppress_next_v = False
            print("[hook] suppress flag — allowing replay", flush=True)
            return _user32.CallNextHookEx(_hook_handle, nCode, wParam, _lp)

        if _replay_lock.locked():
            # During synthetic replay, swallow extra physical key-repeat Ctrl+V events.
            # Passing them through can produce duplicate pastes in search fields.
            print("[hook] replay in progress — suppressing extra Ctrl+V", flush=True)
            return 1

        try:
            from active_window_llm import (
                detect_llm_window,
                get_foreground_app_label,
                llm_debug_enabled,
                log_guardrail_active_window_banner,
            )
            from guardrail_runtime import get_monitor_llm_only, is_monitoring_paused

            if is_monitoring_paused():
                return _user32.CallNextHookEx(
                    _hook_handle, nCode, wParam, ctypes.c_void_p(lParam)
                )

            # TEMP: disable LLM-only filter for testing — set GUARDRAIL_ALL_WINDOWS=1
            _force_all_windows = os.environ.get("GUARDRAIL_ALL_WINDOWS", "0") == "1"

            is_llm, agent_name, url, cleaned_title = detect_llm_window(
                debug=llm_debug_enabled()
            )

            banner_label = agent_name if is_llm else get_foreground_app_label()
            log_guardrail_active_window_banner(is_llm, banner_label)

            if get_monitor_llm_only() and not is_llm and not _force_all_windows:
                return _user32.CallNextHookEx(
                    _hook_handle, nCode, wParam, ctypes.c_void_p(lParam)
                )

            try:
                import pyperclip

                text = pyperclip.paste()
            except Exception:
                text = ""
            if not text or not isinstance(text, str) or not text.strip():
                return _user32.CallNextHookEx(
                    _hook_handle, nCode, wParam, ctypes.c_void_p(lParam)
                )

            text = text.strip()
            # Coalesce repeated Ctrl+V presses while one paste decision is in-flight.
            # This avoids queueing duplicate jobs from key repeat / long key holds.
            with _capture_lock:
                global _job_inflight, _last_captured_text
                if _job_inflight:
                    print("[hook] paste already in-flight — coalescing duplicate Ctrl+V", flush=True)
                    return 1
                _job_inflight = True
                _last_captured_text = text
            try:
                _score_queue.put_nowait((text, agent_name or "", url, cleaned_title))
            except Exception:
                with _capture_lock:
                    _job_inflight = False
                return _user32.CallNextHookEx(
                    _hook_handle, nCode, wParam, ctypes.c_void_p(lParam)
                )
            return 1
        except Exception as e:
            # Swallows errors so the hook never crashes the process; log when debugging.
            _dbg = os.environ.get("GUARDRAIL_DEBUG_HOOK_ERRORS", "").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            ) or os.environ.get("GUARDRAIL_DEBUG_LLM_DETECT", "").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            if _dbg:
                print(f"[paste-hook] Ctrl+V handler error: {e}", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)
            return _user32.CallNextHookEx(
                _hook_handle, nCode, wParam, ctypes.c_void_p(lParam)
            )

    _MAX_PASTE_CHARS = 50_000

    def _score_one_job(text: str, agent_name: str, url, cleaned_title) -> None:
        from guardrail_runtime import is_guard_snoozed, is_recent_duplicate
        from infer import score_clipboard_with_pii, should_bypass_duplicate_skip_for_text

        try:
            if _emit_replay is None or _emit_scored is None:
                return

            if len(text.strip()) < 3:
                _emit_replay()
                return
            if len(text) > _MAX_PASTE_CHARS:
                text = text[:_MAX_PASTE_CHARS]

            if is_guard_snoozed():
                _emit_replay()
                return

            if is_recent_duplicate(text) and not should_bypass_duplicate_skip_for_text(text):
                _emit_replay()
                return

            try:
                result = score_clipboard_with_pii(text)
                payload = copy.deepcopy(result) if isinstance(result, dict) else {}
            except Exception:
                _emit_replay()
                return

            _emit_scored(text, payload, agent_name, url, cleaned_title)
        finally:
            # Always release coalescing gate after a decision path is emitted.
            with _capture_lock:
                global _job_inflight
                _job_inflight = False

    def _score_worker_loop() -> None:
        while not _stop_score_worker.is_set():
            try:
                job = _score_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if job is None:
                break
            text, agent, url, title = job
            _score_one_job(text, agent, url, title)

    def _hook_pump_loop() -> None:
        global _hook_thread_id, _hook_handle, _hook_proc
        _hook_thread_id = windll.kernel32.GetCurrentThreadId()

        _hook_proc = HOOKPROC(_keyboard_hook)
        # WH_KEYBOARD_LL requires hMod == NULL (hook proc is in this process). Non-NULL can fail (e.g. error 126).
        _hook_handle = _user32.SetWindowsHookExW(
            WH_KEYBOARD_LL,
            _hook_proc,
            None,
            0,
        )
        if _hook_handle:
            print("[paste-hook] Low-level keyboard hook installed (Ctrl+V suppressed until scored).", flush=True)
        else:
            err = windll.kernel32.GetLastError()
            print(
                f"[paste-hook] SetWindowsHookExW failed (last error {err}); "
                "Ctrl+V will not be intercepted.",
                flush=True,
            )

        class MSG(ctypes.Structure):
            _fields_ = [
                ("hwnd", wt.HWND),
                ("message", wt.UINT),
                ("wParam", wt.WPARAM),
                ("lParam", wt.LPARAM),
                ("time", wt.DWORD),
                ("pt", wt.POINT),
            ]

        msg = MSG()
        while windll.user32.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
            windll.user32.TranslateMessage(ctypes.byref(msg))
            windll.user32.DispatchMessageW(ctypes.byref(msg))

        if _hook_handle:
            _user32.UnhookWindowsHookEx(_hook_handle)
            _hook_handle = None
        _hook_proc = None

    def start_paste_hook_threads() -> None:
        global _score_worker_thread, _hook_loop_thread
        if _score_worker_thread is not None and _score_worker_thread.is_alive():
            return

        _stop_score_worker.clear()
        _score_worker_thread = threading.Thread(target=_score_worker_loop, daemon=True)
        _score_worker_thread.start()
        _hook_loop_thread = threading.Thread(target=_hook_pump_loop, daemon=True)
        _hook_loop_thread.start()

    def uninstall_paste_hook() -> None:
        global _hook_handle, _hook_thread_id
        _stop_score_worker.set()
        try:
            _score_queue.put_nowait(None)
        except Exception:
            pass
        if _hook_thread_id:
            windll.user32.PostThreadMessageW(_hook_thread_id, WM_QUIT, 0, 0)
