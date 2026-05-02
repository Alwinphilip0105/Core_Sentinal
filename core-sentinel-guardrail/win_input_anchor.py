"""
Windows: find the HWND with keyboard focus in the foreground window and its screen rect.
Used to anchor the guardrail pill to the right edge of a focused text surface (browser/app).
No extra packages — ctypes + user32 only.
"""

from __future__ import annotations

import ctypes
import sys
from typing import Optional, Tuple

if sys.platform != "win32":
    def get_focused_text_surface_rect_screen() -> Optional[Tuple[int, int, int, int]]:
        return None

    def get_foreground_window_hwnd() -> Optional[int]:
        return None

    def get_foreground_focus_hwnd() -> Optional[int]:
        return None

else:
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    class _RECT(ctypes.Structure):
        _fields_ = [
            ("left", wintypes.LONG),
            ("top", wintypes.LONG),
            ("right", wintypes.LONG),
            ("bottom", wintypes.LONG),
        ]

    def get_foreground_window_hwnd() -> Optional[int]:
        fg = user32.GetForegroundWindow()
        return int(fg) if fg else None

    def get_foreground_focus_hwnd() -> Optional[int]:
        return _foreground_focus_hwnd()

    def _foreground_focus_hwnd() -> Optional[int]:
        fg = user32.GetForegroundWindow()
        if not fg:
            return None
        pid = wintypes.DWORD()
        tid_fg = int(user32.GetWindowThreadProcessId(fg, ctypes.byref(pid)))
        tid_cur = int(kernel32.GetCurrentThreadId())
        if not tid_fg:
            return None
        if tid_fg == tid_cur:
            h = user32.GetFocus()
            return int(h) if h else None
        user32.AttachThreadInput(tid_cur, tid_fg, True)
        try:
            h = user32.GetFocus()
            return int(h) if h else None
        finally:
            user32.AttachThreadInput(tid_cur, tid_fg, False)

    def _class_name(hwnd: int) -> str:
        buf = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(wintypes.HWND(hwnd), buf, 256)
        return buf.value or ""

    def _screen_rect(hwnd: int) -> Optional[Tuple[int, int, int, int]]:
        r = _RECT()
        if not user32.GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(r)):
            return None
        return (int(r.left), int(r.top), int(r.right), int(r.bottom))

    def _is_probable_text_surface(class_name: str) -> bool:
        c = class_name.lower()
        if c in ("edit",):
            return True
        if "richedit" in c:
            return True
        if c == "chrome_renderwidgethosthwnd":
            return True
        if "webview" in c:
            return True
        if "mozillawindowclass" in c or c.startswith("mozilla"):
            return True
        if c.endswith(".webkit.webview"):
            return True
        return False

    def get_focused_text_surface_rect_screen() -> Optional[Tuple[int, int, int, int]]:
        hwnd = _foreground_focus_hwnd()
        if not hwnd:
            return None
        if not _is_probable_text_surface(_class_name(hwnd)):
            return None
        return _screen_rect(hwnd)
