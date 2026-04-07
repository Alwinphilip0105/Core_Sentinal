"""
Locate the focused text input in the foreground window using win32gui + pywinauto (UIA).

Returns (x, y, width, height) in screen coordinates for anchoring the guardrail pill
to the chat input, Grammarly-style. Requires Windows, pywin32, and pywinauto.
"""

from __future__ import annotations

import sys
from typing import Optional, Tuple

Rect = Tuple[int, int, int, int]  # x, y, width, height

# UIA control types treated as text inputs (names from pywinauto / UIA)
_TEXT_INPUT_CONTROL_TYPES = frozenset(
    {
        "Edit",
        "Document",
        "Text",
        "ComboBox",  # editable / search-style fields
    }
)


def get_input_field_rect() -> Optional[Rect]:
    """
    1. Foreground window via win32gui.GetForegroundWindow().
    2. pywinauto UIA GetFocusedElement() for the focused control.
    3. If it is a text-like control and belongs to the foreground top-level window,
       return its bounding rect as (x, y, width, height) in screen coordinates.
    4. Return None if not Windows, imports fail, window is not an LLM app, or no suitable focus.
    """
    if sys.platform != "win32":
        return None

    try:
        import win32gui  # type: ignore[import-untyped]
        from pywinauto.uia_element_info import UIAElementInfo  # type: ignore[import-untyped]
    except ImportError:
        return None

    # Local import avoids loading active_window_llm on non-Windows stubs
    from active_window_llm import is_active_window_llm

    fg_hwnd = int(win32gui.GetForegroundWindow())
    if not fg_hwnd:
        return None

    is_llm, _, _, _ = is_active_window_llm()
    if not is_llm:
        return None

    try:
        focused = UIAElementInfo.GetFocusedElement()
    except Exception:
        return None

    if focused is None:
        return None

    try:
        ct = (focused.control_type or "").strip()
        if ct not in _TEXT_INPUT_CONTROL_TYPES:
            return None
    except Exception:
        return None

    try:
        tl = focused.top_level_parent
        tl_hwnd = int(tl.handle) if tl is not None and getattr(tl, "handle", None) else 0
        if tl_hwnd and tl_hwnd != fg_hwnd:
            return None
    except Exception:
        pass

    try:
        r = focused.rectangle
        if r is None:
            return None
        left = int(r.left)
        top = int(r.top)
        right = int(r.right)
        bottom = int(r.bottom)
        w = max(0, right - left)
        h = max(0, bottom - top)
        if w <= 0 or h <= 0:
            return None
        return (left, top, w, h)
    except Exception:
        return None
