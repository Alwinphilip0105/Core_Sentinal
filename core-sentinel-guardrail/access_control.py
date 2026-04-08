"""
Access control for clipboard guardrail: confirm identity / intent before block remediation UI.
"""

from __future__ import annotations


def request_paste_permission(context: str = "block") -> bool:
    """
    Request user confirmation before showing block-level remediation.

    Returns True if the user confirms (permitted), False if cancelled or denied.
    """
    try:
        import ctypes

        u32 = ctypes.windll.user32  # type: ignore[attr-defined]
        MB_OKCANCEL = 0x01
        MB_ICONWARNING = 0x30
        IDOK = 1
        title = "Clipboard Guardrail"
        if context == "block":
            msg = (
                "Additional confirmation is required before showing options for this blocked paste.\n\n"
                "Click OK to continue, or Cancel to deny."
            )
        else:
            msg = (
                "Confirmation is required for this clipboard action.\n\n"
                "Click OK to continue, or Cancel to deny."
            )
        result = u32.MessageBoxW(0, msg, title, MB_OKCANCEL | MB_ICONWARNING)
        return int(result) == IDOK
    except Exception:
        return False
