"""
Windows clipboard guardrail listener: hooks Ctrl+V, scores with score_clipboard_with_pii,
shows UI by result[\"action\"] (silent / warn / block), and optionally suppresses paste.
"""

import hashlib
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

# Ensure guardrail package is on path when run from project root
_guardrail_dir = Path(__file__).resolve().parent
if str(_guardrail_dir) not in sys.path:
    sys.path.insert(0, str(_guardrail_dir))

import pyperclip
from access_control import request_paste_permission
from infer import _log_scoring_event, _pii_class_for_log, score_clipboard_with_pii
from active_window_llm import (
    detect_llm_window,
    get_foreground_app_label,
    log_guardrail_active_window_banner,
)

# MessageBox constants (Windows)
MB_OK = 0x00
MB_OKCANCEL = 0x01
MB_YESNO = 0x04
MB_ICONINFORMATION = 0x40
MB_ICONWARNING = 0x30
MB_ICONERROR = 0x10
IDOK = 1
IDCANCEL = 2
IDYES = 6
IDNO = 7


def log_event(text: str, result: dict, *, action: str) -> None:
    """Append one audit row to logs/guardrail.db via infer's async queue (same schema as scoring events)."""
    labels = result.get("pii_labels") or []
    risk = str(result.get("risk", "unknown"))
    rs = result.get("risk_score")
    conf = 0.0
    if isinstance(rs, (int, float)):
        conf = min(1.0, max(0.0, float(rs) / 100.0))
    _log_scoring_event(
        text,
        pii_class=_pii_class_for_log(labels, risk),
        confidence=conf,
        action=action,
    )


def show_guardrail_popup(
    title: str,
    message: str,
    level: str,
    allow_override: bool,
) -> bool:
    """
    Show a Windows MessageBox. Returns True if user chose Proceed/Yes, False otherwise.

    level: "info" | "warning" | "error"
    allow_override: if True, show OK (Proceed) / Cancel; return True if Proceed.
                    if False, show OK only and always return False.
    """
    try:
        import ctypes
        u32 = ctypes.windll.user32  # type: ignore[attr-defined]
        if level == "info":
            flags = MB_OK | MB_ICONINFORMATION
        elif level == "warning":
            # Yes = Proceed, No = Cancel
            flags = (MB_YESNO if allow_override else MB_OK) | MB_ICONWARNING
        else:
            flags = MB_OK | MB_ICONERROR
        result = u32.MessageBoxW(0, message, title, flags)
        if allow_override:
            return result == IDYES  # Yes = Proceed
        return False
    except Exception:
        return False


def show_warn_tray_balloon(title: str, message: str) -> None:
    """
    Non-modal Windows tray balloon (best-effort). Does not block the keyboard hook thread.
    Falls back to no-op if PowerShell / NotifyIcon is unavailable.
    """
    def _esc(s: str) -> str:
        return (s or "").replace("'", "''").replace("\r", " ").replace("\n", " ")[:400]

    t = _esc(title)
    m = _esc(message)
    script = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "$ni = New-Object System.Windows.Forms.NotifyIcon; "
        "$ni.Icon = [System.Drawing.SystemIcons]::Information; "
        "$ni.Visible = $true; "
        f"$ni.ShowBalloonTip(8000, '{t}', '{m}', [System.Windows.Forms.ToolTipIcon]::Warning); "
        "Start-Sleep -Seconds 9; $ni.Visible = $false; $ni.Dispose()"
    )

    def _run() -> None:
        try:
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            subprocess.run(
                ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", script],
                capture_output=True,
                timeout=20,
                creationflags=creationflags,
            )
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True).start()


def handle_paste(
    policy_override: Optional[dict],
    policy_mode: str,
    log_callback: Callable[..., None],
    show_allow_toast: bool,
) -> bool:
    """
    Read clipboard, score with score_clipboard_with_pii, log, notify per result[\"action\"].
    Returns True to allow paste (let Ctrl+V through), False to suppress (block only).
    show_allow_toast is ignored; notifications follow \"action\" from the scorer.
    """
    _ = show_allow_toast  # CLI flag kept for compatibility; UI uses result["action"] only.
    # Only run PII analysis when the user is actively pasting into a known LLM target.
    is_llm, agent_name, _url, _cleaned_title = detect_llm_window(debug=False)
    banner_label = agent_name if is_llm else get_foreground_app_label()
    log_guardrail_active_window_banner(is_llm, banner_label)
    if not is_llm:
        return True

    try:
        text = pyperclip.paste()
    except Exception:
        return True
    if not text or not isinstance(text, str):
        return True
    text = text.strip()
    if not text:
        return True

    result = score_clipboard_with_pii(text, policy_override=policy_override)
    if os.environ.get("GUARDRAIL_DEBUG_REMEDIATION", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        print(f"[result keys] {list(result.keys())}", flush=True)
        print(f"[result] action={result.get('action')} spans={len(result.get('spans') or [])} ", flush=True)
    risk = result["risk"]
    decision = result["decision"]
    block = result["block"]
    message = result["message"]
    triggers = result.get("triggers", [])
    action = result.get("action")
    if action not in ("silent", "warn", "block"):
        action = {"allow": "silent", "warn": "warn", "block": "block"}.get(decision, "silent")
        if result.get("critical_secret_detected"):
            action = "block"

    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    log_callback(
        risk=risk,
        decision=decision,
        block=block,
        action=action,
        pii_override_applied=result["pii_override_applied"],
        triggers=triggers,
        policy_mode=policy_mode,
        text_hash=text_hash,
    )

    if action == "silent":
        return True

    if action == "warn":
        show_warn_tray_balloon("Clipboard Guardrail", message)
        return True

    if action == "block":
        permitted = request_paste_permission("block")
        if not permitted:
            log_event(text, result, action="auth_denied")
            show_warn_tray_balloon(
                "Clipboard Guardrail",
                "Paste blocked: confirmation was cancelled or denied.",
            )
            return False
        show_guardrail_popup(
            "Clipboard Guardrail Blocked",
            message,
            "error",
            allow_override=False,
        )
        return False

    # Fallback for unexpected action values (legacy mapping)
    show_guardrail_popup(
        "Clipboard Guardrail Blocked",
        message,
        "error",
        allow_override=False,
    )
    return False


def run_listener(
    policy_override: Optional[dict] = None,
    policy_mode: str = "default",
    log_callback: Optional[Callable[..., None]] = None,
    show_allow_toast: bool = False,
) -> None:
    """
    Register Ctrl+V hook and run. On each paste: score, log, notify by action; suppress paste only on block.
    log_callback(...) is called with risk, decision, block, action, pii_override_applied, triggers, policy_mode, text_hash.
    """
    import keyboard

    def noop_log(*args, **kwargs):
        pass

    log_fn = log_callback or noop_log

    def on_key(event):
        if getattr(event, "event_type", None) != "down":
            return True
        if getattr(event, "name", None) != "v":
            return True
        try:
            if keyboard.is_pressed("ctrl"):
                suppress = not handle_paste(
                    policy_override, policy_mode, log_fn, show_allow_toast
                )
                return not suppress
        except Exception:
            pass
        return True

    keyboard.hook(on_key)
    try:
        keyboard.wait()
    except KeyboardInterrupt:
        pass
