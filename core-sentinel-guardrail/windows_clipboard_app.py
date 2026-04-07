"""
Windows clipboard guardrail app: run the PII risk pipeline on every paste (Ctrl+V)
and show allow/warn/block popups. Logs events to logs/clipboard_events.jsonl.

Usage:
  python windows_clipboard_app.py              # default mode: high -> warn
  python windows_clipboard_app.py --mode strict # strict mode: high -> block
  python windows_clipboard_app.py --show-allow-toast   # ignored; UI follows scorer action
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_guardrail_dir = Path(__file__).resolve().parent
if str(_guardrail_dir) not in sys.path:
    sys.path.insert(0, str(_guardrail_dir))

from clipboard_listener import run_listener

LOGS_DIR = _guardrail_dir / "logs"
CLIPBOARD_EVENTS_JSONL = LOGS_DIR / "clipboard_events.jsonl"


def make_log_callback(log_path: Path):
    def log(
        risk: str,
        decision: str,
        block: bool,
        action: str,
        pii_override_applied: bool,
        triggers: list,
        policy_mode: str,
        text_hash: str,
    ):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "risk": risk,
            "decision": decision,
            "block": block,
            "action": action,
            "pii_override_applied": pii_override_applied,
            "triggers": triggers,
            "policy_mode": policy_mode,
            "text_hash": text_hash,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    return log


def main():
    parser = argparse.ArgumentParser(description="Clipboard guardrail: score paste with PII pipeline.")
    parser.add_argument(
        "--mode",
        choices=["default", "strict"],
        default="default",
        help="default: high->warn; strict: high->block (allow_warn_instead=False)",
    )
    parser.add_argument(
        "--show-allow-toast",
        action="store_true",
        help="Ignored: UI is driven by scorer action (silent/warn/block). Kept for CLI compatibility.",
    )
    parser.add_argument(
        "--log",
        default=str(CLIPBOARD_EVENTS_JSONL),
        help="Path to clipboard_events.jsonl (default: logs/clipboard_events.jsonl)",
    )
    args = parser.parse_args()

    policy_override = None
    if args.mode == "strict":
        policy_override = {"high": {"allow_warn_instead": False}}

    log_path = Path(args.log)
    log_callback = make_log_callback(log_path)

    print(f"Clipboard guardrail started. Mode={args.mode}, log={log_path}")
    print("Press Ctrl+V to paste. Ctrl+C to exit.")
    run_listener(
        policy_override=policy_override,
        policy_mode=args.mode,
        log_callback=log_callback,
        show_allow_toast=args.show_allow_toast,
    )


if __name__ == "__main__":
    main()
