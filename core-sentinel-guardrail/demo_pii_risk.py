#!/usr/bin/env python3
"""
Demo CLI: read text from stdin or a file, run score_clipboard_with_pii, print risk and message.

Usage:
  python demo_pii_risk.py                    # read from stdin
  python demo_pii_risk.py file.txt           # read from file
  python demo_pii_risk.py --demo             # run 3-4 example outputs (low/allow, med/warn, high/warn, high/block)

Toggle high-risk behavior:
  Set GUARDRAIL_HIGH_BLOCK=1 to force block for high risk (otherwise high -> warn when allow_warn_instead=True).
  In --demo, the fourth example uses policy_override to show high/block without changing global config.
"""

import os
import sys
from pathlib import Path

# Allow running from project root when core-sentinel-guardrail is not on path
_guardrail_dir = Path(__file__).resolve().parent
if str(_guardrail_dir) not in sys.path:
    sys.path.insert(0, str(_guardrail_dir))

from infer import load_pii_policy, score_clipboard_with_pii


def print_result(result: dict, preview: str, title: str = "") -> None:
    if title:
        print(f"\n--- {title} ---")
    print("Risk:", result["risk"])
    print("Decision:", result["decision"])
    print("Block:", result["block"])
    print("Message:", result.get("message", "(none)"))
    print("Override applied:", result["pii_override_applied"])
    print("PII labels:", result["pii_labels"])
    print("Preview:", preview)


def run_demo() -> None:
    """Run 3-4 example calls and print outputs for low/allow, med/warn, high/warn, high/block."""
    policy = load_pii_policy()
    high_warn_instead = policy.get("high", {}).get("allow_warn_instead", True)
    print("POLICY['high']['allow_warn_instead'] =", high_warn_instead)
    print("(Set config/pii_policy.json or env to change. Example 4 forces block via policy_override.)")

    examples = [
        ("1. Low / Allow (no PII)", "Team standup at 9am tomorrow. No sensitive data here."),
        ("2. Med / Warn (names or location)", "Meeting with John Smith in Newark next week."),
        ("3. High / Warn (high risk, policy=warn)", "Naman lost $6000000 in the prediction market due to unforeseen circumstances."),
    ]
    for title, text in examples:
        result = score_clipboard_with_pii(text)
        preview = text[:80] + ("..." if len(text) > 80 else "")
        print_result(result, preview, title)

    # 4. High / Block: same as high-risk content but with policy_override to force block
    title4 = "4. High / Block (same content, policy_override: allow_warn_instead=False)"
    text4 = "Customer SSN 123-45-6789 and account details for wire transfer."
    result4 = score_clipboard_with_pii(text4, policy_override={"high": {"allow_warn_instead": False}})
    preview4 = text4[:80] + ("..." if len(text4) > 80 else "")
    print_result(result4, preview4, title4)

    print("\n--- Summary ---")
    print("To make high risk always block: set in config/pii_policy.json:")
    print('  { "high": { "allow_warn_instead": false } }')
    print("Or in code: infer.POLICY['high']['allow_warn_instead'] = False")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in ("--demo", "-d"):
        run_demo()
        return

    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            sys.exit(1)
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    else:
        text = sys.stdin.read().strip()

    if not text:
        print("No text to score (empty stdin or file). Use --demo for example runs.")
        sys.exit(0)

    result = score_clipboard_with_pii(text)
    preview = text[:120] + ("..." if len(text) > 120 else "")

    print("Risk:", result["risk"])
    print("Decision:", result["decision"])
    print("Block:", result["block"])
    print("Message:", result.get("message", "(none)"))
    print("Override applied:", result["pii_override_applied"])
    if result["pii_override_applied"]:
        print("Risk before override:", result["pii_risk_before_override"])
    print("PII labels:", result["pii_labels"])
    print("Preview:", preview)


if __name__ == "__main__":
    main()
