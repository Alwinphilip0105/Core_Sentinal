import sys
from pathlib import Path

# Add core-sentinel-guardrail so "infer" is importable when run from project root or scripts/
_guardrail_dir = Path(__file__).resolve().parent.parent / "core-sentinel-guardrail"
sys.path.insert(0, str(_guardrail_dir))

from infer import score_clipboard, score_clipboard_with_pii

examples = [
    "Hey, can you summarize this team meeting for me?",
    "Employee payroll for March: John Doe, salary 95000, bonus 12000, account 44321098.",
    "SAP Vendor ID: SAP-882314, invoice total $45,120, internal approval pending.",
    "My class notes say the assignment deadline is next Thursday.",
    "Customer SSN 123-45-6789 and bank routing 021000021 are listed here.",
    "Revenue by region: East 1.2M, West 980k, Q2 forecast attached.",
    "Please rewrite this email in a more professional tone.",
    "Internal M&A target list and projected EBITDA multiples for review.",
]

HARDCODED_TESTS = [
    ("John", "silent", "plain first name"),
    ("917-555-0132", "warn", "US phone number"),
    ("sk-abc123XYZ789secretkey", "block", "API key string"),
]


def _pii_class_display(result: dict) -> str:
    labels = result.get("pii_labels") or []
    if labels:
        return "/".join(str(x) for x in labels)
    return str(result.get("risk", "?"))


if __name__ == "__main__":
    for i, text in enumerate(examples, 1):
        result = score_clipboard(text)
        print(f"\n--- Example {i} ---")
        print(text)
        print(result)

    print("\n=== Hardcoded threshold smoke tests ===")
    for text, expected_action, description in HARDCODED_TESTS:
        result = score_clipboard_with_pii(text)
        pii_class = _pii_class_display(result)
        score = result.get("risk_score", 0)
        action = result.get("action", "?")
        print(f"\n[{description}] input={text!r}")
        print(f"  pii_class: {pii_class}")
        print(f"  score: {score}")
        print(f"  action: {action} (expected: {expected_action})")
        if action != expected_action:
            print(
                "  WARNING: action does not match expected value; "
                "thresholds or policy (risk_policy.json / pii_policy / per_class_thresholds) may not be loaded correctly."
            )
