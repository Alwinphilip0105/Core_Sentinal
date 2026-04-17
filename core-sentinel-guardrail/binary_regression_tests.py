"""
Binary-mode regression checks:
1) Runtime threshold wiring (reads inference_binary_labels.prob_threshold_risky)
2) Deterministic severity mapping checks
3) End-to-end paste checks on mixed safe/risky snippets (10+)

Run:
  .\\.venv\\Scripts\\python.exe binary_regression_tests.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from infer import (  # noqa: E402
    _binary_risky_threshold,
    _binary_severity_from_regex_ner,
    _labels_from_scores,
    load_risk_policy,
    preload_guardrail_model,
    score_clipboard_with_pii,
)


def _must(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _show_case_result(name: str, out: dict) -> None:
    print(
        f"  {name:<30} risk={out.get('risk'):<4} "
        f"decision={out.get('decision'):<5} action={out.get('action'):<6} "
        f"score={int(out.get('risk_score', 0)):>3}"
    )


def _run_runtime_wiring_checks() -> None:
    pol = load_risk_policy()
    threshold = _binary_risky_threshold(pol)
    conf = pol.get("inference_binary_labels", {}) if isinstance(pol, dict) else {}
    policy_threshold = float(conf.get("prob_threshold_risky", 0.5))
    print("\n[1] Runtime threshold wiring")
    print(f"  policy threshold={policy_threshold:.3f}, infer threshold={threshold:.3f}")
    _must(abs(threshold - policy_threshold) < 1e-9, "infer threshold must match policy threshold")

    just_above = min(0.99, threshold + 0.01)
    just_below = max(0.01, threshold - 0.01)
    lab_hi = _labels_from_scores([{"label": "safe", "score": 1.0 - just_above}, {"label": "risky", "score": just_above}])
    lab_lo = _labels_from_scores([{"label": "safe", "score": 1.0 - just_below}, {"label": "risky", "score": just_below}])
    _must(lab_hi == ["risky"], "labels_from_scores should return risky above threshold")
    _must(lab_lo == ["safe"], "labels_from_scores should return safe below threshold")


def _run_severity_mapping_checks() -> None:
    print("\n[2] Binary severity mapping")
    thr = _binary_risky_threshold(load_risk_policy())
    s1 = _binary_severity_from_regex_ner(
        "general text",
        ["email pattern"],
        ["safe"],
        p_risky=0.2,
        risky_threshold=thr,
    )
    s2 = _binary_severity_from_regex_ner(
        "patient summary from hospital",
        [],
        ["PERSON"],
        p_risky=0.45,
        risky_threshold=thr,
    )
    s3 = _binary_severity_from_regex_ner(
        "nothing sensitive",
        [],
        [],
        p_risky=max(0.01, thr - 0.02),
        risky_threshold=thr,
    )
    s4 = _binary_severity_from_regex_ner(
        "payload",
        ["credit card pattern"],
        [],
        p_risky=0.1,
        risky_threshold=thr,
    )
    print(f"  email-> {s1}, person+medical-> {s2}, below-threshold-> {s3}, credit-card-> {s4}")
    _must(s1 == "med", "email/phone cues should map to med")
    _must(s2 == "high", "person+medical context should map to high")
    _must(s3 == "low", "below-threshold without cues should map to low")
    _must(s4 == "high", "critical regex cues should map to high")


def _run_end_to_end_cases() -> None:
    print("\n[3] End-to-end paste matrix")
    cases = [
        {"name": "safe roadmap", "text": "Roadmap draft for Q3 planning and release timing.", "risk": "low", "action": "silent"},
        {"name": "safe incident recap", "text": "Incident postmortem timeline with sanitized examples only.", "risk": "low", "action": "silent"},
        {"name": "email only", "text": "Please contact team at support@example.org for follow-up.", "risk": "med", "action": "warn"},
        {"name": "phone only", "text": "Reach me at 415-555-0119 after lunch.", "risk": "med", "action": "warn"},
        {"name": "ssn", "text": "Employee SSN 123-45-6789 for payroll ticket.", "risk": "high"},
        {"name": "ssn spaced", "text": "Employee SSN 123 45 6789 for payroll ticket.", "risk": "high"},
        {"name": "credit card", "text": "Card 4532 1488 0343 6467 failed verification.", "risk": "high"},
        {"name": "card split digits", "text": "Card 4 5 3 2 1 4 8 8 0 3 4 3 6 4 6 7 failed verification.", "risk": "high"},
        {"name": "iban", "text": "Transfer to IBAN GB29NWBK60161331926819 by EOD.", "risk": "high"},
        {
            "name": "aws creds",
            "text": "AWS_ACCESS_KEY_ID=AKIA_REDACTED_EXAMPLE AWS_SECRET_ACCESS_KEY=REDACTED_AWS_SECRET_PLACEHOLDER",
            "critical": True,
            "action": "block",
        },
        {"name": "jwt bearer", "text": "Authorization: Bearer REDACTED.JWT.PAYLOAD", "critical": True, "action": "block"},
        {
            "name": "split api key",
            "text": "Temporary key looks like sk - live - abc123XYZsecretkey9999 in notes.",
            "action": "warn",
        },
        {"name": "mrn", "text": "Patient MRN: 1234567 and treatment note attached.", "risk": "high"},
        {
            "name": "email+address",
            "text": "Ship docs to 742 Evergreen Terrace Springfield and notify ops@example.org.",
            "risk": "med",
            "action": "warn",
        },
        {"name": "safe docs", "text": "Documentation update for API rate-limit behavior and retries.", "risk": "low"},
    ]

    for case in cases:
        out = score_clipboard_with_pii(case["text"])
        _show_case_result(case["name"], out)
        if "risk" in case:
            _must(out.get("risk") == case["risk"], f"{case['name']}: expected risk={case['risk']}, got {out.get('risk')}")
        if "action" in case:
            _must(out.get("action") == case["action"], f"{case['name']}: expected action={case['action']}, got {out.get('action')}")
        if case.get("critical"):
            _must(bool(out.get("critical_secret_detected")), f"{case['name']}: expected critical_secret_detected=True")


def main() -> None:
    preload_guardrail_model()
    _run_runtime_wiring_checks()
    _run_severity_mapping_checks()
    _run_end_to_end_cases()
    print("\nAll binary regression checks passed.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nFAIL: {e}", file=sys.stderr)
        sys.exit(1)

