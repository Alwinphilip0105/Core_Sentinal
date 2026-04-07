"""
Adversarial-style checks for the guardrail stack: obfuscated PII patterns, base64-like secrets,
very short strings, and mixed-language text. Uses score_clipboard_with_pii() so regex overrides
and critical-secret heuristics run together with the model.

Run: python adversarial_tests.py
Exit code 1 if any expectation fails (model-dependent cases may be marked soft).
"""

from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from infer import preload_guardrail_model, score_clipboard_with_pii

# Optional: set GUARDRAIL_ADV_STRICT=1 to treat soft checks as hard failures
_STRICT = os.environ.get("GUARDRAIL_ADV_STRICT", "").strip() in ("1", "true", "yes")


def _must(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _soft(cond: bool, msg: str) -> bool:
    """Model-dependent expectation: warn only unless GUARDRAIL_ADV_STRICT=1."""
    if cond:
        return True
    if _STRICT:
        raise AssertionError(f"[soft->hard] {msg}")
    print(f"  [soft] {msg}")
    return False


def run_case(name: str, text: str, **expect) -> None:
    """expect: risk=, decision=, block=, critical_secret_detected=, triggers_contains="""
    r = score_clipboard_with_pii(text)
    print(f"\n--- {name} ---")
    print(f"  risk={r.get('risk')} decision={r.get('decision')} block={r.get('block')} critical={r.get('critical_secret_detected')}")
    if "risk" in expect:
        _must(r.get("risk") == expect["risk"], f"expected risk={expect['risk']}, got {r.get('risk')}")
    if "decision" in expect:
        _must(r.get("decision") == expect["decision"], f"expected decision={expect['decision']}, got {r.get('decision')}")
    if "block" in expect:
        _must(bool(r.get("block")) == bool(expect["block"]), f"expected block={expect['block']}, got {r.get('block')}")
    if "critical_secret_detected" in expect:
        _must(
            bool(r.get("critical_secret_detected")) == bool(expect["critical_secret_detected"]),
            f"expected critical_secret_detected={expect['critical_secret_detected']}",
        )
    if expect.get("triggers_contains"):
        sub = expect["triggers_contains"]
        tr = " ".join(r.get("triggers") or [])
        _must(sub.lower() in tr.lower() or sub.lower() in (r.get("message") or "").lower(), f"expected trigger hint {sub!r} in output")


def long_text_truncation_test() -> None:
    """
    Long paste must use multiple sliding windows; SSN in the middle must still be visible to
    regex + policy (not dropped by single 128-token truncation).
    """
    text = (
        "My name is John Smith. " * 30
        + "My SSN is 123-45-6789. "
        + "The weather is nice. " * 30
    )
    result = score_clipboard_with_pii(text)

    print("\n--- long_text_truncation_test ---")
    print(f"  chunks_scored={result.get('chunks_scored')}")
    print(f"  token_count={result.get('token_count')}")
    spans = result.get("spans") if isinstance(result.get("spans"), list) else []
    print(f"  spans found ({len(spans)}):")
    for s in spans:
        print(f"    {s}")

    _must(result.get("chunks_scored", 0) > 1, "chunks_scored must be > 1 (multi-window)")
    _must(result.get("text_truncated") is True, "text_truncated must be True")
    _must(
        result.get("action") in ("warn", "block"),
        f"action must be warn or block, got {result.get('action')!r}",
    )
    # Regex spans use "SSN pattern"; 9-class PII models may use "ID"
    ssn_flagged = any(
        s.get("class") == "ID" or "SSN" in str(s.get("class", ""))
        for s in spans
    )
    if ssn_flagged:
        print("PASS")
    else:
        print("FAIL — truncation not fixed")
        raise AssertionError(
            "SSN not found in spans (expected class ID or a class containing SSN, e.g. SSN pattern)"
        )


def main() -> None:
    preload_guardrail_model()

    long_text_truncation_test()

    # --- Obfuscated PII (spacing / dashes) — regex overrides should elevate risk vs bare digits ---
    run_case(
        "SSN spaced",
        "Please confirm SSN 123 45 6789 for payroll.",
        triggers_contains="SSN",
    )
    run_case(
        "SSN dashed",
        "Tax ID 123-45-6789 attached.",
        triggers_contains="SSN",
    )
    run_case(
        "Card spaced groups",
        "Charge card 4532 1488 0343 6467 verified.",
        triggers_contains="card",
    )

    # --- Base64-shaped secret material (high entropy blob) ---
    blob = base64.b64encode(b"sk_live_test_0123456789abcdef").decode("ascii")
    run_case(
        "base64-like blob",
        f"Paste this token: {blob}",
    )
    r_b64 = score_clipboard_with_pii(f"config: {blob}")
    _soft(
        r_b64.get("risk") == "high" or r_b64.get("critical_secret_detected"),
        "base64 blob: expected high risk or critical_secret heuristics",
    )

    # --- Very short strings (empty is deterministic; tiny strings are model-dependent) ---
    run_case("empty", "   ", risk="low", decision="allow", block=False)
    for label, t in [("single char", "x"), ("two chars", "hi")]:
        r = score_clipboard_with_pii(t)
        print(f"\n--- {label} ---")
        print(f"  risk={r.get('risk')} decision={r.get('decision')}")
        _must(r.get("risk") in ("low", "med", "high"), "risk must be low|med|high")
        _soft(
            r.get("decision") == "allow",
            f"{label}: model often allows very short text; got decision={r.get('decision')}",
        )

    # --- Mixed-language (Latin + CJK) with embedded fake card ---
    run_case(
        "mixed language + card",
        "会议记录：请使用卡号 4111 1111 1111 1111 支付押金。",
        triggers_contains="card",
    )

    # --- JWT-like (critical secret path) ---
    jwt_like = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
    r_jwt = score_clipboard_with_pii(f"Bearer {jwt_like}")
    _must(
        r_jwt.get("critical_secret_detected") or r_jwt.get("decision") in ("warn", "block"),
        "JWT-like string should trigger critical secret or elevated decision",
    )

    # --- IP / email obfuscation still in scope for regex ---
    run_case(
        "IPv4 in text",
        "Server at 192.168.0.1 is down.",
        triggers_contains="IP",
    )

    print("\n" + "=" * 60)
    print("adversarial_tests: all hard checks passed.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nFAIL: {e}", file=sys.stderr)
        sys.exit(1)
