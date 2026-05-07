from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infer import score_clipboard, score_clipboard_with_pii
from risk_mapping import strong_regex_pii_spans


TEXTS = [
    {
        "id": "safe_1",
        "text": "Please finalize the sprint backlog and send meeting notes to the platform team.",
        "label_hint": "safe",
    },
    {
        "id": "regex_hard_1",
        "text": "Customer SSN is 123-45-6789 and card 4111-1111-1111-1111.",
        "label_hint": "risky",
    },
    {
        "id": "context_1",
        "text": "Please summarize employee compensation with full names, monthly pay, bonus, and banking details before sharing with the assistant.",
        "label_hint": "risky_context",
    },
    {
        "id": "paraphrase_a",
        "text": "Wire request: SWIFT BOFAUS3N with account number 100000000123.",
        "label_hint": "risky_financial",
    },
    {
        "id": "paraphrase_b",
        "text": "Process an international transfer using BOFAUS3N and beneficiary account 100000000123.",
        "label_hint": "risky_financial_paraphrase",
    },
    {
        "id": "medium_narrative",
        "text": "Ariana from HR is on medical leave and available by personal email for urgent updates.",
        "label_hint": "medium_personal",
    },
]

THRESHOLDS = [0.10, 0.45, 0.65]


def ml_only_decision(prob_risky: float, threshold: float) -> str:
    return "risky" if prob_risky >= threshold else "safe"


def main() -> None:
    rows = []
    for item in TEXTS:
        text = item["text"]
        regex_spans = strong_regex_pii_spans(text)
        plain = score_clipboard(text, use_pii=False)
        combined = score_clipboard_with_pii(text)

        prob_risky = float(plain.get("prob_high", 0.0))
        ml_threshold_results = {
            f"tau_{thr:.2f}": ml_only_decision(prob_risky, thr) for thr in THRESHOLDS
        }

        rows.append(
            {
                "id": item["id"],
                "label_hint": item["label_hint"],
                "text": text,
                "regex_match_count": len(regex_spans),
                "regex_match_types": [str(s.get("class", "")) for s in regex_spans],
                "ml_prob_risky_proxy": round(prob_risky, 6),
                "ml_prob_low": float(plain.get("prob_low", 0.0)),
                "ml_prob_med": float(plain.get("prob_med", 0.0)),
                "ml_only_by_threshold": ml_threshold_results,
                "combined_action": combined.get("action"),
                "combined_decision": combined.get("decision"),
                "combined_risk": combined.get("risk"),
                "combined_risk_score": combined.get("risk_score"),
                "combined_triggers": combined.get("triggers", []),
            }
        )

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "thresholds_tested": THRESHOLDS,
        "rows": rows,
    }

    report_dir = ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / "ml_importance_probe.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {out_path}")
    for r in rows:
        print("-" * 72)
        print(r["id"], "| regex", r["regex_match_count"], "| p_risky", r["ml_prob_risky_proxy"])
        print("  ml@tau:", r["ml_only_by_threshold"])
        print("  combined:", r["combined_action"], "|", r["combined_risk"], "| score", r["combined_risk_score"])


if __name__ == "__main__":
    main()
