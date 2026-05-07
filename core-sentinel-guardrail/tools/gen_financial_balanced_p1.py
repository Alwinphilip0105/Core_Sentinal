#!/usr/bin/env python3
"""
Generate balanced FINANCIAL training lines (synthetic placeholders only).

Risky tiers: wires, account excerpts, insurer claim flows. Safe tiers: earnings
summaries and internal budget prose without actionable credentials.

Run from core-sentinel-guardrail/:
  python tools/gen_financial_balanced_p1.py

Writes data/extra_pools/financial_balanced_p1.jsonl — point
GUARDRAIL_FINANCIAL_BALANCED_JSONL at this file for P1-C training merges.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

EXAMPLES: list[tuple[str, str]] = [
    # --- Risky: wires / treasury ---
    (
        "Wire desk: originate USD 842,910 to beneficiary Acme Imports Ltd referencing "
        "contract W-88412. SWIFT intermediary field must show our nostro at City Bank NYC; "
        "do not fax full wire instructions externally.",
        "high",
    ),
    (
        "Treasury OPS: ACH batch ACH-90331 failed OFAC scrub — resubmit using updated "
        "ABA routing ********* and masking account ********3210 once compliance clears.",
        "high",
    ),
    (
        "Remit GBP 129,440 to Barclays sort code 12-34-56 account ********90; reference "
        "invoice INV-EU-2025-7741 only. Recipient IBAN masked in portal.",
        "high",
    ),
    # --- Risky: account / card context ---
    (
        "Cardholder disputed charge at merchant MCC 5411; BIN lookup resolved to Platinum "
        "portfolio; last-four on dispute form reads 8841.",
        "med",
    ),
    (
        "Statement PDF for savings product shows YTD APY move; customer asked us to fax "
        "page 4 with MICR routing visible — escalate per policy.",
        "med",
    ),
    # --- Risky: insurance ---
    (
        "Adjuster noted claim CLM-993812 attached photos of totaled vehicle VIN ***************; "
        "third-party claimant counsel requested HIPAA-limited IME summary.",
        "high",
    ),
    (
        "Underwriting flagged policy POL-8871201 renewal: premium reserve adjustment after "
        "loss run from carrier portal export (member IDs redacted except control hash).",
        "med",
    ),
    # --- Safe: earnings / IR ---
    (
        "Q3 shareholder deck (public): consolidated revenue rose 7% YoY excluding FX; "
        "management reiterated medium-term EBITDA margin corridor in prepared remarks.",
        "low",
    ),
    (
        "Form 10-Q highlights: capex pacing below plan due to deferred facility spend; "
        "no guidance change versus prior quarterly call transcript on investor site.",
        "low",
    ),
    (
        "Press release headline only: company announces quarterly dividend unchanged; "
        "record date disclosed in appendix table on corporate newsroom.",
        "low",
    ),
    # --- Safe: budgets / FP&A ---
    (
        "FP&A circulated OPEX walk for marketing: sequential uptick ties to phased campaign "
        "launch; rounding differences vs prior forecast are presentation-only.",
        "low",
    ),
    (
        "Budget memo — facilities: postpone HVAC capex into next fiscal bucket; occupancy "
        "assumptions unchanged from steering committee baseline.",
        "low",
    ),
    (
        "Cost center rollup shows travel below policy cap; outliers explained by one-off "
        "customer summit (already approved exception ticket FIN-991).",
        "low",
    ),
    (
        "Internal KPI note: churn improved in SMB segment quarter over quarter; product-led "
        "growth initiatives called out qualitatively only.",
        "low",
    ),
    (
        "Scenario model B assumes flat macro; sensitivity table omits speculative FX hedges.",
        "low",
    ),
    (
        "Procurement recap: SaaS renewal basket negotiated to flat pricing; vendor names are "
        "generic placeholders in sandbox deck.",
        "low",
    ),
    (
        "Workforce planning appendix lists headcount ranges by region — no individual names.",
        "low",
    ),
    (
        "Audit committee packet excerpt: contingent liability disclosure narrative matches "
        "prior year language with refreshed dates.",
        "low",
    ),
    (
        "Controller desk note — month-end close: intercompany elimination entries balanced; "
        "no material JE reversals flagged.",
        "low",
    ),
    (
        "Regional sales update: pipeline coverage healthy; largest deal sizes described as "
        "bands without customer identifiers.",
        "low",
    ),
]


def main() -> None:
    from data import MAX_TEXT_LEN, MIN_TEXT_LEN

    out = _ROOT / "data" / "extra_pools" / "financial_balanced_p1.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    skipped = 0
    rows: list[dict] = []
    for text, risk in EXAMPLES:
        t = text.strip()
        if len(t) < MIN_TEXT_LEN or len(t) > MAX_TEXT_LEN:
            skipped += 1
            continue
        rows.append({"text": t, "risk": risk, "source": "synthetic_financial_balanced_p1"})
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Written {len(rows)} lines to {out} (skipped length: {skipped})")


if __name__ == "__main__":
    main()
