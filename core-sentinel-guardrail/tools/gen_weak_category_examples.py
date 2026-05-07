#!/usr/bin/env python3
"""
Generate training examples for weak PII categories:
FINANCIAL, AUTH, HEALTH, LOCATION, NAME

Run from core-sentinel-guardrail/:
  python tools/gen_weak_category_examples.py

Writes data/extra_pools/weak_category_examples.jsonl (gitignored pool dir — copy into
your pipeline or merge as needed).
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

random.seed(42)

# ── FINANCIAL examples ────────────────────────────────
FINANCIAL_RISKY = [
    "My credit card number is 4532 1234 5678 9012 exp 04/26",
    "Account number 78234501 routing 021000021",
    "Invoice to policy number POL-2024-88821",
    "Wire to IBAN GB29 NWBK 6016 1331 9268 19",
    "My SSN for tax: 234-56-7890, bank acct 11223344",
    "Health plan beneficiary ID: HB-992-JKL",
    "Charge account 5512 3456 7890 1234 CVV 321",
    "SWIFT code BOFAUS3N account 00293847561",
    "My Venmo is @johnsmith payment ref INV-00123",
    "EIN 12-3456789 for the LLC filing",
]
FINANCIAL_SAFE = [
    "The market cap is 2.4 billion dollars",
    "Q3 revenue was up 12% year over year",
    "Interest rates rose by 25 basis points",
    "The invoice total is $450 for services rendered",
    "Our budget for Q4 is approximately 1.2M",
    "Stock price closed at $148.32 yesterday",
    "The fund returned 8.4% annualized over 5 years",
]

# ── AUTH examples ─────────────────────────────────────
AUTH_RISKY = [
    "My password is Fluffy2024! please don't share",
    "API key: sk-abc123XYZ789def456ghi",
    "SSH private key -----BEGIN RSA PRIVATE KEY-----",
    "Bearer token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
    "GitHub PAT: ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    "AWS secret key AKIAIOSFODNN7EXAMPLE",
    "DB password: prod_pass@word#99 host: db.internal",
    "Slack token xoxb-123456789-abcdefghijk",
    "Login: admin Password: P@ssw0rd123!",
    "Client secret: a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4",
    "OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxx",
    "My PIN is 4821, backup code 839201",
]
AUTH_SAFE = [
    "Please reset your password using the link below",
    "Two-factor authentication is now required",
    "The login page is at auth.example.com",
    "API documentation is available at docs.example.com",
    "Contact IT to request access credentials",
    "Authentication failed — please try again",
]

# ── HEALTH examples ───────────────────────────────────
HEALTH_RISKY = [
    "Patient John Smith DOB 03/15/1982 MRN 00123456",
    "Diagnosis: Type 2 diabetes, prescribed metformin 500mg",
    "Insurance: BlueCross member ID XYZ123456789",
    "Lab result: HbA1c 7.2% for patient Jane Doe",
    "Rx: Lisinopril 10mg, patient SSN 123-45-6789",
    "Admitted: 04/12/2024, ICD-10 E11.9, room 302",
    "NPI 1234567890 prescribed for DOB 1990-06-01",
    "Mental health note: patient reports anxiety, PHQ-9 score 14",
    "HIV positive, CD4 count 450, on ART since 2019",
    "Surgical history: appendectomy 2018, patient ID P-88821",
]
HEALTH_SAFE = [
    "Please consult a doctor before taking any medication",
    "The hospital visiting hours are 9am to 8pm",
    "Health insurance open enrollment ends December 15",
    "Exercise for 30 minutes daily improves wellbeing",
    "The clinic is located at 123 Medical Center Drive",
    "Annual physicals are recommended for all adults",
]

# ── LOCATION examples ─────────────────────────────────
LOCATION_RISKY = [
    "Patient address: 42 Elm Street, Boston MA 02134",
    "Ship to: 100 Main St Apt 4B, New York NY 10001",
    "Home address: 7 Oak Lane, Austin TX 78701",
    "Employee lives at 55 Pine Rd, Chicago IL 60601",
    "Billing address: 9 River View, Seattle WA 98101",
    "Next of kin at 3 Maple Ave, Denver CO 80201",
    "GPS coords: 40.7128 N 74.0060 W (patient home)",
    "ZIP code 90210 linked to John Smith account",
]
LOCATION_SAFE = [
    "The London office handles EMEA accounts",
    "Our New York headquarters is in Midtown",
    "The conference will be held in San Francisco",
    "Teams across Europe and Asia collaborate daily",
    "The nearest branch is downtown Chicago",
    "We ship to all 50 US states",
]

# ── NAME examples (precision improvement) ────────────
NAME_SAFE_EXTRA = [
    "James is a common name in English-speaking countries",
    "The author John wrote several bestselling novels",
    "Michael Jordan is widely regarded as the best player",
    "Contact the team at support@example.com",
    "Dr. Smith presented at the medical conference",
    "The CEO announced the merger last Tuesday",
    "Please reach out to our account manager",
    "The report was authored by the analytics team",
]


def _count_prefix(rows: list[dict], prefix: str) -> int:
    """Count rows whose label equals prefix or starts with prefix + '_'."""
    return sum(
        1
        for r in rows
        if (lab := str(r.get("label", ""))) == prefix or lab.startswith(prefix + "_")
    )


def main() -> None:
    rows: list[dict] = []

    def add(texts: list[str], risk: str, label: str) -> None:
        for t in texts:
            rows.append(
                {
                    "text": t,
                    "risk": risk,
                    "label": label,
                    "source": f"synthetic_{label.lower()}",
                }
            )

    add(FINANCIAL_RISKY, "high", "FINANCIAL")
    add(FINANCIAL_SAFE, "low", "FINANCIAL_safe")
    add(AUTH_RISKY, "high", "AUTH")
    add(AUTH_SAFE, "low", "AUTH_safe")
    add(HEALTH_RISKY, "high", "HEALTH")
    add(HEALTH_SAFE, "low", "HEALTH_safe")
    add(LOCATION_RISKY, "high", "LOCATION")
    add(LOCATION_SAFE, "low", "LOCATION_safe")
    add(NAME_SAFE_EXTRA, "low", "NAME_safe")

    random.shuffle(rows)

    out = _ROOT / "data" / "extra_pools" / "weak_category_examples.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Written {len(rows)} examples to {out}")
    for prefix in ("FINANCIAL", "AUTH", "HEALTH", "LOCATION", "NAME"):
        n = _count_prefix(rows, prefix)
        print(f"  {prefix}: {n} examples")


if __name__ == "__main__":
    main()
