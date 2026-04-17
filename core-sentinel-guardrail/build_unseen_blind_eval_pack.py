"""
Build an unseen blind-eval pack with leakage filtering.

Outputs:
  - reports/blind_eval_unseen.csv
  - reports/safe_snippets_unseen.csv

The generator avoids near-duplicate phrases from known tuning/test files by
comparing candidate text against extracted phrase corpora.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"


EXPECTED_ACTION = {"safe": "silent", "medium": "warn", "risky": "block"}


@dataclass
class Row:
    text: str
    label: str
    expected_action: str
    subgroup: str


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _token_set(text: str) -> set[str]:
    return set(_normalize(text).split())


def _jaccard(a: str, b: str) -> float:
    sa = _token_set(a)
    sb = _token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))


def _extract_quoted_strings(py_text: str) -> list[str]:
    out: list[str] = []
    for m in re.finditer(r"([\"'])(.{20,}?)\1", py_text, flags=re.DOTALL):
        s = m.group(2).replace("\n", " ").strip()
        if len(s) >= 20:
            out.append(s)
    return out


def _extract_csv_texts(path: Path) -> list[str]:
    if not path.is_file():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [str(r.get("text", "")).strip() for r in reader if str(r.get("text", "")).strip()]


def _load_leakage_corpus(extra_csv: Path | None) -> list[str]:
    corpus: list[str] = []
    known_py = [
        ROOT / "custom_phrase_matrix.py",
        ROOT / "custom_phrase_matrix_100.py",
        ROOT / "binary_regression_tests.py",
    ]
    for p in known_py:
        if p.is_file():
            corpus.extend(_extract_quoted_strings(p.read_text(encoding="utf-8")))

    default_csv = Path.home() / "Downloads" / "core_sentinel_test_set.csv"
    for p in [default_csv, extra_csv]:
        if p is not None:
            corpus.extend(_extract_csv_texts(p))

    # Deduplicate after normalization
    seen: set[str] = set()
    uniq: list[str] = []
    for s in corpus:
        n = _normalize(s)
        if not n or n in seen:
            continue
        seen.add(n)
        uniq.append(s)
    return uniq


def _is_leaky(candidate: str, corpus: list[str], threshold: float = 0.62) -> bool:
    for known in corpus:
        if _jaccard(candidate, known) >= threshold:
            return True
    return False


def _safe_candidates() -> Iterable[Row]:
    project_nouns = ["release notes", "sprint plan", "incident timeline", "deployment checklist", "dashboard summary"]
    verbs = ["review", "finalize", "schedule", "prepare", "publish"]
    teams = ["platform", "data engineering", "security operations", "qa", "developer experience"]
    for i in range(1, 401):
        yield Row(
            text=f"Please {random.choice(verbs)} the {random.choice(project_nouns)} for the {random.choice(teams)} team in cycle {i}.",
            label="safe",
            expected_action=EXPECTED_ACTION["safe"],
            subgroup="safe_ops_text",
        )
        yield Row(
            text=f"Function sample {i}: def calc_{i}(items): return [x for x in items if x.get('active')]",
            label="safe",
            expected_action=EXPECTED_ACTION["safe"],
            subgroup="safe_code",
        )
        yield Row(
            text=f"Status brief {i}: build pipeline healthy, test coverage stable, no identifiers included.",
            label="safe",
            expected_action=EXPECTED_ACTION["safe"],
            subgroup="safe_brief",
        )


def _medium_candidates() -> Iterable[Row]:
    names = ["Ariana", "Mateo", "Nadia", "Jonas", "Leila", "Rohan", "Mina", "Ethan"]
    departments = ["Finance", "HR", "Sales", "Operations", "Procurement", "Support"]
    health = ["medical leave", "recovering from surgery", "out sick", "family health issue"]
    systems = ["staging cluster", "billing service", "feature flag service", "analytics worker"]
    for i in range(1, 401):
        n = random.choice(names)
        d = random.choice(departments)
        yield Row(
            text=f"{n} from {d} will join the planning review for quarter {i}.",
            label="medium",
            expected_action=EXPECTED_ACTION["medium"],
            subgroup="med_name_role",
        )
        yield Row(
            text=f"Forecast draft {i}: operating cost and budget assumptions updated for next cycle.",
            label="medium",
            expected_action=EXPECTED_ACTION["medium"],
            subgroup="med_finance_context",
        )
        yield Row(
            text=f"Team notice {i}: colleague is {random.choice(health)} and available by email for urgent tasks.",
            label="medium",
            expected_action=EXPECTED_ACTION["medium"],
            subgroup="med_health_implicit",
        )
        yield Row(
            text=f"Infra note {i}: {random.choice(systems)} requires configuration approval before rollout.",
            label="medium",
            expected_action=EXPECTED_ACTION["medium"],
            subgroup="med_system_context",
        )


def _risky_candidates() -> Iterable[Row]:
    for i in range(1, 601):
        yield Row(
            text=f"Vendor onboarding {i}: Tax ID 12-{3400000 + i} and account records attached.",
            label="risky",
            expected_action=EXPECTED_ACTION["risky"],
            subgroup="risk_taxid",
        )
        yield Row(
            text=f"Payroll export {i} includes employee IDs, salaries, bonus amounts, and banking details.",
            label="risky",
            expected_action=EXPECTED_ACTION["risky"],
            subgroup="risk_payroll",
        )
        yield Row(
            text=f"Wire request {i}: SWIFT BOFAUS3N with account number {100000000000 + i}.",
            label="risky",
            expected_action=EXPECTED_ACTION["risky"],
            subgroup="risk_swift_account",
        )
        yield Row(
            text=f"Credential dump {i}: SECRET_KEY_BASE='b1c9e0f8c0f4e8a9d0a9f6e8d2c1a7f{i:03d}'",
            label="risky",
            expected_action=EXPECTED_ACTION["risky"],
            subgroup="risk_secret_key_base",
        )
        yield Row(
            text=f"Identity artifact {i}: PAN number ABCDE{1200 + i:04d}F and driver's license DL-{420190000000 + i}.",
            label="risky",
            expected_action=EXPECTED_ACTION["risky"],
            subgroup="risk_pan_license",
        )


def _collect_unique(
    candidates: Iterable[Row],
    *,
    target: int,
    leakage_corpus: list[str],
    leaky_threshold: float,
) -> list[Row]:
    out: list[Row] = []
    seen_norm: set[str] = set()
    for row in candidates:
        n = _normalize(row.text)
        if not n or n in seen_norm:
            continue
        if _is_leaky(row.text, leakage_corpus, threshold=leaky_threshold):
            continue
        seen_norm.add(n)
        out.append(row)
        if len(out) >= target:
            break
    return out


def _write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label", "expected_action", "subgroup"])
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "text": r.text,
                    "label": r.label,
                    "expected_action": r.expected_action,
                    "subgroup": r.subgroup,
                }
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate unseen blind/safe eval CSVs with leakage filtering")
    p.add_argument("--blind-out", type=Path, default=REPORTS / "blind_eval_unseen.csv")
    p.add_argument("--safe-out", type=Path, default=REPORTS / "safe_snippets_unseen.csv")
    p.add_argument("--blind-per-class", type=int, default=40, help="Rows per class for blind set")
    p.add_argument("--safe-total", type=int, default=80, help="Rows for safe-only false-positive set")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--leaky-threshold", type=float, default=0.62)
    p.add_argument("--extra-known-csv", type=Path, default=None, help="Optional extra known phrases CSV")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    leakage_corpus = _load_leakage_corpus(args.extra_known_csv)
    print(f"Loaded leakage corpus entries: {len(leakage_corpus)}")

    safe_rows = _collect_unique(
        _safe_candidates(),
        target=max(args.safe_total, args.blind_per_class),
        leakage_corpus=leakage_corpus,
        leaky_threshold=args.leaky_threshold,
    )
    med_rows = _collect_unique(
        _medium_candidates(),
        target=args.blind_per_class,
        leakage_corpus=leakage_corpus,
        leaky_threshold=args.leaky_threshold,
    )
    risky_rows = _collect_unique(
        _risky_candidates(),
        target=args.blind_per_class,
        leakage_corpus=leakage_corpus,
        leaky_threshold=args.leaky_threshold,
    )

    if len(safe_rows) < max(args.safe_total, args.blind_per_class):
        raise SystemExit("Could not generate enough non-leaky safe rows. Lower threshold or increase templates.")
    if len(med_rows) < args.blind_per_class:
        raise SystemExit("Could not generate enough non-leaky medium rows. Lower threshold or increase templates.")
    if len(risky_rows) < args.blind_per_class:
        raise SystemExit("Could not generate enough non-leaky risky rows. Lower threshold or increase templates.")

    blind_rows = safe_rows[: args.blind_per_class] + med_rows + risky_rows
    random.shuffle(blind_rows)
    safe_only = safe_rows[: args.safe_total]

    _write_csv(args.blind_out, blind_rows)
    _write_csv(args.safe_out, safe_only)

    print(f"Wrote blind set: {args.blind_out} rows={len(blind_rows)}")
    print(f"Wrote safe set:  {args.safe_out} rows={len(safe_only)}")
    print(
        "Blind composition: "
        f"safe={args.blind_per_class}, medium={args.blind_per_class}, risky={args.blind_per_class}"
    )


if __name__ == "__main__":
    main()

