"""
Release readiness gate for Core Sentinel binary guardrail.

What it does in one run:
1) Blind-eval check on an unseen labeled set
2) False-positive check on safe-only snippets
3) Canary rollout recommendation with rollback thresholds

Usage:
  .\\.venv\\Scripts\\python.exe release_readiness_gate.py ^
    --blind-set C:\\path\\to\\blind_eval.csv ^
    --safe-set C:\\path\\to\\safe_snippets.csv

Accepted input formats:
  - .csv with columns: text,label[,expected_action]
  - .jsonl with keys: text,label[,expected_action]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from infer import score_clipboard_with_pii  # noqa: E402


DEFAULT_GATE_CONFIG = _ROOT / "config" / "release_gate.json"
DEFAULT_REPORT_PATH = _ROOT / "reports" / "release_readiness_report.json"


EXPECTED_ACTION_BY_LABEL = {
    "safe": "silent",
    "low": "silent",
    "medium": "warn",
    "med": "warn",
    "risky": "block",
    "high": "block",
    "critical": "block",
}


@dataclass
class EvalRow:
    text: str
    label: str
    expected_action: str
    source: str


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _expected_from_label(label: str) -> str:
    lab = (label or "").strip().lower()
    return EXPECTED_ACTION_BY_LABEL.get(lab, "warn")


def _iter_rows_csv(path: Path) -> Iterable[EvalRow]:
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=2):
            text = str(row.get("text", "")).strip()
            label = str(row.get("label", "")).strip().lower()
            expected = str(row.get("expected_action", "")).strip().lower()
            if not text:
                continue
            if not expected:
                expected = _expected_from_label(label)
            yield EvalRow(text=text, label=label, expected_action=expected, source=f"{path.name}:{idx}")


def _iter_rows_jsonl(path: Path) -> Iterable[EvalRow]:
    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            text = str(obj.get("text", "")).strip()
            label = str(obj.get("label", "")).strip().lower()
            expected = str(obj.get("expected_action", "")).strip().lower()
            if not text:
                continue
            if not expected:
                expected = _expected_from_label(label)
            yield EvalRow(text=text, label=label, expected_action=expected, source=f"{path.name}:{idx}")


def _load_eval_rows(path: Path) -> list[EvalRow]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing eval file: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return list(_iter_rows_csv(path))
    if suffix == ".jsonl":
        return list(_iter_rows_jsonl(path))
    raise ValueError(f"Unsupported file format for {path}. Use .csv or .jsonl")


def _rate(numer: int, denom: int) -> float:
    if denom <= 0:
        return 0.0
    return float(numer) / float(denom)


def _evaluate_rows(rows: list[EvalRow], *, context: str = "document") -> dict:
    passed = 0
    by_label: dict[str, dict[str, int]] = {}
    mismatches: list[dict] = []
    counts = {
        "safe_total": 0,
        "safe_fp": 0,
        "medium_total": 0,
        "medium_warn_ok": 0,
        "risky_total": 0,
        "risky_block_ok": 0,
    }

    for i, row in enumerate(rows, start=1):
        out = score_clipboard_with_pii(row.text, context=context)
        got = str(out.get("action", "silent")).strip().lower()
        ok = got == row.expected_action
        passed += int(ok)

        lab = row.label or "unknown"
        lab_bin = by_label.setdefault(lab, {"total": 0, "passed": 0})
        lab_bin["total"] += 1
        lab_bin["passed"] += int(ok)

        if lab in ("safe", "low"):
            counts["safe_total"] += 1
            if got != "silent":
                counts["safe_fp"] += 1
        if lab in ("medium", "med"):
            counts["medium_total"] += 1
            if got == "warn":
                counts["medium_warn_ok"] += 1
        if lab in ("risky", "high", "critical"):
            counts["risky_total"] += 1
            if got == "block":
                counts["risky_block_ok"] += 1

        if not ok:
            mismatches.append(
                {
                    "row": i,
                    "source": row.source,
                    "label": lab,
                    "expected_action": row.expected_action,
                    "predicted_action": got,
                    "predicted_risk": out.get("risk"),
                    "risk_score": out.get("risk_score"),
                    "text": row.text,
                }
            )

    total = len(rows)
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": _rate(passed, total),
        "by_label": {
            k: {
                "total": v["total"],
                "passed": v["passed"],
                "pass_rate": _rate(v["passed"], v["total"]),
            }
            for k, v in sorted(by_label.items())
        },
        "safe_false_positive_rate": _rate(counts["safe_fp"], counts["safe_total"]),
        "medium_warn_recall": _rate(counts["medium_warn_ok"], counts["medium_total"]),
        "risky_block_recall": _rate(counts["risky_block_ok"], counts["risky_total"]),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def _load_gate_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing gate config: {path}")
    cfg = _load_json(path)
    if not isinstance(cfg, dict):
        raise ValueError("Gate config must be a JSON object.")
    return cfg


def _status_line(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _evaluate_gate(blind: dict, safe_eval: dict | None, gate_cfg: dict) -> dict:
    criteria = gate_cfg.get("criteria", {})
    rollout = gate_cfg.get("canary_rollout", {})
    rollback = gate_cfg.get("rollback_triggers", {})

    blind_pass_min = float(criteria.get("blind_pass_rate_min", 0.92))
    medium_recall_min = float(criteria.get("blind_medium_warn_recall_min", 0.85))
    risky_recall_min = float(criteria.get("blind_risky_block_recall_min", 0.98))
    safe_fp_max = float(criteria.get("safe_false_positive_rate_max", 0.03))

    safe_fp_metric = (
        float(safe_eval.get("safe_false_positive_rate", 0.0))
        if isinstance(safe_eval, dict)
        else float(blind.get("safe_false_positive_rate", 0.0))
    )

    checks = [
        {
            "name": "blind_pass_rate",
            "threshold": blind_pass_min,
            "value": float(blind.get("pass_rate", 0.0)),
            "ok": float(blind.get("pass_rate", 0.0)) >= blind_pass_min,
        },
        {
            "name": "blind_medium_warn_recall",
            "threshold": medium_recall_min,
            "value": float(blind.get("medium_warn_recall", 0.0)),
            "ok": float(blind.get("medium_warn_recall", 0.0)) >= medium_recall_min,
        },
        {
            "name": "blind_risky_block_recall",
            "threshold": risky_recall_min,
            "value": float(blind.get("risky_block_recall", 0.0)),
            "ok": float(blind.get("risky_block_recall", 0.0)) >= risky_recall_min,
        },
        {
            "name": "safe_false_positive_rate",
            "threshold": safe_fp_max,
            "value": safe_fp_metric,
            "ok": safe_fp_metric <= safe_fp_max,
        },
    ]

    gate_passed = all(ch["ok"] for ch in checks)
    recommended_rollout = (
        "promote_to_full"
        if gate_passed
        else f"canary_{int(float(rollout.get('starting_percent', 0.1)) * 100)}pct"
    )
    promote_allowed = gate_passed

    return {
        "gate_passed": gate_passed,
        "promote_allowed": promote_allowed,
        "recommended_rollout": recommended_rollout,
        "checks": checks,
        "canary_rollout": rollout,
        "rollback_triggers": rollback,
    }


def _print_human_summary(report: dict) -> None:
    blind = report["blind_eval"]
    safe_eval = report.get("safe_eval")
    decision = report["release_decision"]

    print("\n=== Release Readiness Gate ===")
    print(
        f"Blind eval: total={blind['total']} pass={blind['passed']} "
        f"fail={blind['failed']} pass_rate={blind['pass_rate']*100:.1f}%"
    )
    print(
        f"Blind recalls: medium_warn={blind['medium_warn_recall']*100:.1f}% "
        f"risky_block={blind['risky_block_recall']*100:.1f}%"
    )
    if isinstance(safe_eval, dict):
        print(
            f"Safe FP check: total={safe_eval['total']} fp_rate={safe_eval['safe_false_positive_rate']*100:.2f}%"
        )
    else:
        print(
            f"Safe FP check: using blind safe subset fp_rate={blind['safe_false_positive_rate']*100:.2f}%"
        )

    print("\nChecks:")
    for ch in decision["checks"]:
        symbol = _status_line(bool(ch["ok"]))
        if ch["name"] == "safe_false_positive_rate":
            print(
                f"  {symbol} {ch['name']}: {ch['value']*100:.2f}% "
                f"(max {ch['threshold']*100:.2f}%)"
            )
        else:
            print(
                f"  {symbol} {ch['name']}: {ch['value']*100:.2f}% "
                f"(min {ch['threshold']*100:.2f}%)"
            )

    print("\nDecision:")
    print(f"  gate_passed={decision['gate_passed']}")
    print(f"  promote_allowed={decision['promote_allowed']}")
    print(f"  recommended_rollout={decision['recommended_rollout']}")
    rollout = decision.get("canary_rollout", {})
    if isinstance(rollout, dict) and rollout:
        print(
            f"  canary plan: start={int(float(rollout.get('starting_percent', 0.1))*100)}% "
            f"step-up={int(float(rollout.get('step_up_percent', 0.1))*100)}% "
            f"duration_hours={int(rollout.get('canary_duration_hours', 24))}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Blind eval + safe FP + canary release gate")
    p.add_argument("--blind-set", type=Path, required=True, help="Path to blind eval .csv or .jsonl")
    p.add_argument("--safe-set", type=Path, default=None, help="Path to safe-only .csv or .jsonl")
    p.add_argument("--gate-config", type=Path, default=DEFAULT_GATE_CONFIG, help="Path to gate config JSON")
    p.add_argument("--out", type=Path, default=DEFAULT_REPORT_PATH, help="Output report JSON path")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    blind_rows = _load_eval_rows(args.blind_set)
    if not blind_rows:
        raise SystemExit("Blind eval set is empty.")
    blind_eval = _evaluate_rows(blind_rows, context="document")

    safe_eval = None
    if args.safe_set is not None:
        safe_rows = _load_eval_rows(args.safe_set)
        # Force expected action to silent for safe-only check.
        normalized = [EvalRow(text=r.text, label="safe", expected_action="silent", source=r.source) for r in safe_rows]
        safe_eval = _evaluate_rows(normalized, context="document")

    gate_cfg = _load_gate_config(args.gate_config)
    decision = _evaluate_gate(blind_eval, safe_eval, gate_cfg)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "blind_set": str(args.blind_set),
        "safe_set": str(args.safe_set) if args.safe_set else None,
        "gate_config": str(args.gate_config),
        "blind_eval": blind_eval,
        "safe_eval": safe_eval,
        "release_decision": decision,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    _print_human_summary(report)
    print(f"\nWrote report: {args.out}")


if __name__ == "__main__":
    main()

