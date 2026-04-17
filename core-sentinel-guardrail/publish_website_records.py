from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
DOCS = ROOT.parent / "docs"
OUT_PATH = DOCS / "data" / "model_records.json"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _to_float(v: Any) -> float | None:
    try:
        n = float(v)
        return n if n == n else None
    except (TypeError, ValueError):
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _derive_macro_auprc(pr: dict[str, Any]) -> float | None:
    classes = pr.get("classes")
    if not isinstance(classes, dict):
        return None
    vals: list[float] = []
    for key in ("safe", "risky"):
        node = classes.get(key)
        if isinstance(node, dict):
            v = _to_float(node.get("auprc"))
            if v is not None:
                vals.append(v)
    if not vals:
        return None
    return sum(vals) / len(vals)


def build_model_records() -> dict[str, Any]:
    binary_threshold = _load_json(REPORTS / "binary_threshold_check.json")
    threshold_cal = _load_json(REPORTS / "threshold_calibration.json")
    pr_curve = _load_json(REPORTS / "pr_curve_summary.json")
    unseen_gate = _load_json(REPORTS / "release_readiness_report_unseen.json")

    rec = binary_threshold.get("recommended") if isinstance(binary_threshold.get("recommended"), dict) else {}
    cal_rec = threshold_cal.get("recommended") if isinstance(threshold_cal.get("recommended"), dict) else {}
    risky_pr = ((pr_curve.get("classes") or {}).get("risky") or {}) if isinstance(pr_curve.get("classes"), dict) else {}

    accuracy = _to_float(rec.get("accuracy"))
    precision = _to_float(rec.get("risky_precision"))
    recall = _to_float(rec.get("risky_recall"))
    f1 = _to_float(rec.get("risky_f1"))
    fpr = _to_float(rec.get("fpr"))
    macro_f1 = _to_float(rec.get("macro_f1"))
    safe_recall = _to_float(rec.get("safe_recall"))
    threshold = _to_float(rec.get("threshold"))
    macro_auprc = _derive_macro_auprc(pr_curve)
    risky_auprc = _to_float(risky_pr.get("auprc"))

    release_decision = unseen_gate.get("release_decision") if isinstance(unseen_gate.get("release_decision"), dict) else {}

    return {
        "artifact_version": 1,
        "generated_at": _now_iso(),
        "source": {
            "holdout_threshold_check": "core-sentinel-guardrail/reports/binary_threshold_check.json",
            "calibration": "core-sentinel-guardrail/reports/threshold_calibration.json",
            "pr_curve": "core-sentinel-guardrail/reports/pr_curve_summary.json",
            "release_gate_unseen": "core-sentinel-guardrail/reports/release_readiness_report_unseen.json",
        },
        "dataset": {
            "train_samples": 10455,
            "val_samples": 485,
            "test_samples": 2289,
            "total_samples_approx": 13229,
            "label_balance_train": {"low": 3485, "med": 3485, "high": 3485},
            "pii_categories": 24,
            "data_sources": 6,
            "source_names": [
                "NVIDIA Nemotron-PII pool",
                "Enron email pool",
                "Faker business synthetic",
                "safe negatives",
                "Patronus stub set",
                "user feedback merges",
            ],
        },
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "macro_f1": macro_f1,
            "safe_recall": safe_recall,
            "fpr_non_high_as_high": fpr,
            "auc": macro_auprc,
            "risky_auprc": risky_auprc,
            "recommended_threshold": threshold,
            "selection_rule": binary_threshold.get("selection_rule"),
        },
        "calibration": {
            "prob_threshold_risky": _to_float(cal_rec.get("prob_threshold_risky")),
            "accuracy": _to_float(cal_rec.get("accuracy")),
            "precision": _to_float(cal_rec.get("precision")),
            "recall": _to_float(cal_rec.get("recall")),
            "f1": _to_float(cal_rec.get("f1")),
            "fpr": _to_float(cal_rec.get("fpr")),
            "selection_rule": threshold_cal.get("selection_rule"),
        },
        "release_gate": {
            "gate_passed": bool(release_decision.get("gate_passed")) if release_decision else None,
            "promote_allowed": bool(release_decision.get("promote_allowed")) if release_decision else None,
            "recommended_rollout": release_decision.get("recommended_rollout") if release_decision else None,
        },
    }


def publish_records() -> Path:
    payload = build_model_records()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return OUT_PATH


def main() -> None:
    out = publish_records()
    print(f"Wrote website model records: {out.as_posix()}")


if __name__ == "__main__":
    main()

