from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
DOCS = ROOT.parent / "docs"
OUT_PATH = DOCS / "data" / "model_records.json"
BENCHMARK_PR_PNG_SRC = ROOT / "outputs" / "precision_recall_curve.png"
ML_ASSETS_DIR = DOCS / "ml" / "assets"
BENCHMARK_PR_PNG_DST = ML_ASSETS_DIR / "precision_recall_curve.png"


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


def _downsample_list(vals: list[Any], max_points: int = 96) -> list[float]:
    """Reduce curve size for JSON embedded in the website (keeps shape)."""
    if not vals:
        return []
    n = len(vals)
    if n <= max_points:
        return [float(vals[i]) for i in range(n)]
    out: list[float] = []
    for i in range(max_points):
        j = int(round(i * (n - 1) / max(1, max_points - 1)))
        j = max(0, min(n - 1, j))
        out.append(float(vals[j]))
    return out


def _pair_downsample(prec: list[Any], rec: list[Any], max_points: int = 96) -> tuple[list[float], list[float]]:
    if len(prec) != len(rec) or not prec:
        return [], []
    n = len(prec)
    if n <= max_points:
        return (
            [float(prec[i]) for i in range(n)],
            [float(rec[i]) for i in range(n)],
        )
    idx = [int(round(i * (n - 1) / max(1, max_points - 1))) for i in range(max_points)]
    idx = sorted(set(idx))
    return (
        [float(prec[i]) for i in idx],
        [float(rec[i]) for i in idx],
    )


def _pr_curve_downsample(
    prec: list[Any], rec: list[Any], thr: list[Any] | None, max_points: int = 96
) -> tuple[list[float], list[float], list[float] | None]:
    """Downsample PR tuples together so score-threshold axis stays aligned for the ML dashboard."""
    if len(prec) != len(rec) or not prec:
        return [], [], None
    n = len(prec)
    thr_ok = bool(thr) and len(thr) == n
    if n <= max_points:
        tprec = [float(prec[i]) for i in range(n)]
        trec = [float(rec[i]) for i in range(n)]
        tthr = [float(thr[i]) for i in range(n)] if thr_ok else None
        return tprec, trec, tthr
    idx = [int(round(i * (n - 1) / max(1, max_points - 1))) for i in range(max_points)]
    idx = sorted(set(idx))
    tprec = [float(prec[i]) for i in idx]
    trec = [float(rec[i]) for i in idx]
    tthr = [float(thr[i]) for i in idx] if thr_ok else None
    return tprec, trec, tthr


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

    tp = int(rec.get("tp") or 0)
    tn = int(rec.get("tn") or 0)
    fp = int(rec.get("fp") or 0)
    fn = int(rec.get("fn") or 0)
    fnr = (fn / (tp + fn)) if (tp + fn) > 0 else None
    tnr = (tn / (tn + fp)) if (tn + fp) > 0 else None

    roc_full = _load_json(REPORTS / "roc_curve_summary.json")
    roc_hc = (
        (roc_full.get("high_class_curve") or {})
        if isinstance(roc_full.get("high_class_curve"), dict)
        else {}
    )
    roc_auc = _to_float(roc_hc.get("auc"))

    pr_full = _load_json(REPORTS / "pr_curve_summary.json")
    pr_hc = (
        (pr_full.get("high_class_curve") or {})
        if isinstance(pr_full.get("high_class_curve"), dict)
        else {}
    )
    pr_prec = pr_hc.get("precision_values") or []
    pr_rec = pr_hc.get("recall_values") or []
    pr_thr = pr_hc.get("thresholds") or []
    dsp, dsr, dthr = _pr_curve_downsample(pr_prec, pr_rec, pr_thr if isinstance(pr_thr, list) else None, 96)
    charts_pr_hc: dict[str, Any] = {
            "positive_class": pr_hc.get("positive_class") or "risky",
            "precision_values": dsp,
            "recall_values": dsr,
            "operating_point": pr_hc.get("operating_point") or {},
            "holdout_threshold_point": {
                "precision": _to_float(rec.get("risky_precision")),
                "recall": _to_float(rec.get("risky_recall")),
                "threshold": _to_float(rec.get("threshold")),
                "note": "Test split: precision/recall at the recommended binary threshold (binary_threshold_check.json).",
            },
    }
    if dthr is not None and len(dthr) == len(dsp):
        charts_pr_hc["thresholds"] = dthr
    charts_pr: dict[str, Any] = {"high_class_curve": charts_pr_hc}

    roc_fpr = roc_hc.get("fpr_values") or []
    roc_tpr = roc_hc.get("tpr_values") or []
    rf, rt = _pair_downsample(roc_fpr, roc_tpr, 96)
    charts_roc: dict[str, Any] = {
        "high_class_curve": {
            "positive_class": roc_hc.get("positive_class") or "risky",
            "fpr_values": rf,
            "tpr_values": rt,
            "auc": roc_auc,
            "operating_point": roc_hc.get("operating_point") or {},
            "holdout_threshold_point": {
                "fpr": _to_float(rec.get("fpr")),
                "tpr": _to_float(rec.get("risky_recall")),
                "threshold": _to_float(rec.get("threshold")),
                "note": "FPR/TPR at holdout operating threshold (risky positive)",
            },
        }
    }

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
            "false_negative_rate_risky": fnr,
            "true_negative_rate_safe": tnr,
            "roc_auc": roc_auc,
            "auc": macro_auprc,
            "risky_auprc": risky_auprc,
            "recommended_threshold": threshold,
            "selection_rule": binary_threshold.get("selection_rule"),
            "confusion_holdout": {
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "n_eval": binary_threshold.get("n_test"),
                "positive_class": "risky",
            },
            "metric_notes": {
                "fpr_definition": "FPR = FP/(FP+TN): fraction of true-safe labeled risky at the holdout threshold.",
                "fnr_definition": "FNR = FN/(TP+FN): fraction of true-risky missed (1 - recall).",
                "fpr_vs_fnr": "~4.1% is FNR at threshold 0.45, not FPR. Holdout FPR (~53%) is high — raise threshold or retrain to reduce.",
            },
        },
        "charts": {
            "pr_curve_summary": charts_pr,
            "roc_curve_summary": charts_roc,
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


def copy_benchmark_pr_figure_to_docs() -> Path | None:
    """Copy matplotlib PR benchmark PNG (professor_final_benchmark.py) into docs/ml/assets/."""
    if not BENCHMARK_PR_PNG_SRC.is_file():
        return None
    ML_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BENCHMARK_PR_PNG_SRC, BENCHMARK_PR_PNG_DST)
    return BENCHMARK_PR_PNG_DST


def publish_records() -> Path:
    payload = build_model_records()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    pr_fig = copy_benchmark_pr_figure_to_docs()
    if pr_fig:
        print(f"Copied benchmark PR figure: {pr_fig.as_posix()}")
    return OUT_PATH


def main() -> None:
    out = publish_records()
    print(f"Wrote website model records: {out.as_posix()}")


if __name__ == "__main__":
    main()

