"""
1) apply_calibrated_thresholds (default CLI): read reports/pr_curve_summary.json (from
   evaluate_test.py), write per-class block/warn thresholds into config/risk_policy.json
   under policy["thresholds"].

2) run_validation_threshold_sweep: legacy sweep on the validation Arrow split (FPR cap);
   writes reports/threshold_calibration.json. Run:
   python calibrate_thresholds.py --validation-sweep
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

from data import ARROW_SAVE_DIR
from infer import predict_3class_bucket_from_probs
from risk_policy_loader import clear_risk_policy_cache, get_inference_3class_label_thresholds, load_merged_risk_policy
from train import load_arrow_splits

MODEL_DIR = _ROOT / "models" / "tinybert_guardrail"
REPORTS_DIR = _ROOT / "reports"
CONFIG_PATH = _ROOT / "config" / "risk_policy.json"
PR_CURVE_SUMMARY_PATH = REPORTS_DIR / "pr_curve_summary.json"

# Default FPR cap on non-high rows predicting as high (low+med negatives)
DEFAULT_MAX_FPR = 0.05


def _resolve_arrow_dir() -> Path:
    raw = os.environ.get("GUARDRAIL_ARROW_SAVE_DIR", ARROW_SAVE_DIR)
    p = Path(raw)
    return p if p.is_absolute() else (_ROOT / p)


def _predict_buckets_batch(
    prob_low: np.ndarray,
    prob_med: np.ndarray,
    prob_high: np.ndarray,
    th: float,
    tm: float,
) -> np.ndarray:
    """Vectorized 3-class bucket: same logic as predict_3class_bucket_from_probs for fixed th,tm."""
    return np.where(prob_high > th, 2, np.where(prob_med > tm, 1, 0))


def apply_calibrated_thresholds() -> None:
    """
    Read PR curve summary from evaluate_test.py, merge calibrated block/warn thresholds into
    config/risk_policy.json under policy["thresholds"][class_name].
    """
    if not PR_CURVE_SUMMARY_PATH.is_file():
        raise SystemExit(
            f"Missing {PR_CURVE_SUMMARY_PATH}. Run: python evaluate_test.py (generates pr_curve_summary.json)"
        )
    if not CONFIG_PATH.is_file():
        raise SystemExit(f"Missing {CONFIG_PATH}")

    with open(PR_CURVE_SUMMARY_PATH, encoding="utf-8") as f:
        pr = json.load(f)

    with open(CONFIG_PATH, encoding="utf-8") as f:
        policy = json.load(f)

    classes = pr.get("classes")
    if not isinstance(classes, dict):
        raise SystemExit("pr_curve_summary.json: missing or invalid 'classes'")

    for class_name, metrics in classes.items():
        if not isinstance(metrics, dict):
            continue
        t = metrics.get("threshold_at_precision_085")
        if t is None:
            continue
        block = round(float(t), 3)
        warn = round(max(0.30, block - 0.15), 3)

        if "thresholds" not in policy:
            policy["thresholds"] = {}
        if class_name not in policy["thresholds"]:
            policy["thresholds"][class_name] = {}

        policy["thresholds"][class_name]["block"] = block
        policy["thresholds"][class_name]["warn"] = warn
        policy["thresholds"][class_name]["derived_from"] = "pr_curve_precision_0.85"
        aup_raw = metrics.get("auprc")
        policy["thresholds"][class_name]["auprc"] = (
            round(float(aup_raw), 4) if aup_raw is not None else 0.0
        )

    policy["_calibrated_at"] = pr.get("generated_at", "")
    policy["_calibration_source"] = "reports/pr_curve_summary.json"

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(policy, f, indent=2)

    try:
        clear_risk_policy_cache()
    except Exception:
        pass

    print("Thresholds updated in config/risk_policy.json")
    print("Run the app and verify with manual_probe.py (if present).")


def run_validation_threshold_sweep() -> None:
    """Legacy: sweep validation softmax; writes reports/threshold_calibration.json."""
    max_fpr = float(os.environ.get("GUARDRAIL_CALIB_MAX_FPR", str(DEFAULT_MAX_FPR)))
    arrow_dir = _resolve_arrow_dir()
    datasets = load_arrow_splits(str(arrow_dir))

    if "validation" not in datasets or datasets["validation"].num_rows == 0:
        raise SystemExit(
            f"No validation split under {arrow_dir}. Need a non-empty validation set to calibrate."
        )

    val_ds = datasets["validation"].rename_column("label", "labels")

    if not MODEL_DIR.is_dir():
        raise SystemExit(f"Model not found at {MODEL_DIR}")

    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR.resolve()))
    nl = int(getattr(model.config, "num_labels", 3))
    if nl != 3:
        print(f"Calibration script is for 3-class models; this checkpoint has num_labels={nl}. Exiting.")
        return

    args = TrainingArguments(
        output_dir=str(MODEL_DIR / "calibrate_tmp"),
        per_device_eval_batch_size=32,
        report_to="none",
    )
    trainer = Trainer(model=model, args=args)
    pred_out = trainer.predict(val_ds)
    logits = np.asarray(pred_out.predictions)
    labels = np.asarray(pred_out.label_ids)
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    prob_low = probs[:, 0]
    prob_med = probs[:, 1]
    prob_high = probs[:, 2]

    ths = np.linspace(0.05, 0.95, 37)
    tms = np.linspace(0.05, 0.95, 37)

    policy = load_merged_risk_policy(CONFIG_PATH)
    base_th, base_tm = get_inference_3class_label_thresholds(policy)

    best_overall: tuple[float, float, float, float] | None = None
    feasible: list[tuple[float, float, float, float]] = []

    n_high = int((labels == 2).sum())
    n_non_high = int((labels != 2).sum())
    if n_high == 0:
        print("WARNING: no 'high' labels in validation set — recall is undefined.")

    for th in ths:
        for tm in tms:
            pred = _predict_buckets_batch(prob_low, prob_med, prob_high, float(th), float(tm))
            tp = int(((pred == 2) & (labels == 2)).sum())
            fn = int(((pred != 2) & (labels == 2)).sum())
            recall_h = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fp = int(((pred == 2) & (labels != 2)).sum())
            fpr = fp / n_non_high if n_non_high > 0 else 0.0

            cand = (recall_h, fpr, float(th), float(tm))
            if fpr <= max_fpr:
                feasible.append(cand)
            if best_overall is None:
                best_overall = cand
            elif recall_h > best_overall[0] or (recall_h == best_overall[0] and fpr < best_overall[1]):
                best_overall = cand

    print("\n" + "=" * 60)
    print("Threshold calibration (validation set, 3-class)")
    print("=" * 60)
    print(f"  Samples: {len(labels)}  |  true high: {n_high}  |  max FPR (non-high -> pred high): {max_fpr:.2%}")

    assert best_overall is not None
    if feasible:
        feasible.sort(key=lambda x: (-x[0], x[1], x[2], x[3]))
        r, f, th_opt, tm_opt = feasible[0]
        print(f"\n  Best under FPR cap: prob_threshold_high={th_opt:.4f}, prob_threshold_med={tm_opt:.4f}")
        print(f"    recall_high={r:.6f}  FPR={f:.6f}")
    else:
        print(f"\n  No (th, tm) pair satisfied FPR <= {max_fpr:.2%}. Showing unconstrained best recall.")
        r, f, th_opt, tm_opt = best_overall
        print(f"  Best recall (unconstrained FPR={f:.6f}): prob_threshold_high={th_opt:.4f}, prob_threshold_med={tm_opt:.4f}")
        print(f"    recall_high={r:.6f}")

    base_pred = _predict_buckets_batch(prob_low, prob_med, prob_high, base_th, base_tm)
    base_tp = int(((base_pred == 2) & (labels == 2)).sum())
    base_fn = int(((base_pred != 2) & (labels == 2)).sum())
    base_recall = base_tp / (base_tp + base_fn) if (base_tp + base_fn) > 0 else 0.0
    base_fp = int(((base_pred == 2) & (labels != 2)).sum())
    base_fpr = base_fp / n_non_high if n_non_high > 0 else 0.0
    print(f"\n  Current policy (inference_3class_labels): th={base_th:.4f}, tm={base_tm:.4f}")
    print(f"    recall_high={base_recall:.6f}  FPR={base_fpr:.6f}")

    pl, pm, ph = float(prob_low[0]), float(prob_med[0]), float(prob_high[0])
    v_model = int(predict_3class_bucket_from_probs(pl, pm, ph, policy))
    v_np = int(_predict_buckets_batch(prob_low[:1], prob_med[:1], prob_high[:1], base_th, base_tm)[0])
    assert v_model == v_np, "Calibration bucket must match infer helper"

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "threshold_calibration.json"
    rec_r, rec_f = (feasible[0][0], feasible[0][1]) if feasible else (best_overall[0], best_overall[1])
    summary = {
        "max_fpr": max_fpr,
        "n_val": int(len(labels)),
        "n_true_high": n_high,
        "current_policy": {"prob_threshold_high": base_th, "prob_threshold_med": base_tm},
        "current_metrics": {"recall_high": base_recall, "fpr_non_high_as_high": base_fpr},
        "recommended": {
            "prob_threshold_high": float(th_opt),
            "prob_threshold_med": float(tm_opt),
            "recall_high": float(rec_r),
            "fpr_non_high_as_high": float(rec_f),
        },
        "note": "Copy recommended values into config/risk_policy.json under inference_3class_labels after review.",
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Wrote {out_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--validation-sweep", "--val-sweep"):
        run_validation_threshold_sweep()
    else:
        apply_calibrated_thresholds()
