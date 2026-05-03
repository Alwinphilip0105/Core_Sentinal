"""
1) apply_calibrated_thresholds (default CLI): read reports/pr_curve_summary.json (from
   evaluate_test.py), write per-class block/warn thresholds into config/risk_policy.json
   under policy["thresholds"].

2) run_validation_threshold_sweep: sweep on the validation Arrow split with a safety-first
   feasible set: high recall floor (GUARDRAIL_CALIB_MIN_HIGH_RECALL), FPR cap, med recall floor.
   Writes reports/threshold_calibration.json. Run:
   python calibrate_thresholds.py --validation-sweep
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
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


def _resolve_calib_model_dir() -> Path:
    """
    Directory passed to from_pretrained for calibration.

    GUARDRAIL_CALIB_MODEL_DIR overrides. Otherwise, if the latest checkpoint under
    MODEL_DIR has a 2-logit classifier but root model.safetensors is still 3-logit
    (common on Windows when the overlay mmap-holds the root file), prefer that checkpoint.
    """
    env = (os.environ.get("GUARDRAIL_CALIB_MODEL_DIR") or "").strip()
    if env:
        p = Path(env)
        if not p.is_dir():
            raise SystemExit(f"GUARDRAIL_CALIB_MODEL_DIR is not a directory: {p}")
        return p.resolve()
    try:
        import safetensors.torch as st

        root_w = MODEL_DIR / "model.safetensors"
        root_n = None
        if root_w.is_file():
            root_n = int(st.load_file(str(root_w))["classifier.weight"].shape[0])
        ck_dirs = sorted(
            MODEL_DIR.glob("checkpoint-*"),
            key=lambda x: int(x.name.split("-", 1)[1])
            if x.name.split("-", 1)[1].isdigit()
            else -1,
        )
        for ck in reversed(ck_dirs):
            wf = ck / "model.safetensors"
            if not wf.is_file():
                continue
            cn = int(st.load_file(str(wf))["classifier.weight"].shape[0])
            if cn == 2 and root_n != 2:
                print(
                    f"[calibrate] Using {ck.name} (2-class head); "
                    f"root model.safetensors has classifier.out_features={root_n}"
                )
                return ck.resolve()
    except Exception:
        pass
    return MODEL_DIR.resolve()
CONFIG_PATH = _ROOT / "config" / "risk_policy.json"
PR_CURVE_SUMMARY_PATH = REPORTS_DIR / "pr_curve_summary.json"
THRESHOLD_CALIBRATION_PATH = REPORTS_DIR / "threshold_calibration.json"

# Default FPR cap on non-high rows predicting as high (low+med negatives)
DEFAULT_MAX_FPR = 0.05
DEFAULT_MIN_MED_RECALL = 0.35
# Prefer catching true high-risk rows before optimizing headline precision (guardrail priority).
DEFAULT_MIN_HIGH_RECALL = 0.85
DEFAULT_BINARY_SELECTION_RULE = "max_recall_under_fpr_cap"


def _binary_selection_rule() -> str:
    """
    Binary calibration objective:
      - max_recall_under_fpr_cap (default, recall-first safety mode)
      - max_f1_under_fpr_cap
      - max_f2_under_fpr_cap
    """
    raw = (os.environ.get("GUARDRAIL_BINARY_CALIB_OBJECTIVE") or "").strip().lower()
    if raw in ("max_f2_under_fpr_cap", "f2"):
        return "max_f2_under_fpr_cap"
    if raw in ("max_f1_under_fpr_cap", "f1"):
        return "max_f1_under_fpr_cap"
    return DEFAULT_BINARY_SELECTION_RULE


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


def _binary_metrics_at_threshold(
    labels: np.ndarray,
    prob_risky: np.ndarray,
    threshold: float,
) -> dict:
    """Compute binary metrics at a given risky-prob threshold."""
    y_true = (np.asarray(labels).astype(np.int64) > 0).astype(np.int64)
    y_pred = (np.asarray(prob_risky).astype(np.float64) >= float(threshold)).astype(np.int64)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    n = max(1, int(y_true.size))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    beta = 2.0
    f2 = (
        (1.0 + beta**2) * precision * recall / (beta**2 * precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / n
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f2": float(f2),
        "fpr": float(fpr),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _find_two_thresholds(
    labels: np.ndarray,
    prob_risky: np.ndarray,
    max_fpr_block: float = 0.05,
    max_fpr_warn: float = 0.15,
    min_recall_block: float = 0.85,
    min_recall_warn: float = 0.60,
) -> dict:
    """
    Find (t_warn, t_block) pair for three-zone binary overlay.

    Zones:
      prob < t_warn              → safe  (silent, green)
      t_warn <= prob < t_block   → warn  (amber, show panel)
      prob >= t_block            → block (red, hold paste)
    """
    y_true = (np.asarray(labels) > 0).astype(np.int64)
    n_pos = int(y_true.sum())

    if n_pos == 0:
        print("WARNING: no positive labels — " "two-threshold sweep skipped.")
        return {}

    best = None
    best_score = -np.inf
    results: list[dict] = []

    t_blocks = np.linspace(0.35, 0.92, 24)

    for t_block in t_blocks:
        pred_b = (prob_risky >= t_block).astype(np.int64)
        tp_b = int(((pred_b == 1) & (y_true == 1)).sum())
        fp_b = int(((pred_b == 1) & (y_true == 0)).sum())
        fn_b = int(((pred_b == 0) & (y_true == 1)).sum())
        tn_b = int(((pred_b == 0) & (y_true == 0)).sum())
        rec_b = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0.0
        fpr_b = fp_b / (fp_b + tn_b) if (fp_b + tn_b) > 0 else 0.0
        prec_b = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0.0

        if fpr_b > max_fpr_block:
            continue
        if rec_b < min_recall_block:
            continue

        t_warns = np.linspace(0.05, t_block - 0.05, 18)
        for t_warn in t_warns:
            in_warn = ((prob_risky >= t_warn) & (prob_risky < t_block)).astype(np.int64)

            tp_w = int(((in_warn == 1) & (y_true == 1)).sum())
            fp_w = int(((in_warn == 1) & (y_true == 0)).sum())

            pred_any = (prob_risky >= t_warn).astype(np.int64)
            tp_any = int(((pred_any == 1) & (y_true == 1)).sum())
            fp_any = int(((pred_any == 1) & (y_true == 0)).sum())
            tn_any = int(((pred_any == 0) & (y_true == 0)).sum())
            rec_any = tp_any / n_pos if n_pos > 0 else 0.0
            fpr_any = fp_any / (fp_any + tn_any) if (fp_any + tn_any) > 0 else 0.0

            if fpr_any > max_fpr_warn:
                continue
            if rec_any < min_recall_warn:
                continue

            prec_w = tp_w / (tp_w + fp_w) if (tp_w + fp_w) > 0 else 0.0

            score = (
                rec_b * 10.0
                + rec_any * 3.0
                + prec_b * 2.0
                + prec_w * 1.0
                - fpr_b * 5.0
                - fpr_any * 2.0
            )

            candidate = {
                "t_warn": float(t_warn),
                "t_block": float(t_block),
                "block_recall": float(rec_b),
                "block_precision": float(prec_b),
                "block_fpr": float(fpr_b),
                "warn_tp": int(tp_w),
                "warn_fp": int(fp_w),
                "warn_precision": float(prec_w),
                "combined_recall": float(rec_any),
                "combined_fpr": float(fpr_any),
                "score": float(score),
            }
            results.append(candidate)
            if score > best_score:
                best_score = score
                best = candidate

    if best is None:
        print(
            "WARNING: no feasible (t_warn, t_block) found "
            f"under constraints. Returning unconstrained best from {len(results)} candidates."
        )
        best = max(results, key=lambda r: r["score"]) if results else {}

    return best or {}


def _write_human_summary(summary: dict, out_path: Path) -> None:
    two = summary.get("two_threshold_recommended") or {}
    rec = summary.get("recommended") or {}

    t_warn = two.get("t_warn", rec.get("prob_threshold_risky", "?"))
    t_block = two.get("t_block", "not set")
    br = float(two.get("block_recall", rec.get("recall", 0)))
    cr = float(two.get("combined_recall", 0))
    cfp = float(two.get("combined_fpr", rec.get("fpr", 0)))
    bfp = float(two.get("block_fpr", 0))

    lines = [
        "Core Sentinel — Binary Threshold Calibration",
        "=" * 52,
        f"Generated  : {summary.get('generated_at', '')}",
        f"Mode       : binary (safe / risky)",
        f"Val samples: {summary.get('n_val', '?')}",
        "",
        "RECOMMENDED THRESHOLDS:",
        f"  Warn  (amber) : prob >= {t_warn}",
        f"  Block (red)   : prob >= {t_block}",
        "",
        "THREE ZONES:",
        f"  prob < {t_warn}  → SAFE  — silent, green dot",
        f"  {t_warn} <= prob < {t_block}  → WARN  — amber, panel shown, proceed allowed",
        f"  prob >= {t_block}  → BLOCK — red, paste held until user acts",
        "",
        "PERFORMANCE:",
        f"  Block recall        : {br:.1%}  — catches {br:.1%} of dangerous pastes",
        f"  Block false pos rate: {bfp:.1%}  — {bfp:.1%} of safe pastes wrongly blocked",
        f"  Combined recall     : {cr:.1%}  — {cr:.1%} of risky pastes get any warning",
        f"  Combined false pos  : {cfp:.1%}  — {cfp:.1%} of safe pastes see any warning",
        "",
        "NEXT STEPS:",
        "  python calibrate_thresholds.py",
        "  → writes to config/risk_policy.json",
        "  Restart overlay to pick up new thresholds.",
    ]

    txt = "\n".join(str(l) for l in lines)
    txt_path = out_path.with_suffix(".txt")
    txt_path.write_text(txt, encoding="utf-8")
    print(f"  Human summary -> {txt_path}")


def apply_calibrated_thresholds() -> None:
    """
    Read PR curve summary from evaluate_test.py, merge calibrated block/warn thresholds into
    config/risk_policy.json under policy["thresholds"][class_name].
    """
    if not CONFIG_PATH.is_file():
        raise SystemExit(f"Missing {CONFIG_PATH}")

    if THRESHOLD_CALIBRATION_PATH.is_file():
        try:
            with open(THRESHOLD_CALIBRATION_PATH, encoding="utf-8") as f:
                calib = json.load(f)
        except Exception:
            calib = None
        if isinstance(calib, dict) and str(calib.get("calibration_mode", "")).lower() == "binary":
            rec = calib.get("recommended") if isinstance(calib.get("recommended"), dict) else {}
            thr = rec.get("prob_threshold_risky")
            if thr is not None:
                with open(CONFIG_PATH, encoding="utf-8") as f:
                    policy = json.load(f)
                policy["inference_binary_labels"] = {
                    "prob_threshold_risky": float(thr),
                    "max_fpr": float(calib.get("max_fpr", DEFAULT_MAX_FPR)),
                    "selected_by": str(calib.get("selection_rule", DEFAULT_BINARY_SELECTION_RULE)),
                    "metrics": {
                        "accuracy": float(rec.get("accuracy", 0.0)),
                        "precision": float(rec.get("precision", 0.0)),
                        "recall": float(rec.get("recall", 0.0)),
                        "f1": float(rec.get("f1", 0.0)),
                        "f2": float(rec.get("f2", 0.0)),
                        "fpr": float(rec.get("fpr", 0.0)),
                    },
                    "source": "reports/threshold_calibration.json",
                }
                policy["_calibrated_at"] = calib.get("generated_at", "")
                policy["_calibration_source"] = "reports/threshold_calibration.json"
                policy["_mode"] = "binary"
                policy["_3class_status"] = "deprecated — binary mode active"
                policy["_note"] = (
                    "Active thresholds: inference_binary_labels (single) "
                    "and inference_binary_two_threshold (warn + block). "
                    "inference_3class_labels is retained for reference only."
                )
                two = calib.get("two_threshold_recommended") or {}
                if two and two.get("t_warn") is not None and two.get("t_block") is not None:
                    policy["inference_binary_two_threshold"] = {
                        "prob_threshold_warn": float(two["t_warn"]),
                        "prob_threshold_block": float(two["t_block"]),
                        "block_recall": float(two.get("block_recall", 0)),
                        "block_fpr": float(two.get("block_fpr", 0)),
                        "combined_recall": float(two.get("combined_recall", 0)),
                        "combined_fpr": float(two.get("combined_fpr", 0)),
                        "note": (
                            "prob < prob_threshold_warn → safe (silent). "
                            "prob_threshold_warn <= prob < prob_threshold_block "
                            "→ warn (amber). "
                            "prob >= prob_threshold_block → block (red)."
                        ),
                    }
                    print(
                        "Two-threshold written to risk_policy.json: "
                        f"warn={two['t_warn']:.3f}, "
                        f"block={two['t_block']:.3f}"
                    )
                with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                    json.dump(policy, f, indent=2)
                try:
                    clear_risk_policy_cache()
                except Exception:
                    pass
                print("Binary threshold updated in config/risk_policy.json from threshold_calibration.json")
                applied_summary = {
                    "generated_at": policy.get("_calibrated_at", ""),
                    "n_val": calib.get("n_val"),
                    "recommended": policy.get("inference_binary_labels", {}),
                    "two_threshold_recommended": {
                        "t_warn": (policy.get("inference_binary_two_threshold") or {}).get(
                            "prob_threshold_warn"
                        ),
                        "t_block": (policy.get("inference_binary_two_threshold") or {}).get(
                            "prob_threshold_block"
                        ),
                        "block_recall": (policy.get("inference_binary_two_threshold") or {}).get(
                            "block_recall", 0
                        ),
                        "combined_recall": (policy.get("inference_binary_two_threshold") or {}).get(
                            "combined_recall", 0
                        ),
                        "combined_fpr": (policy.get("inference_binary_two_threshold") or {}).get(
                            "combined_fpr", 0
                        ),
                        "block_fpr": (policy.get("inference_binary_two_threshold") or {}).get(
                            "block_fpr", 0
                        ),
                    },
                }
                _write_human_summary(applied_summary, REPORTS_DIR / "threshold_applied")
                return

    if not PR_CURVE_SUMMARY_PATH.is_file():
        raise SystemExit(
            f"Missing {PR_CURVE_SUMMARY_PATH}. Run: python evaluate_test.py (generates pr_curve_summary.json)"
        )

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
    """Sweep validation thresholds: high-recall floor, FPR cap, med-recall floor (safety-first order)."""
    max_fpr = float(os.environ.get("GUARDRAIL_CALIB_MAX_FPR", str(DEFAULT_MAX_FPR)))
    min_med_recall = float(os.environ.get("GUARDRAIL_CALIB_MIN_MED_RECALL", str(DEFAULT_MIN_MED_RECALL)))
    min_high_recall = float(os.environ.get("GUARDRAIL_CALIB_MIN_HIGH_RECALL", str(DEFAULT_MIN_HIGH_RECALL)))
    arrow_dir = _resolve_arrow_dir()
    datasets = load_arrow_splits(str(arrow_dir))

    if "validation" not in datasets or datasets["validation"].num_rows == 0:
        raise SystemExit(
            f"No validation split under {arrow_dir}. Need a non-empty validation set to calibrate."
        )

    val_ds = datasets["validation"].rename_column("label", "labels")

    if not MODEL_DIR.is_dir():
        raise SystemExit(f"Model not found at {MODEL_DIR}")

    calib_dir = _resolve_calib_model_dir()
    model = AutoModelForSequenceClassification.from_pretrained(str(calib_dir))
    nl = int(model.classifier.out_features)
    if nl == 2:
        selection_rule = _binary_selection_rule()
        max_fpr_binary = float(os.environ.get("GUARDRAIL_CALIB_MAX_FPR_BINARY", str(max_fpr)))
        args = TrainingArguments(
            output_dir=str(calib_dir / "calibrate_tmp"),
            per_device_eval_batch_size=32,
            report_to="none",
        )
        trainer = Trainer(model=model, args=args)
        pred_out = trainer.predict(val_ds)
        logits = np.asarray(pred_out.predictions)
        labels = np.asarray(pred_out.label_ids)
        probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
        prob_risky = probs[:, 1]

        thresholds = [round(float(x), 2) for x in np.arange(0.10, 0.901, 0.05)]
        rows = [_binary_metrics_at_threshold(labels, prob_risky, t) for t in thresholds]
        feasible = [r for r in rows if float(r["fpr"]) <= max_fpr_binary]
        if selection_rule == "max_f2_under_fpr_cap":
            sort_key = lambda r: (-float(r["f2"]), float(r["fpr"]), -float(r["recall"]), -float(r["precision"]))
            fallback_note = "No threshold met FPR cap; recommended threshold is global best F2."
            selected_note = "Recommended threshold maximizes F2 under FPR cap."
        elif selection_rule == "max_f1_under_fpr_cap":
            sort_key = lambda r: (-float(r["f1"]), float(r["fpr"]), -float(r["recall"]), -float(r["precision"]))
            fallback_note = "No threshold met FPR cap; recommended threshold is global best F1."
            selected_note = "Recommended threshold maximizes F1 under FPR cap."
        else:
            sort_key = lambda r: (-float(r["recall"]), -float(r["precision"]), -float(r["f1"]), float(r["fpr"]))
            fallback_note = "No threshold met FPR cap; recommended threshold is global best recall."
            selected_note = "Recommended threshold maximizes recall under FPR cap."
        if feasible:
            feasible.sort(key=sort_key)
            best = feasible[0]
            note = selected_note
        else:
            rows_sorted = sorted(rows, key=sort_key)
            best = rows_sorted[0]
            note = fallback_note

        print("\n" + "=" * 60)
        print("Threshold calibration (validation set, binary)")
        print("=" * 60)
        print(
            f"  Samples: {len(labels)}  |  max FPR (safe->risky): {max_fpr_binary:.2%}  |  "
            f"grid: 0.10..0.90 step 0.05"
        )
        print(
            f"\n  Recommended threshold: prob_threshold_risky={float(best['threshold']):.2f}\n"
            f"    accuracy={float(best['accuracy']):.6f}  precision={float(best['precision']):.6f}  "
            f"recall={float(best['recall']):.6f}  f1={float(best['f1']):.6f}  "
            f"f2={float(best['f2']):.6f}  fpr={float(best['fpr']):.6f}"
        )

        print("\n--- Two-threshold sweep (warn + block) ---")
        two_thr = _find_two_thresholds(
            labels,
            prob_risky,
            max_fpr_block=float(os.environ.get("GUARDRAIL_MAX_FPR_BLOCK", "0.05")),
            max_fpr_warn=float(os.environ.get("GUARDRAIL_MAX_FPR_WARN", "0.15")),
            min_recall_block=float(os.environ.get("GUARDRAIL_MIN_RECALL_BLOCK", "0.85")),
            min_recall_warn=float(os.environ.get("GUARDRAIL_MIN_RECALL_WARN", "0.60")),
        )
        if two_thr:
            print(
                f"  t_warn={two_thr['t_warn']:.3f}  "
                f"t_block={two_thr['t_block']:.3f}\n"
                f"  block_recall={two_thr['block_recall']:.1%}  "
                f"block_fpr={two_thr['block_fpr']:.1%}\n"
                f"  combined_recall={two_thr['combined_recall']:.1%}  "
                f"combined_fpr={two_thr['combined_fpr']:.1%}"
            )

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = REPORTS_DIR / "threshold_calibration.json"
        summary = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "calibration_mode": "binary",
            "max_fpr": float(max_fpr_binary),
            "selection_rule": selection_rule,
            "n_val": int(len(labels)),
            "grid": rows,
            "recommended": {
                "prob_threshold_risky": float(best["threshold"]),
                "accuracy": float(best["accuracy"]),
                "precision": float(best["precision"]),
                "recall": float(best["recall"]),
                "f1": float(best["f1"]),
                "f2": float(best["f2"]),
                "fpr": float(best["fpr"]),
            },
            "two_threshold_recommended": two_thr,
            "note": note,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Wrote {out_path}")
        _write_human_summary(summary, out_path)
        return
    if nl != 3:
        print(f"Calibration script is for 3-class models; this checkpoint has num_labels={nl}. Exiting.")
        return

    # ── 3-CLASS PATH (DEPRECATED IN BINARY MODE) ─────────
    # Core Sentinel standardised on binary (safe/risky) mode.
    # label_config.json: binary_mode=true, num_labels=2.
    # This branch executes only if a 3-class checkpoint is
    # loaded (num_labels=3 in config.json).
    # To restore 3-class: rebuild Arrow with 3-class labels,
    # retrain, and update label_config.json accordingly.
    # ─────────────────────────────────────────────────────

    args = TrainingArguments(
        output_dir=str(calib_dir / "calibrate_tmp"),
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

    # (recall_high, fpr_non_high_as_high, recall_med, th, tm)
    best_overall: tuple[float, float, float, float, float] | None = None
    feasible: list[tuple[float, float, float, float, float]] = []

    n_high = int((labels == 2).sum())
    n_med = int((labels == 1).sum())
    n_non_high = int((labels != 2).sum())
    if n_high == 0:
        print("WARNING: no 'high' labels in validation set — recall is undefined.")

    for th in ths:
        for tm in tms:
            pred = _predict_buckets_batch(prob_low, prob_med, prob_high, float(th), float(tm))
            tp = int(((pred == 2) & (labels == 2)).sum())
            fn = int(((pred != 2) & (labels == 2)).sum())
            recall_h = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            tp_med = int(((pred == 1) & (labels == 1)).sum())
            fn_med = int(((pred != 1) & (labels == 1)).sum())
            recall_med = tp_med / (tp_med + fn_med) if (tp_med + fn_med) > 0 else 0.0
            fp = int(((pred == 2) & (labels != 2)).sum())
            fpr = fp / n_non_high if n_non_high > 0 else 0.0

            cand = (recall_h, fpr, recall_med, float(th), float(tm))
            if (
                fpr <= max_fpr
                and recall_med >= min_med_recall
                and recall_h >= min_high_recall
            ):
                feasible.append(cand)
            if best_overall is None:
                best_overall = cand
            elif (
                recall_h > best_overall[0]
                or (recall_h == best_overall[0] and recall_med > best_overall[2])
                or (recall_h == best_overall[0] and recall_med == best_overall[2] and fpr < best_overall[1])
            ):
                best_overall = cand

    print("\n" + "=" * 60)
    print("Threshold calibration (validation set, 3-class)")
    print("=" * 60)
    print(
        f"  Samples: {len(labels)}  |  true high: {n_high}  |  true med: {n_med}  |  "
        f"max FPR (non-high -> pred high): {max_fpr:.2%}  |  min med recall: {min_med_recall:.2%}  |  "
        f"min high recall: {min_high_recall:.2%}"
    )

    assert best_overall is not None
    if feasible:
        feasible.sort(key=lambda x: (-x[0], -x[2], x[1], x[3], x[4]))
        r, f, r_med, th_opt, tm_opt = feasible[0]
        print(f"\n  Best under FPR cap: prob_threshold_high={th_opt:.4f}, prob_threshold_med={tm_opt:.4f}")
        print(f"    recall_high={r:.6f}  recall_med={r_med:.6f}  FPR={f:.6f}")
    else:
        print(
            f"\n  No (th, tm) pair satisfied FPR <= {max_fpr:.2%}, med recall >= {min_med_recall:.2%}, "
            f"and high recall >= {min_high_recall:.2%}. Showing unconstrained best recall."
        )
        r, f, r_med, th_opt, tm_opt = best_overall
        print(f"  Best recall (unconstrained FPR={f:.6f}): prob_threshold_high={th_opt:.4f}, prob_threshold_med={tm_opt:.4f}")
        print(f"    recall_high={r:.6f}  recall_med={r_med:.6f}")

    base_pred = _predict_buckets_batch(prob_low, prob_med, prob_high, base_th, base_tm)
    base_tp = int(((base_pred == 2) & (labels == 2)).sum())
    base_fn = int(((base_pred != 2) & (labels == 2)).sum())
    base_recall = base_tp / (base_tp + base_fn) if (base_tp + base_fn) > 0 else 0.0
    base_tp_med = int(((base_pred == 1) & (labels == 1)).sum())
    base_fn_med = int(((base_pred != 1) & (labels == 1)).sum())
    base_recall_med = base_tp_med / (base_tp_med + base_fn_med) if (base_tp_med + base_fn_med) > 0 else 0.0
    base_fp = int(((base_pred == 2) & (labels != 2)).sum())
    base_fpr = base_fp / n_non_high if n_non_high > 0 else 0.0
    base_high_as_low = int(((labels == 2) & (base_pred == 0)).sum())
    base_high_as_med = int(((labels == 2) & (base_pred == 1)).sum())
    print(f"\n  Current policy (inference_3class_labels): th={base_th:.4f}, tm={base_tm:.4f}")
    print(f"    recall_high={base_recall:.6f}  recall_med={base_recall_med:.6f}  FPR={base_fpr:.6f}")
    print(f"    true_high->pred_low={base_high_as_low}  true_high->pred_med={base_high_as_med} (worst misses)")

    pl, pm, ph = float(prob_low[0]), float(prob_med[0]), float(prob_high[0])
    v_model = int(predict_3class_bucket_from_probs(pl, pm, ph, policy))
    v_np = int(_predict_buckets_batch(prob_low[:1], prob_med[:1], prob_high[:1], base_th, base_tm)[0])
    assert v_model == v_np, "Calibration bucket must match infer helper"

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "threshold_calibration.json"
    rec_r, rec_f, rec_m = (feasible[0][0], feasible[0][1], feasible[0][2]) if feasible else (
        best_overall[0],
        best_overall[1],
        best_overall[2],
    )
    opt_pred = _predict_buckets_batch(prob_low, prob_med, prob_high, float(th_opt), float(tm_opt))
    opt_high_as_low = int(((labels == 2) & (opt_pred == 0)).sum())
    opt_high_as_med = int(((labels == 2) & (opt_pred == 1)).sum())
    if feasible:
        print(f"\n  Recommended (feasible) true_high->pred_low={opt_high_as_low}  true_high->pred_med={opt_high_as_med}")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "max_fpr": max_fpr,
        "min_med_recall": min_med_recall,
        "min_high_recall": min_high_recall,
        "n_val": int(len(labels)),
        "n_true_high": n_high,
        "n_true_med": n_med,
        "current_policy": {"prob_threshold_high": base_th, "prob_threshold_med": base_tm},
        "current_metrics": {
            "recall_high": base_recall,
            "recall_med": base_recall_med,
            "fpr_non_high_as_high": base_fpr,
            "true_high_as_low": base_high_as_low,
            "true_high_as_med": base_high_as_med,
        },
        "recommended": {
            "prob_threshold_high": float(th_opt),
            "prob_threshold_med": float(tm_opt),
            "recall_high": float(rec_r),
            "recall_med": float(rec_m),
            "fpr_non_high_as_high": float(rec_f),
            "true_high_as_low": opt_high_as_low,
            "true_high_as_med": opt_high_as_med,
        },
        "note": (
            "Safety-first: feasible pairs require min high recall, FPR cap, and med recall floor. "
            "Copy recommended values into config/risk_policy.json under inference_3class_labels after review."
        ),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Wrote {out_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--validation-sweep", "--val-sweep"):
        run_validation_threshold_sweep()
    else:
        apply_calibrated_thresholds()
