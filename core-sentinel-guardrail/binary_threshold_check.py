"""
Quick binary threshold comparison on the held-out test split.

Runs fixed thresholds (default: 0.35/0.40/0.45/0.50), prints precision/recall tradeoffs,
and recommends the best production threshold by macro-F1.

Usage:
  .\\.venv\\Scripts\\python.exe binary_threshold_check.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

from data import ARROW_SAVE_DIR
from train import load_arrow_splits

_ROOT = Path(__file__).resolve().parent
MODEL_DIR = _ROOT / "models" / "tinybert_guardrail"
REPORTS_DIR = _ROOT / "reports"


def _metrics_at_threshold(y_true: np.ndarray, p_risky: np.ndarray, thr: float) -> dict:
    pred = (p_risky >= float(thr)).astype(np.int64)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())

    prec_r = tp / (tp + fp) if (tp + fp) else 0.0
    rec_r = tp / (tp + fn) if (tp + fn) else 0.0
    f1_r = (2 * prec_r * rec_r / (prec_r + rec_r)) if (prec_r + rec_r) else 0.0

    prec_s = tn / (tn + fn) if (tn + fn) else 0.0
    rec_s = tn / (tn + fp) if (tn + fp) else 0.0
    f1_s = (2 * prec_s * rec_s / (prec_s + rec_s)) if (prec_s + rec_s) else 0.0

    macro_f1 = 0.5 * (f1_s + f1_r)
    acc = (tp + tn) / max(1, y_true.size)
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "threshold": float(thr),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "safe_recall": float(rec_s),
        "safe_precision": float(prec_s),
        "risky_recall": float(rec_r),
        "risky_precision": float(prec_r),
        "risky_f1": float(f1_r),
        "fpr": float(fpr),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main() -> None:
    datasets = load_arrow_splits(ARROW_SAVE_DIR)
    if "test" not in datasets or datasets["test"].num_rows == 0:
        raise SystemExit("Missing non-empty test split.")
    test_ds = datasets["test"].rename_column("label", "labels")

    if not MODEL_DIR.is_dir():
        raise SystemExit(f"Model not found: {MODEL_DIR}")
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR.resolve()))
    if int(getattr(model.config, "num_labels", 0)) != 2:
        raise SystemExit("binary_threshold_check.py expects a 2-class model.")

    args = TrainingArguments(
        output_dir=str(MODEL_DIR / "threshold_check_tmp"),
        per_device_eval_batch_size=32,
        report_to="none",
    )
    trainer = Trainer(model=model, args=args)
    out = trainer.predict(test_ds)
    logits = np.asarray(out.predictions)
    labels = np.asarray(out.label_ids).astype(np.int64)
    y_true = (labels > 0).astype(np.int64)
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    p_risky = probs[:, 1]

    thresholds = [0.35, 0.40, 0.45, 0.50]
    rows = [_metrics_at_threshold(y_true, p_risky, t) for t in thresholds]
    rows_sorted = sorted(
        rows,
        key=lambda r: (-r["macro_f1"], -r["safe_recall"], -r["risky_precision"], r["fpr"]),
    )
    best = rows_sorted[0]

    print("\n" + "=" * 74)
    print("Binary threshold check (test holdout)")
    print("=" * 74)
    print("  thr  |  acc    macro_f1  safe_rec  risky_prec  risky_rec   fpr")
    for r in rows:
        print(
            f"  {r['threshold']:.2f} |  {r['accuracy']:.4f}  {r['macro_f1']:.4f}    "
            f"{r['safe_recall']:.4f}    {r['risky_precision']:.4f}      {r['risky_recall']:.4f}   {r['fpr']:.4f}"
        )
    print(
        f"\nRecommended production threshold: {best['threshold']:.2f} "
        f"(macro_f1={best['macro_f1']:.4f}, safe_recall={best['safe_recall']:.4f}, "
        f"risky_precision={best['risky_precision']:.4f}, fpr={best['fpr']:.4f})"
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "binary_threshold_check.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_test": int(y_true.size),
        "thresholds": rows,
        "recommended": best,
        "selection_rule": "max macro_f1, then safe_recall, then risky_precision, then min fpr",
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

