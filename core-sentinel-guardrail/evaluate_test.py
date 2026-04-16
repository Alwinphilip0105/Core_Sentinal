"""
Formal test-set evaluation: load saved TinyBERT guardrail + tokenized Arrow test split,
run Trainer.evaluate(), print accuracy, macro F1, per-class P/R/F1, and high-class metrics.
Then PR-curve analysis (AUPRC, threshold at precision ≥0.85), confusion matrix, classification report,
and a safety_profile block (true high predicted as low/med vs benign flagged as high).

Run from core-sentinel-guardrail/ (or set GUARDRAIL_ARROW_SAVE_DIR like data.py).
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
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from data import ARROW_SAVE_DIR
from train import LABEL_CONFIG_FILENAME, load_arrow_splits, make_compute_metrics

# Saved fine-tuned weights (same as train.SAVE_DIR)
MODEL_DIR = _ROOT / "models" / "tinybert_guardrail"


def _resolve_arrow_dir() -> Path:
    raw = os.environ.get("GUARDRAIL_ARROW_SAVE_DIR", ARROW_SAVE_DIR)
    p = Path(raw)
    return p if p.is_absolute() else (_ROOT / p)


def _safety_profile_argmax(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
) -> dict:
    """
    Aggregate errors aligned with guardrail priorities: missing true high is worse than
    flagging benign text. Uses argmax predictions (same as confusion matrix).
    """
    def _idx(name: str) -> int | None:
        return label_names.index(name) if name in label_names else None

    lo = _idx("low")
    med = _idx("med")
    hi = _idx("high")
    if hi is None:
        return {"note": "No 'high' label in id2label; safety_profile skipped."}

    n = int(len(y_true))
    mask_th = y_true == hi
    n_true_high = int(mask_th.sum())
    n_true_low = int((y_true == lo).sum()) if lo is not None else 0
    n_true_med = int((y_true == med).sum()) if med is not None else 0

    th_as_lo = int(((y_true == hi) & (lo is not None) & (y_pred == lo)).sum())
    th_as_med = int(((y_true == hi) & (med is not None) & (y_pred == med)).sum())
    th_as_hi = int(((y_true == hi) & (y_pred == hi)).sum())

    high_miss = int(n_true_high - th_as_hi)
    high_recall = float(th_as_hi / n_true_high) if n_true_high > 0 else 0.0

    low_as_hi = int(((y_true == lo) & (lo is not None) & (y_pred == hi)).sum())
    med_as_hi = int(((y_true == med) & (med is not None) & (y_pred == hi)).sum())

    worst = float(th_as_lo / n_true_high) if n_true_high > 0 else 0.0

    return {
        "n_test": n,
        "n_true_low": n_true_low,
        "n_true_med": n_true_med,
        "n_true_high": n_true_high,
        "true_high_predicted_as_low": th_as_lo,
        "true_high_predicted_as_med": th_as_med,
        "true_high_predicted_as_high": th_as_hi,
        "high_miss_count": high_miss,
        "high_recall_argmax": round(high_recall, 6),
        "true_high_as_low_rate_given_true_high": round(worst, 6),
        "true_low_predicted_as_high": low_as_hi,
        "true_med_predicted_as_high": med_as_hi,
        "interpretation": (
            "Worst failures: true_high_predicted_as_low (high treated as safe). "
            "Secondary: true_low_predicted_as_high (noisy benign flags). "
            "Threshold calibration should prioritize high recall before tightening headline F1."
        ),
    }


def _load_label_config(arrow_dir: Path, model_num_labels: int) -> tuple[int, dict[int, str]]:
    """Prefer arrow_dir/label_config.json; else fall back to model.config.num_labels + id2label."""
    path = arrow_dir / LABEL_CONFIG_FILENAME
    if path.exists():
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        num_labels = int(cfg["num_labels"])
        id2label = {int(k): str(v) for k, v in cfg.get("id2label", {}).items()}
        return num_labels, id2label
    # Fallback: trust saved model (3-class default)
    return model_num_labels, {0: "low", 1: "med", 2: "high"}


def main() -> None:
    arrow_dir = _resolve_arrow_dir()
    datasets = load_arrow_splits(str(arrow_dir))

    if "test" not in datasets or datasets["test"].num_rows == 0:
        raise SystemExit(
            f"No non-empty test split under {arrow_dir / 'test'}. "
            "Regenerate data with data.py so test exists."
        )

    test_ds = datasets["test"].rename_column("label", "labels")

    if not MODEL_DIR.is_dir():
        raise SystemExit(f"Model directory not found: {MODEL_DIR}. Train with train.py first.")

    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR.resolve()))
    _ = AutoTokenizer.from_pretrained(str(MODEL_DIR.resolve()))  # ensures vocab files present alongside weights

    num_labels_cfg, id2label = _load_label_config(arrow_dir, int(getattr(model.config, "num_labels", 3)))
    model_nl = int(getattr(model.config, "num_labels", num_labels_cfg))
    if model_nl != num_labels_cfg:
        print(
            f"WARNING: model num_labels={model_nl} vs label_config num_labels={num_labels_cfg}; "
            "using model.config for metrics."
        )
        num_labels_cfg = model_nl
        id2label = {int(k): str(v) for k, v in (model.config.id2label or {}).items()}

    compute_metrics = make_compute_metrics(num_labels_cfg, id2label)

    eval_out = MODEL_DIR / "eval_test_run"
    args = TrainingArguments(
        output_dir=str(eval_out),
        per_device_eval_batch_size=16,
        dataloader_drop_last=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    # Single evaluate() pass — matches validation metric definitions in train.py
    metrics = trainer.evaluate(eval_dataset=test_ds)

    print("\n" + "=" * 60)
    print("Test split evaluation (trainer.evaluate)")
    print("=" * 60)
    for k in sorted(metrics.keys()):
        v = metrics[k]
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    # Trainer prefixes sklearn metrics with eval_; highlight high-class when present
    if "high" in [id2label.get(i, str(i)) for i in range(num_labels_cfg)]:
        print("\n--- High-risk class (from same eval pass) ---")
        for k in ("eval_precision_high", "eval_recall_high", "eval_f1_high"):
            if k in metrics:
                print(f"  {k}: {metrics[k]:.6f}")

    # --- PR curve + confusion matrix (raw softmax on full test split; separate from evaluate()) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    torch_cols = [c for c in ("input_ids", "attention_mask", "labels") if c in test_ds.column_names]
    if not torch_cols:
        print("\nWARNING: test split missing tokenized columns; skipping PR curve section.")
        print("\nDone.")
        return

    # Plain Python lists avoid datasets numpy/torch formatters (some versions import torchvision.io.VideoReader)
    test_batches = test_ds.with_format("python", columns=torch_cols)
    loader = DataLoader(test_batches, batch_size=32, shuffle=False)

    def _to_long_on_device(x: torch.Tensor | np.ndarray | list, dev: torch.device) -> torch.Tensor:
        # HF Dataset + DataLoader with format("python") can yield [seq_len] list of tensors
        # each shaped [batch] (column-major); model expects [batch, seq_len].
        if isinstance(x, torch.Tensor):
            return x.to(device=dev, dtype=torch.long)
        if isinstance(x, list) and len(x) and isinstance(x[0], torch.Tensor):
            return torch.stack(x, dim=0).transpose(0, 1).to(device=dev, dtype=torch.long)
        return torch.as_tensor(x, dtype=torch.long, device=dev)

    all_probs: list[np.ndarray] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = _to_long_on_device(batch["input_ids"], device)
            attention_mask = _to_long_on_device(batch["attention_mask"], device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = softmax(logits, dim=-1).cpu().numpy()
            all_probs.extend(probs)
            lab = batch["labels"]
            if isinstance(lab, torch.Tensor):
                all_labels.extend(lab.cpu().numpy().tolist())
            else:
                all_labels.extend(np.asarray(lab).reshape(-1).tolist())

    all_probs_arr = np.asarray(all_probs, dtype=np.float64)
    all_labels_arr = np.asarray(all_labels, dtype=np.int64)
    n_samples = len(all_labels_arr)

    label_names = [str(id2label.get(i, f"class_{i}")) for i in range(num_labels_cfg)]

    pr_rows: list[tuple[str, float, str, str]] = []
    classes_out: dict = {}

    for i in range(num_labels_cfg):
        cname = label_names[i]
        y_true_binary = (all_labels_arr == i).astype(np.int32)
        y_scores = all_probs_arr[:, i]

        if y_true_binary.sum() == 0 or y_true_binary.sum() == n_samples:
            pr_auc = float("nan")
            threshold_at_p85 = None
            recall_at_p85 = None
            precision, recall, thresholds = precision_recall_curve(y_true_binary, y_scores)
        else:
            precision, recall, thresholds = precision_recall_curve(y_true_binary, y_scores)
            # sklearn: len(precision)==len(recall)==len(thresholds)+1; pair (p,r) at each threshold
            order = np.argsort(recall)
            pr_auc = float(auc(recall[order], precision[order]))

            threshold_at_p85 = None
            recall_at_p85 = None
            if len(thresholds) > 0:
                for p, r, t in zip(precision[1:], recall[1:], thresholds):
                    if p >= 0.85:
                        threshold_at_p85 = float(t)
                        recall_at_p85 = float(r)
                        break

        pr_rows.append(
            (
                cname,
                pr_auc,
                "-" if threshold_at_p85 is None else f"{threshold_at_p85:.4f}",
                "-" if recall_at_p85 is None else f"{recall_at_p85:.4f}",
            )
        )
        auprc_json = None if (isinstance(pr_auc, float) and np.isnan(pr_auc)) else pr_auc
        classes_out[cname] = {
            "auprc": auprc_json,
            "threshold_at_precision_085": threshold_at_p85,
            "recall_at_threshold": recall_at_p85,
        }

    print("\n" + "=" * 60)
    print("PR curve analysis (test split, softmax)")
    print("=" * 60)
    col_w = max(len("Class"), max(len(r[0]) for r in pr_rows))
    print(f"  {'Class':<{col_w}} | AUPRC | Threshold@P>=0.85 | Recall@that_threshold")
    for name, auprc, th_s, rec_s in pr_rows:
        aup_s = "nan" if np.isnan(auprc) else f"{auprc:.4f}"
        print(f"  {name:<{col_w}} | {aup_s:>5} | {th_s:>18} | {rec_s:>21}")

    preds = [int(all_probs_arr[j].argmax()) for j in range(n_samples)]
    print("\n" + "=" * 60)
    print("Confusion matrix (rows=true, cols=pred)")
    print("=" * 60)
    cm = confusion_matrix(all_labels_arr, preds, labels=list(range(num_labels_cfg)))
    print(cm)
    print("\n" + "=" * 60)
    print("Classification report")
    print("=" * 60)
    present_labels = sorted(set(all_labels))
    present_names = [label_names[i] for i in present_labels]

    print(
        classification_report(
            all_labels,
            preds,
            labels=present_labels,
            target_names=present_names,
            zero_division=0,
        )
    )

    preds_arr = np.asarray(preds, dtype=np.int64)
    safety = _safety_profile_argmax(all_labels_arr, preds_arr, label_names)
    print("\n" + "=" * 60)
    print("Safety-oriented profile (argmax; see README: safety-first metrics)")
    print("=" * 60)
    for key in (
        "n_true_high",
        "true_high_predicted_as_low",
        "true_high_predicted_as_med",
        "high_miss_count",
        "high_recall_argmax",
        "true_low_predicted_as_high",
        "true_med_predicted_as_high",
    ):
        if key in safety:
            print(f"  {key}: {safety[key]}")

    roc_auc_macro: float | None = None
    if n_samples > 0 and num_labels_cfg >= 2:
        try:
            u = np.unique(all_labels_arr)
            if len(u) >= 2:
                roc_auc_macro = float(
                    roc_auc_score(
                        all_labels_arr,
                        all_probs_arr,
                        multi_class="ovr",
                        average="macro",
                    )
                )
        except ValueError:
            roc_auc_macro = None

    reports_dir = _ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / "pr_curve_summary.json"
    out_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(MODEL_DIR.resolve().as_posix()),
        "test_split_size": int(n_samples),
        "classes": classes_out,
        "safety_profile": safety,
        "roc_auc_macro": roc_auc_macro,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\n  Wrote {summary_path}")
    if roc_auc_macro is not None:
        print(f"\n  Macro ROC-AUC (OvR, test split): {roc_auc_macro:.6f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
