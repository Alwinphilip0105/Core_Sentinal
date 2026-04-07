"""
Error analysis on the validation set: confusion matrix, accuracy, and sampled
misclassifications (especially ID and OTHER_PII). Writes to reports/error_analysis.json.
Run from core-sentinel-guardrail: python run_error_analysis.py
"""

import json
import random
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

_ARROW_DIR = Path(__file__).resolve().parent / "arrow_datasets"
_MODEL_DIR = Path(__file__).resolve().parent / "models" / "tinybert_guardrail"
_REPORTS_DIR = Path(__file__).resolve().parent / "reports"
_MAX_EXAMPLES_PER_CLASS = 15  # max misclassified examples to store per (true_class, pred_class) or per class


def main():
    val_path = _ARROW_DIR / "validation"
    if not val_path.is_dir():
        raise FileNotFoundError(f"Validation split not found at {val_path}. Run data.py first.")
    val_ds = load_from_disk(str(val_path))
    n_val = val_ds.num_rows
    if n_val == 0:
        raise ValueError("Validation set is empty.")

    # Label config
    label_config_path = _ARROW_DIR / "label_config.json"
    if not label_config_path.exists():
        raise FileNotFoundError(f"Label config not found at {label_config_path}.")
    with open(label_config_path, encoding="utf-8") as f:
        label_config = json.load(f)
    num_labels = int(label_config["num_labels"])
    id2label_raw = label_config.get("id2label", {})
    id2label = {int(k): str(v) for k, v in id2label_raw.items()}
    label_names = [id2label.get(i, str(i)) for i in range(num_labels)]

    # Model + tokenizer
    if not _MODEL_DIR.exists():
        raise FileNotFoundError(f"Model not found at {_MODEL_DIR}. Run train.py first.")
    tokenizer = AutoTokenizer.from_pretrained(str(_MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(_MODEL_DIR))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Get labels from dataset (column is "label")
    labels_col = "labels" if "labels" in val_ds.column_names else "label"
    true_labels = np.array(val_ds[labels_col], dtype=np.int64)

    # Predict in batches (use existing input_ids/attention_mask from dataset)
    batch_size = 32
    all_preds = []
    for i in range(0, n_val, batch_size):
        batch = val_ds.select(range(i, min(i + batch_size, n_val)))
        input_ids = torch.tensor(batch["input_ids"], dtype=torch.long, device=device)
        attention_mask = torch.tensor(batch["attention_mask"], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.append(preds)
    preds = np.concatenate(all_preds, axis=0)

    # Metrics
    acc = accuracy_score(true_labels, preds)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        true_labels, preds, average="macro", zero_division=0
    )
    cm = confusion_matrix(true_labels, preds, labels=list(range(num_labels)))

    # Build per-class error lists: (true_label, pred_label) -> list of indices
    errors_by_pair = {}
    for idx in range(n_val):
        t, p = int(true_labels[idx]), int(preds[idx])
        if t != p:
            key = (t, p)
            errors_by_pair.setdefault(key, []).append(idx)

    # Sample misclassified examples for ID and OTHER_PII (and optionally all)
    focus_classes = ["ID", "OTHER_PII"]  # plus we can add others
    focus_ids = [i for i, name in enumerate(label_names) if name in focus_classes]
    sampled = {}
    for class_id in range(num_labels):
        name = label_names[class_id]
        # True = class_id, predicted something else
        missed = [idx for (t, p) in errors_by_pair if t == class_id for idx in errors_by_pair[(t, p)]]
        # Predicted = class_id, true was something else
        fp = [idx for (t, p) in errors_by_pair if p == class_id for idx in errors_by_pair[(t, p)]]
        random.seed(42)
        missed_sample = random.sample(missed, min(_MAX_EXAMPLES_PER_CLASS, len(missed))) if missed else []
        fp_sample = random.sample(fp, min(_MAX_EXAMPLES_PER_CLASS, len(fp))) if fp else []
        # Decode text from input_ids for display
        def get_text(idx):
            ids = val_ds[int(idx)]["input_ids"]
            return tokenizer.decode(ids, skip_special_tokens=True).strip()[:200]
        sampled[name] = {
            "n_true": int((true_labels == class_id).sum()),
            "n_pred": int((preds == class_id).sum()),
            "n_correct": int(((true_labels == class_id) & (preds == class_id)).sum()),
            "n_missed": len(missed),
            "n_false_positive": len(fp),
            "examples_missed": [
                {"idx": int(idx), "text_preview": get_text(idx), "true": name, "pred": label_names[int(preds[idx])]}
                for idx in missed_sample
            ],
            "examples_false_positive": [
                {"idx": int(idx), "text_preview": get_text(idx), "true": label_names[int(true_labels[idx])], "pred": name}
                for idx in fp_sample
            ],
        }

    # Summary
    report = {
        "n_validation": n_val,
        "accuracy": float(acc),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f_macro),
        "confusion_matrix": cm.tolist(),
        "label_names": label_names,
        "per_class": sampled,
    }
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _REPORTS_DIR / "error_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Error analysis written to {out_path}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro P/R/F1: {p_macro:.4f} / {r_macro:.4f} / {f_macro:.4f}")
    print(f"  Confusion matrix (rows=true, cols=pred):")
    print(f"    Labels: {label_names}")
    for i, row in enumerate(cm):
        print(f"    {label_names[i]}: {row.tolist()}")
    print(f"  Per-class details and sampled errors: see {out_path}")


if __name__ == "__main__":
    main()
