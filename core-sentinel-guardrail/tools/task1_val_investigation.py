"""One-off: train/val overlap + test set metrics (Task 1)."""
import hashlib
import json
from collections import Counter

import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import BertForSequenceClassification

_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
ARROW = _ROOT / "arrow_datasets"
MODEL = _ROOT / "models" / "tinybert_guardrail"  # root file now 2-class


def example_hash(ex) -> str:
    # Arrow is tokenized; no raw "text" — use input_ids as identity.
    arr = np.asarray(ex["input_ids"], dtype=np.int32)
    return hashlib.md5(arr.tobytes()).hexdigest()


def main() -> None:
    print("=== Task 1a: train/val overlap (by input_ids identity) ===\n")
    ds = load_from_disk(str(ARROW))
    train_ds = ds["train"]
    val_ds = ds["validation"]
    train_hashes = {example_hash(train_ds[i]) for i in range(len(train_ds))}
    val_hashes = [example_hash(val_ds[i]) for i in range(len(val_ds))]
    overlap = [h for h in val_hashes if h in train_hashes]
    print(f"Train size (unique texts): {len(train_hashes)}")
    print(f"Val size (rows):          {len(val_hashes)}")
    n = len(val_hashes)
    pct = (len(overlap) / n) if n else 0.0
    print(f"Overlapping texts:        {len(overlap)} ({pct:.1%})")
    if pct > 0.05:
        print(">>> Leakage > 5%: data leakage risk for val metrics.")
    else:
        print(">>> Overlap <= 5%: no strong text-level leakage signal.")

    print("\n=== Task 1b: test set metrics ===\n")
    test_ds = ds["test"]
    labels = np.asarray(test_ds["label"])

    model = BertForSequenceClassification.from_pretrained(str(MODEL))
    model.eval()

    all_probs, all_preds = [], []
    BATCH = 32
    n_test = len(test_ds)
    with torch.no_grad():
        for i in range(0, n_test, BATCH):
            sub = test_ds[i : i + BATCH]
            enc = {
                "input_ids": torch.tensor(sub["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(sub["attention_mask"], dtype=torch.long),
                "token_type_ids": torch.tensor(sub["token_type_ids"], dtype=torch.long),
            }
            out = model(**enc)
            probs = torch.softmax(out.logits, dim=-1)[:, 1]
            preds = (probs >= 0.5).long()
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())

    labels_list = labels.tolist()
    print(f"Test set metrics (n={len(labels)})")
    print(f"  Accuracy : {accuracy_score(labels_list, all_preds):.4f}")
    print(f"  Precision: {precision_score(labels_list, all_preds, zero_division=0):.4f}")
    print(f"  Recall   : {recall_score(labels_list, all_preds, zero_division=0):.4f}")
    print(f"  F1       : {f1_score(labels_list, all_preds, zero_division=0):.4f}")
    try:
        auc = roc_auc_score(labels_list, all_probs)
    except ValueError as e:
        auc = float("nan")
        print(f"  AUC-ROC  : undefined ({e})")
    else:
        print(f"  AUC-ROC  : {auc:.4f}")
    print(f"\nConfusion matrix (rows=true, cols=pred) [0=safe, 1=risky]:")
    print(confusion_matrix(labels_list, all_preds))
    print(f"\nClass distribution in test:")
    print(dict(Counter(labels_list)))

    _pol = json.loads((_ROOT / "config" / "risk_policy.json").read_text(encoding="utf-8"))
    _two = _pol.get("inference_binary_two_threshold", {})
    t_warn = float(_two.get("prob_threshold_warn", 0.50))
    t_block = float(_two.get("prob_threshold_block", 0.92))
    print(
        f"\n[thresholds] t_warn={t_warn:.3f}  t_block={t_block:.3f}  (from risk_policy.json)"
    )
    zone = []
    for p in all_probs:
        if p >= t_block:
            zone.append("block")
        elif p >= t_warn:
            zone.append("warn")
        else:
            zone.append("safe")
    print(f"\nZone distribution at t_warn={t_warn}, t_block={t_block}:")
    print(dict(Counter(zone)))
    warn_risky = sum(
        1
        for p, l in zip(all_probs, labels_list)
        if t_warn <= p < t_block and l == 1
    )
    block_risky = sum(1 for p, l in zip(all_probs, labels_list) if p >= t_block and l == 1)
    print(f"  risky in warn zone:  {warn_risky}")
    print(f"  risky in block zone: {block_risky}")

    f1v = f1_score(labels_list, all_preds, zero_division=0)
    print("\n>>> Task 1c:")
    if pct > 0.05:
        print(
            f"  Train/val overlap {pct:.1%} — val metrics may be optimistic; "
            "prefer test metrics for trust."
        )
    if f1v >= 0.80:
        print(f"  Test F1={f1v:.3f} — strong generalization despite val leakage risk.")
    elif f1v < 0.70:
        print(f"  Test F1={f1v:.3f} — TEST METRICS POOR — investigate before trusting thresholds.")


if __name__ == "__main__":
    main()
