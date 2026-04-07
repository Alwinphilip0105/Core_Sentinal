import os
import json

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from sklearn.metrics import confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

DATA_PATH = "arrow_datasets/test"
MODEL_PATH = "models/tinybert_guardrail"
REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

TEXT_CANDIDATES = ["text", "content", "input_text", "prompt", "sentence"]
LABEL_CANDIDATES = ["label", "labels", "risk", "risk_label", "target"]


def infer_columns(df):
    cols = list(df.columns)
    if "input_ids" in cols and "label" in cols:
        return None, "label"  # tokenized dataset
    text_col = next((c for c in TEXT_CANDIDATES if c in cols), None)
    label_col = next((c for c in LABEL_CANDIDATES if c in cols), None)
    if text_col is None:
        raise ValueError(f"Could not find text column in {cols}")
    if label_col is None:
        raise ValueError(f"Could not find label column in {cols}")
    return text_col, label_col


def normalize_label(x):
    if isinstance(x, str):
        return x.strip().lower()
    return x


ds = load_from_disk(DATA_PATH)
df = ds.to_pandas()
text_col, label_col = infer_columns(df)
is_tokenized = text_col is None

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# id2label: use low, med, high (guardrail convention); config may have LABEL_0/1/2
num_labels = getattr(model.config, "num_labels", 3)
id2label = {i: ["low", "med", "high"][i] for i in range(num_labels)}
label2id = {v: k for k, v in id2label.items()}

if is_tokenized:
    # Tokenized path: run model on input_ids/attention_mask
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    batch_size = 32
    all_pred_ids = []
    all_scores = []
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i : i + batch_size]
        input_ids = torch.tensor(np.array(batch["input_ids"].tolist()), dtype=torch.long).to(device)
        attention_mask = torch.tensor(np.array(batch["attention_mask"].tolist()), dtype=torch.long).to(device)
        token_type_ids = None
        if "token_type_ids" in batch.columns:
            token_type_ids = torch.tensor(np.array(batch["token_type_ids"].tolist()), dtype=torch.long).to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = out.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        pred_ids = logits.argmax(dim=-1).cpu().tolist()
        all_pred_ids.extend(pred_ids)
        all_scores.extend([float(probs[j, pred_ids[j]]) for j in range(len(pred_ids))])
    pred_labels = [id2label[int(x)] for x in all_pred_ids]
    pred_scores = all_scores
else:
    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        max_length=256,
    )
    texts = df[text_col].astype(str).tolist()
    raw_preds = clf(texts, batch_size=32)
    if raw_preds and isinstance(raw_preds[0], list):
        raw_preds = [x[0] for x in raw_preds]
    pred_labels = [str(x["label"]).lower() for x in raw_preds]
    pred_scores = [float(x["score"]) for x in raw_preds]
    if "medium" in pred_labels or any("medium" in str(x) for x in raw_preds):
        pred_labels = ["med" if x == "medium" else x for x in pred_labels]

true_vals = df[label_col].tolist()
if all(isinstance(x, (int, float)) for x in true_vals):
    true_labels = [id2label.get(int(x), id2label.get(x, "unknown")) for x in true_vals]
else:
    true_labels = [normalize_label(x) for x in true_vals]
    true_labels = ["med" if x == "medium" else x for x in true_labels]

labels_order = ["low", "med", "high"]
present_labels = [x for x in labels_order if x in set(true_labels) or x in set(pred_labels)]
if not present_labels:
    present_labels = sorted(set(true_labels) | set(pred_labels))

cm = confusion_matrix(true_labels, pred_labels, labels=present_labels)
report = classification_report(
    true_labels,
    pred_labels,
    labels=present_labels,
    output_dict=True,
    zero_division=0,
)

cm_df = pd.DataFrame(cm, index=[f"true_{x}" for x in present_labels], columns=[f"pred_{x}" for x in present_labels])
cm_df.to_csv(os.path.join(REPORT_DIR, "confusion_matrix.csv"))

with open(os.path.join(REPORT_DIR, "classification_report.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

# Build output table (no text column if tokenized)
out = df[[label_col]].copy()
if not is_tokenized and text_col in df.columns:
    out[text_col] = df[text_col]
if "risk" in df.columns:
    out["risk"] = df["risk"]
out["true_label"] = true_labels
out["pred_label"] = pred_labels
out["pred_score"] = pred_scores
out["correct"] = out["true_label"] == out["pred_label"]
out.to_csv(os.path.join(REPORT_DIR, "test_predictions.csv"), index=False)

mistakes = out[out["correct"] == False].copy()
mistakes = mistakes.sort_values("pred_score", ascending=False)
mistakes.to_csv(os.path.join(REPORT_DIR, "misclassifications.csv"), index=False)

high_precision = report.get("high", {}).get("precision", None)
high_recall = report.get("high", {}).get("recall", None)
high_f1 = report.get("high", {}).get("f1-score", None)

summary = {
    "labels_order_used": present_labels,
    "num_test_rows": len(out),
    "num_errors": int((~out["correct"]).sum()),
    "accuracy": report.get("accuracy"),
    "high_precision": high_precision,
    "high_recall": high_recall,
    "high_f1": high_f1,
}

with open(os.path.join(REPORT_DIR, "eval_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n=== EVAL SUMMARY ===")
print(json.dumps(summary, indent=2))
print("\n=== CONFUSION MATRIX ===")
print(cm_df.to_string())
print("\n=== FIRST 10 MISTAKES ===")
display_cols = [c for c in ["risk", "true_label", "pred_label", "pred_score"] if c in mistakes.columns]
print(mistakes.head(10)[display_cols].to_string(index=False))
