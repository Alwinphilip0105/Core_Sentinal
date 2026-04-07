"""
Evaluate the guardrail model on the manual evaluation set.
Loads data/manual_eval/manual_eval.jsonl and models/tinybert_guardrail; writes reports/.
Labels: low, med, high (normalize medium -> med).
"""

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from transformers import pipeline

# Paths: run from repo root (Core_Sentinal); guardrail is core-sentinel-guardrail/
ROOT = Path(__file__).resolve().parent.parent
GUARDRAIL_ROOT = ROOT / "core-sentinel-guardrail"
MANUAL_EVAL_PATH = GUARDRAIL_ROOT / "data" / "manual_eval" / "manual_eval.jsonl"
MODEL_PATH = GUARDRAIL_ROOT / "models" / "tinybert_guardrail"
REPORT_DIR = ROOT / "reports"

TEXT_KEYS = ("text", "content", "excerpt", "prompt", "input")
LABEL_KEYS = ("risk", "label", "category", "classification")


def ensure_manual_eval_file(path: Path) -> None:
    """Create a minimal manual_eval.jsonl if the file does not exist."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    stub = [
        {"text": "Reminder: team standup 9am.", "risk": "low"},
        {"text": "Internal draft—do not share: Q3 roadmap.", "risk": "med"},
        {"text": "Payroll: direct deposit details for 50 employees.", "risk": "high"},
    ]
    with open(path, "w", encoding="utf-8") as f:
        for row in stub:
            f.write(json.dumps(row) + "\n")
    print(f"Created minimal {path} ({len(stub)} examples). Add more rows for real evaluation.")


def load_manual_eval(path: Path | None = None) -> list[dict]:
    """Load manual eval JSONL; normalize labels to low/med/high (medium -> med)."""
    path = path or MANUAL_EVAL_PATH
    ensure_manual_eval_file(path)
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    out = []
    for r in rows:
        text = None
        for k in TEXT_KEYS:
            if k in r and r[k] not in (None, ""):
                text = str(r[k]).strip()
                break
        if not text:
            continue
        raw = None
        for k in LABEL_KEYS:
            if k in r and r[k] is not None:
                raw = str(r[k]).strip().lower()
                break
        if raw == "medium":
            raw = "med"
        risk = raw if raw in ("low", "med", "high") else "med"
        out.append({"text": text, "risk": risk})
    return out


def main():
    if len(sys.argv) > 1:
        eval_path = Path(sys.argv[1])
    else:
        eval_path = MANUAL_EVAL_PATH
    if len(sys.argv) > 2:
        model_path = Path(sys.argv[2])
    else:
        model_path = MODEL_PATH

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run from guardrail dir: python train.py"
        )

    rows = load_manual_eval(eval_path)
    if not rows:
        raise ValueError(f"No rows loaded from {eval_path}.")

    texts = [r["text"] for r in rows]
    true_labels = [r["risk"] for r in rows]

    try:
        import torch
        device = 0 if torch.cuda.is_available() else -1
    except Exception:
        device = -1

    pipe = pipeline(
        "text-classification",
        model=str(model_path.resolve()),
        tokenizer=str(model_path.resolve()),
        device=device,
        top_k=None,
    )

    ID_TO_RISK = {0: "low", 1: "med", 2: "high"}
    pred_labels = []
    pred_scores = []
    for pred in pipe(texts, truncation=True, max_length=128):
        if isinstance(pred, list):
            best = max(pred, key=lambda x: x["score"])
        else:
            best = pred
        lbl = str(best["label"]).lower()
        if lbl.startswith("label_"):
            idx = int(lbl.replace("label_", "").strip())
            lbl = ID_TO_RISK.get(idx, "med")
        if lbl == "medium":
            lbl = "med"
        if lbl not in ("low", "med", "high"):
            lbl = "med"
        pred_labels.append(lbl)
        pred_scores.append(float(best["score"]))

    labels_order = ["low", "med", "high"]
    present = [x for x in labels_order if x in set(true_labels) or x in set(pred_labels)]
    if not present:
        present = sorted(set(true_labels) | set(pred_labels))

    cm = confusion_matrix(true_labels, pred_labels, labels=present)
    report = classification_report(
        true_labels, pred_labels, labels=present, output_dict=True, zero_division=0
    )
    accuracy = report.get("accuracy", 0.0)
    high_metrics = report.get("high", {})
    precision_high = high_metrics.get("precision", 0.0)
    recall_high = high_metrics.get("recall", 0.0)
    f1_high = high_metrics.get("f1-score", 0.0)

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{x}" for x in present],
        columns=[f"pred_{x}" for x in present],
    )
    out_df = pd.DataFrame({
        "text": texts,
        "true_label": true_labels,
        "pred_label": pred_labels,
        "pred_score": pred_scores,
        "correct": [t == p for t, p in zip(true_labels, pred_labels)],
    })

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cm_df.to_csv(REPORT_DIR / "manual_confusion_matrix.csv")
    out_df.to_csv(REPORT_DIR / "manual_eval_predictions.csv", index=False)
    with open(REPORT_DIR / "manual_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    summary = {
        "n_examples": len(rows),
        "accuracy": accuracy,
        "high_precision": precision_high,
        "high_recall": recall_high,
        "high_f1": f1_high,
    }
    with open(REPORT_DIR / "manual_eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Manual evaluation")
    print("=" * 50)
    print(f"Number of examples: {len(rows)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (high): {precision_high:.4f}")
    print(f"Recall (high): {recall_high:.4f}")
    print(f"F1 (high): {f1_high:.4f}")
    print("Confusion matrix:")
    print(cm_df.to_string())
    print("=" * 50)
    print(f"Reports saved to {REPORT_DIR}")


if __name__ == "__main__":
    main()
