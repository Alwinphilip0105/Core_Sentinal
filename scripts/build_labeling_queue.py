"""
Build a labeling queue from the real candidate pool.
Loads reports/real_candidate_pool.csv; if a trained model exists, runs inference and ranks by uncertainty.
Exports top 200 rows to reports/labeling_queue.csv for manual labeling.
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"
POOL_CSV = REPORTS_DIR / "real_candidate_pool.csv"
QUEUE_CSV = REPORTS_DIR / "labeling_queue.csv"
GUARDRAIL_ROOT = ROOT / "core-sentinel-guardrail"
MODEL_DIR = GUARDRAIL_ROOT / "models" / "tinybert_guardrail"
TOP_N = 200
ID_TO_RISK = {0: "low", 1: "med", 2: "high"}


def load_pool(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(dict(r))
    return rows


def get_pipeline():
    import warnings
    warnings.filterwarnings("ignore", message=".*flash attention.*")
    from transformers import pipeline
    if not MODEL_DIR.exists():
        return None
    return pipeline(
        "text-classification",
        model=str(MODEL_DIR.resolve()),
        tokenizer=str(MODEL_DIR.resolve()),
        top_k=None,
        device=0 if __import__("torch").cuda.is_available() else -1,
    )


def score_with_uncertainty(pipe, text: str) -> dict:
    """Return predicted_label, confidence, second_best_label, second_best_score, uncertainty (1 - confidence)."""
    if not text or not str(text).strip():
        return {"predicted_label": "med", "confidence": 0.0, "second_best_label": "low", "second_best_score": 0.0, "uncertainty": 1.0}
    out = pipe(text.strip(), truncation=True, max_length=128)
    scores = out[0] if out and isinstance(out[0], list) else out
    sorted_scores = sorted(scores, key=lambda x: x["score"], reverse=True)
    best = sorted_scores[0]
    second = sorted_scores[1] if len(sorted_scores) > 1 else {"label": "med", "score": 0.0}
    label_id = int(best["label"].replace("LABEL_", ""))
    conf = best["score"]
    second_id = int(second["label"].replace("LABEL_", ""))
    return {
        "predicted_label": ID_TO_RISK[label_id],
        "confidence": round(conf, 4),
        "second_best_label": ID_TO_RISK[second_id],
        "second_best_score": round(second["score"], 4),
        "uncertainty": round(1.0 - conf, 4),
    }


def main():
    if len(sys.argv) > 1:
        pool_path = Path(sys.argv[1])
    else:
        pool_path = POOL_CSV
    if len(sys.argv) > 2:
        queue_path = Path(sys.argv[2])
    else:
        queue_path = QUEUE_CSV
    if len(sys.argv) > 3:
        top_n = int(sys.argv[3])
    else:
        top_n = TOP_N

    rows = load_pool(pool_path)
    if not rows:
        print(f"No rows in {pool_path}. Run the data pipeline to build real_candidate_pool.csv first.")
        sys.exit(1)

    pipe = get_pipeline()
    if pipe is not None:
        print("Running model inference to compute uncertainty...")
        for r in rows:
            res = score_with_uncertainty(pipe, r.get("text", ""))
            r["predicted_label"] = res["predicted_label"]
            r["confidence"] = res["confidence"]
            r["second_best_label"] = res["second_best_label"]
            r["second_best_score"] = res["second_best_score"]
            r["uncertainty"] = res["uncertainty"]
        rows.sort(key=lambda x: x.get("uncertainty", 0), reverse=True)
    else:
        print("No trained model found; exporting pool order (no uncertainty ranking).")
        for r in rows:
            r["predicted_label"] = ""
            r["confidence"] = ""
            r["second_best_label"] = ""
            r["second_best_score"] = ""
            r["uncertainty"] = 0.0

    queue = rows[:top_n]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(queue_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["text", "source", "predicted_label", "confidence", "second_best_label", "second_best_score"])
        w.writeheader()
        for r in queue:
            w.writerow({
                "text": r.get("text", ""),
                "source": r.get("source", ""),
                "predicted_label": r.get("predicted_label", ""),
                "confidence": r.get("confidence", ""),
                "second_best_label": r.get("second_best_label", ""),
                "second_best_score": r.get("second_best_score", ""),
            })
    print(f"Exported top {len(queue)} rows to {queue_path}")


if __name__ == "__main__":
    main()
