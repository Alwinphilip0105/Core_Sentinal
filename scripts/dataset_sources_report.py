"""
Dataset sources report: source counts, labeled vs unlabeled, class counts, real vs synthetic, avg text length, overlap note.
Loads real_candidate_pool.csv, arrow_datasets, manual_eval, and train_val_manifest.json (if present).
Saves reports/dataset_sources_report.json and reports/source_counts.csv.
"""

import csv
import json
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"
GUARDRAIL_ROOT = ROOT / "core-sentinel-guardrail"
POOL_CSV = REPORTS_DIR / "real_candidate_pool.csv"
MANIFEST_JSON = REPORTS_DIR / "train_val_manifest.json"
REPORT_JSON = REPORTS_DIR / "dataset_sources_report.json"
SOURCE_COUNTS_CSV = REPORTS_DIR / "source_counts.csv"
ARROW_DIR = GUARDRAIL_ROOT / "arrow_datasets"
MANUAL_EVAL_PATH = GUARDRAIL_ROOT / "data" / "manual_eval" / "manual_eval.jsonl"


def load_pool(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(dict(r))
    return rows


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_manual_eval_count(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def load_arrow_splits():
    out = {}
    if not ARROW_DIR.exists():
        return out
    try:
        from datasets import load_from_disk
        ds = load_from_disk(str(ARROW_DIR))
        for split in ("train", "validation", "test"):
            if split in ds:
                sub = ds[split]
                out[split] = {"num_rows": sub.num_rows}
                if "label" in sub.column_names:
                    labels = sub["label"]
                    out[split]["per_class"] = dict(Counter(labels))
                    id2risk = {0: "low", 1: "med", 2: "high"}
                    out[split]["per_class_named"] = {id2risk.get(k, k): v for k, v in out[split]["per_class"].items()}
    except Exception:
        pass
    return out


def main():
    report = {}
    pool = load_pool(POOL_CSV)
    report["pool_total"] = len(pool)
    if pool:
        by_source = Counter(r.get("source", "?") for r in pool)
        report["source_counts"] = dict(by_source)
        labeled = [r for r in pool if r.get("risk")]
        report["pool_labeled"] = len(labeled)
        report["pool_unlabeled"] = len(pool) - len(labeled)
        report["pool_per_class"] = dict(Counter(r["risk"] for r in labeled)) if labeled else {}
        lengths_by_source = {}
        for r in pool:
            src = r.get("source", "?")
            lengths_by_source.setdefault(src, []).append(len(r.get("text", "")))
        report["avg_text_length_per_source"] = {
            src: round(sum(lens) / len(lens), 1) if lens else 0
            for src, lens in lengths_by_source.items()
        }
    else:
        report["source_counts"] = {}
        report["pool_labeled"] = 0
        report["pool_unlabeled"] = 0
        report["pool_per_class"] = {}
        report["avg_text_length_per_source"] = {}

    manifest = load_manifest(MANIFEST_JSON)
    report["real_labeled_count"] = manifest.get("real_labeled_count")
    report["synthetic_count"] = manifest.get("synthetic_count")
    report["train_per_source"] = manifest.get("train_per_source", {})
    report["val_per_source"] = manifest.get("val_per_source", {})

    arrow = load_arrow_splits()
    report["arrow_splits"] = arrow
    report["manual_eval_count"] = load_manual_eval_count(MANUAL_EVAL_PATH)
    report["overlap_note"] = "Overlap of manual_eval with train/val is checked during data pipeline run (see WARNING in data.py output)."

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # source_counts.csv: one row per source with count
    with open(SOURCE_COUNTS_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "count"])
        for src, cnt in report["source_counts"].items():
            w.writerow([src, cnt])
        if manifest.get("train_per_source"):
            for src, cnt in manifest["train_per_source"].items():
                w.writerow([f"train_{src}", cnt])

    # Print summary
    print("Dataset sources report")
    print("  pool_total:", report["pool_total"])
    print("  pool_labeled:", report["pool_labeled"])
    print("  pool_unlabeled:", report["pool_unlabeled"])
    print("  source_counts:", report["source_counts"])
    print("  pool_per_class:", report["pool_per_class"])
    print("  avg_text_length_per_source:", report["avg_text_length_per_source"])
    print("  real_labeled_count:", report["real_labeled_count"])
    print("  synthetic_count:", report["synthetic_count"])
    print("  arrow_splits:", report["arrow_splits"])
    print("  manual_eval_count:", report["manual_eval_count"])
    print("  overlap_note:", report["overlap_note"])
    print(f"Saved {REPORT_JSON} and {SOURCE_COUNTS_CSV}")


if __name__ == "__main__":
    main()
