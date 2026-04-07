import os
import re
import json
import hashlib
from collections import Counter

import pandas as pd
from datasets import load_from_disk

ROOT = "arrow_datasets"
SPLITS = ["train", "validation", "test"]
REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

TEXT_CANDIDATES = ["text", "content", "input_text", "prompt", "sentence"]
LABEL_CANDIDATES = ["label", "labels", "risk", "risk_label", "target"]

def normalize_text(x: str) -> str:
    x = "" if x is None else str(x)
    x = x.strip().lower()
    x = re.sub(r"\s+", " ", x)
    return x

def text_hash(x: str) -> str:
    return hashlib.md5(normalize_text(x).encode("utf-8")).hexdigest()

def infer_columns(ds):
    cols = ds.column_names
    text_col = next((c for c in TEXT_CANDIDATES if c in cols), None)
    label_col = next((c for c in LABEL_CANDIDATES if c in cols), None)
    if text_col is None:
        raise ValueError(f"Could not find text column in {cols}")
    if label_col is None:
        raise ValueError(f"Could not find label column in {cols}")
    return text_col, label_col

loaded = {}
all_hashes = {}
summary = {}

for split in SPLITS:
    path = os.path.join(ROOT, split)
    ds = load_from_disk(path)
    text_col, label_col = infer_columns(ds)

    df = ds.to_pandas()[[text_col, label_col]].copy()
    df["norm_text"] = df[text_col].astype(str).map(normalize_text)
    df["text_hash"] = df[text_col].astype(str).map(text_hash)

    loaded[split] = df
    all_hashes[split] = set(df["text_hash"])

    label_counts = df[label_col].value_counts(dropna=False).to_dict()
    dup_count = int(df["text_hash"].duplicated().sum())

    summary[split] = {
        "rows": int(len(df)),
        "text_column": text_col,
        "label_column": label_col,
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "exact_duplicate_rows_after_normalization": dup_count,
    }

cross_overlap = {}
pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
for a, b in pairs:
    overlap = all_hashes[a].intersection(all_hashes[b])
    cross_overlap[f"{a}_vs_{b}"] = {
        "overlap_count": int(len(overlap)),
        "overlap_pct_of_" + a: round(len(overlap) / max(len(all_hashes[a]), 1), 6),
        "overlap_pct_of_" + b: round(len(overlap) / max(len(all_hashes[b]), 1), 6),
    }

overlap_examples = []
for a, b in pairs:
    overlap = all_hashes[a].intersection(all_hashes[b])
    if overlap:
        sample_hashes = list(overlap)[:20]
        left = loaded[a][loaded[a]["text_hash"].isin(sample_hashes)][["norm_text"]].copy()
        left["from_split"] = a
        right = loaded[b][loaded[b]["text_hash"].isin(sample_hashes)][["norm_text"]].copy()
        right["from_split"] = b
        merged = pd.concat([left, right], ignore_index=True)
        overlap_examples.append(merged)

if overlap_examples:
    overlap_df = pd.concat(overlap_examples, ignore_index=True).drop_duplicates()
    overlap_df.to_csv(os.path.join(REPORT_DIR, "cross_split_overlap_examples.csv"), index=False)

with open(os.path.join(REPORT_DIR, "dataset_audit.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            "summary": summary,
            "cross_split_overlap": cross_overlap,
        },
        f,
        indent=2,
        ensure_ascii=False,
    )

print(json.dumps({"summary": summary, "cross_split_overlap": cross_overlap}, indent=2))

for split in SPLITS:
    df = loaded[split]
    print(f"\n=== {split.upper()} SAMPLE ROWS ===")
    print(df.head(5)[[c for c in df.columns if c not in ['norm_text', 'text_hash']]].to_string(index=False))
