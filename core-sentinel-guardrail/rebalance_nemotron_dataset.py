"""
Rebalance 3-class Arrow dataset splits by downsampling to equal class counts.

Default behavior:
  - Rebalance train split to equal low/med/high counts (downsample to min class).
  - Keep validation/test unchanged unless explicitly enabled via env vars.

Env vars:
  - GUARDRAIL_ARROW_SAVE_DIR (default: arrow_datasets_nemotron)
  - GUARDRAIL_REBALANCE_VAL ("1" to rebalance validation too)
  - GUARDRAIL_REBALANCE_TEST ("1" to rebalance test too)
"""

from __future__ import annotations

import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk


def _counts(ds: Dataset) -> dict:
    return dict(Counter(ds["risk"]))


def _rebalance_split(ds: Dataset, split_name: str, seed: int = 42) -> Dataset:
    rows = [dict(r) for r in ds]
    by_risk = defaultdict(list)
    for r in rows:
        by_risk[str(r.get("risk", "")).lower()].append(r)

    needed = ["low", "med", "high"]
    if not all(k in by_risk and by_risk[k] for k in needed):
        print(f"[rebalance] {split_name}: skipped (missing one of low/med/high). counts={dict((k, len(v)) for k, v in by_risk.items())}")
        return ds

    target = min(len(by_risk["low"]), len(by_risk["med"]), len(by_risk["high"]))
    rng = random.Random(seed)
    sampled = []
    for k in needed:
        sampled.extend(rng.sample(by_risk[k], target))
    rng.shuffle(sampled)
    out = Dataset.from_list(sampled)
    print(f"[rebalance] {split_name}: target_per_class={target}, new_counts={_counts(out)}")
    return out


def main() -> None:
    root = Path(__file__).resolve().parent
    arrow_dir = Path(os.environ.get("GUARDRAIL_ARROW_SAVE_DIR", "arrow_datasets_nemotron"))
    if not arrow_dir.is_absolute():
        arrow_dir = (root / arrow_dir).resolve()

    if not arrow_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {arrow_dir}")

    rebalance_val = os.environ.get("GUARDRAIL_REBALANCE_VAL", "0") == "1"
    rebalance_test = os.environ.get("GUARDRAIL_REBALANCE_TEST", "0") == "1"

    ds = load_from_disk(str(arrow_dir))
    print("[rebalance] before:")
    for split in ("train", "validation", "test"):
        if split in ds:
            print(f"  {split}: n={ds[split].num_rows}, counts={_counts(ds[split])}")

    new_train = _rebalance_split(ds["train"], "train", seed=42) if "train" in ds else None
    new_val = _rebalance_split(ds["validation"], "validation", seed=43) if ("validation" in ds and rebalance_val) else ds.get("validation")
    new_test = _rebalance_split(ds["test"], "test", seed=44) if ("test" in ds and rebalance_test) else ds.get("test")

    out = DatasetDict({k: v for k, v in {"train": new_train, "validation": new_val, "test": new_test}.items() if v is not None})

    tmp_dir = str(arrow_dir) + "_tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    out.save_to_disk(tmp_dir)
    shutil.rmtree(str(arrow_dir))
    os.rename(tmp_dir, str(arrow_dir))

    print("[rebalance] after:")
    for split in ("train", "validation", "test"):
        if split in out:
            print(f"  {split}: n={out[split].num_rows}, counts={_counts(out[split])}")
    print(f"[rebalance] saved: {arrow_dir}")


if __name__ == "__main__":
    main()

