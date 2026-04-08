"""
Export bubble feedback (logs/feedback_store.jsonl + logs/hash_index.jsonl) into
data/user_feedback/export.jsonl for the 3-class training pipeline.

Workflow (multi-machine):
  1. git pull — collect feedback JSONL from all laptops into one repo (or merge files).
  2. python merge_feedback_to_training.py
  3. python data.py multi_real_synthetic   # picks up export.jsonl automatically
  4. python train.py
  5. python evaluate_test.py && python calibrate_thresholds.py   # optional
  6. Optional: --mark-used to flag exported rows in feedback_store.jsonl

Text is taken from hash_index previews (first 500 chars per paste). For full-length
training text, extend record_feedback() to store full text elsewhere later.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_guardrail_dir = Path(__file__).resolve().parent
if str(_guardrail_dir) not in sys.path:
    sys.path.insert(0, str(_guardrail_dir))

from feedback_store import mark_feedback_used, export_feedback_to_training_jsonl


def main() -> int:
    p = argparse.ArgumentParser(description="Export feedback JSONL for data.py / train.py")
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output JSONL path (default: data/user_feedback/export.jsonl)",
    )
    p.add_argument(
        "--mark-used",
        action="store_true",
        help="Set used_for_training on exported rows in feedback_store.jsonl",
    )
    args = p.parse_args()
    stats = export_feedback_to_training_jsonl(out_path=args.out)
    print(
        f"Wrote {stats['written']} rows to {stats['path']}\n"
        f"  rows using full text (feedback_fulltext.jsonl): {stats.get('from_fulltext', 0)}\n"
        f"  skipped (no text): {stats.get('skipped_no_text', stats.get('skipped_no_preview', 0))}\n"
        f"  skipped (bad label): {stats['skipped_bad_label']}"
    )
    if args.mark_used and stats.get("hashes"):
        mark_feedback_used(list(stats["hashes"]))
        print(f"Marked {len(stats['hashes'])} feedback row(s) as used_for_training.")
    elif stats["written"] == 0:
        print("Nothing to export — check logs/feedback_store.jsonl and logs/hash_index.jsonl.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
