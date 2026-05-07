#!/usr/bin/env python3
"""
Export approved Layer Inspector corrections to training JSONL.

Reads:  logs/span_corrections_approved.jsonl
Writes: data/user_feedback/export_span_corrections.jsonl

Next: python merge_feedback_to_training.py   # bubble feedback → export.jsonl
      python data.py multi_real_synthetic    # merges both exports automatically

Or only span export exists — data.py still picks up export_span_corrections.jsonl.
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from span_feedback_store import SPAN_APPROVED_PATH, export_approved_to_training_jsonl  # noqa: E402


def main() -> int:
    if not SPAN_APPROVED_PATH.is_file():
        print(f"No approved file yet: {SPAN_APPROVED_PATH}", file=sys.stderr)
        print("Approve pending rows first: python tools/review_span_corrections.py approve 1")
    stats = export_approved_to_training_jsonl()
    print(
        f"Wrote {stats['written']} rows → {stats['path']}\n"
        f"  skipped_bad_json={stats.get('skipped_bad_json', 0)} "
        f"skipped_dup_or_short={stats.get('skipped_dup_or_short', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
