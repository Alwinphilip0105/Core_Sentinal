#!/usr/bin/env python3
"""
Review Layer Inspector span corrections before they enter the training export.

  python tools/review_span_corrections.py stats
  python tools/review_span_corrections.py list-pending [--last N]
  python tools/review_span_corrections.py approve LINE   # 1-based line index in pending file
  python tools/review_span_corrections.py export-training [--no-dedupe]

After export-training, merge into the active-learning JSONL consumed by data.py:

  python tools/merge_span_corrections_to_training.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from span_feedback_store import (  # noqa: E402
    SPAN_APPROVED_PATH,
    SPAN_PENDING_PATH,
    approve_pending_by_line_number,
    count_approved_lines,
    count_pending_lines,
    export_approved_to_training_jsonl,
    read_pending_records,
)


def cmd_stats() -> int:
    print(f"pending: {count_pending_lines()}  ({SPAN_PENDING_PATH})")
    print(f"approved: {count_approved_lines()}  ({SPAN_APPROVED_PATH})")
    return 0


def cmd_list_pending(last: int | None) -> int:
    rows = read_pending_records()
    for i, r in enumerate(rows, start=1):
        if last is not None and i <= len(rows) - last:
            continue
        rid = r.get("id", "?")
        cr = r.get("corrected_risk", "?")
        pr = r.get("predicted_risk", "?")
        nfix = len(r.get("span_fixes") or [])
        prev = str(r.get("text", "")).replace("\n", " ")[:120]
        print(f"{i:3}  id={rid}  {pr}->{cr}  fixes={nfix}  text={prev!r}")
    return 0


def cmd_approve(line: int) -> int:
    row = approve_pending_by_line_number(line)
    if row is None:
        print(f"No pending row at line {line}.", file=sys.stderr)
        return 1
    print(f"Approved id={row.get('id')} → {SPAN_APPROVED_PATH}")
    return 0


def cmd_export_training(dedupe: bool) -> int:
    stats = export_approved_to_training_jsonl(dedupe=dedupe)
    print(
        f"Wrote {stats['written']} training rows → {stats['path']}\n"
        f"  skipped_bad_json={stats['skipped_bad_json']} "
        f"skipped_dup_or_short={stats['skipped_dup_or_short']}"
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Review span corrections for active learning")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("stats", help="Count pending vs approved lines")

    pl = sub.add_parser("list-pending", help="Print pending rows (compact)")
    pl.add_argument("--last", type=int, default=None, help="Show only last N rows")

    ap = sub.add_parser("approve", help="Approve one pending row by 1-based line number")
    ap.add_argument("line", type=int, help="Line number from list-pending order (full file)")

    ex = sub.add_parser("export-training", help="Write export_span_corrections.jsonl from approved")
    ex.add_argument("--no-dedupe", action="store_true", help="Allow duplicate snippets")

    args = p.parse_args()
    if args.cmd == "stats":
        return cmd_stats()
    if args.cmd == "list-pending":
        return cmd_list_pending(args.last)
    if args.cmd == "approve":
        return cmd_approve(args.line)
    if args.cmd == "export-training":
        return cmd_export_training(dedupe=not args.no_dedupe)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
