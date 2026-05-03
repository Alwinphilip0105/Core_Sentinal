#!/usr/bin/env python3
"""
Merge a sklearn classification_report output_dict (JSON) into
docs/data/model_records.json as metrics.per_category.

Usage (from core-sentinel-guardrail/, PowerShell — no line-continuation backslash):
  python tools/update_model_records_per_category.py reports/pii_category_report.json

Step 1 — In your notebook or eval script (y_true / y_pred must exist there):
  from sklearn.metrics import classification_report
  import json
  from pathlib import Path
  report = classification_report(
      y_true, y_pred,
      target_names=[
          "CONTACT", "ID", "OTHER_PII", "NAME",
          "FINANCIAL", "LOCATION", "AUTH", "HEALTH",
      ],
      output_dict=True,
  )
  Path("reports/pii_category_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

Step 2 — Merge (this tool):
  python tools/update_model_records_per_category.py reports/pii_category_report.json

Step 3 — Reload docs/ml/ in the browser; console may show:
  [ML dashboard] Category F1 loaded from model_records (8 categories)

Demo report only (no real eval): python tools/gen_demo_pii_category_report.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model_records_per_category import (  # noqa: E402
    merge_per_category_into_model_records,
    per_category_f1_from_classification_report,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "report_json",
        type=Path,
        help="JSON file: sklearn classification_report(..., output_dict=True)",
    )
    ap.add_argument(
        "--mr-path",
        type=Path,
        default=None,
        help="Override model_records.json path (default: ../docs/data/model_records.json)",
    )
    args = ap.parse_args()
    if not args.report_json.is_file():
        raise SystemExit(
            f"Report file not found: {args.report_json.resolve()}\n"
            "Pass a real path to JSON from classification_report(..., output_dict=True). "
            '"path/to/report.json" in docs is only an example.'
        )
    report = json.loads(args.report_json.read_text(encoding="utf-8-sig"))
    if not isinstance(report, dict):
        raise SystemExit("report_json must be a JSON object")
    per_cat = per_category_f1_from_classification_report(report)
    if not per_cat:
        raise SystemExit("No per-class rows with f1-score found in report")
    out = merge_per_category_into_model_records(per_cat, mr_path=args.mr_path)
    print("per_category written:", per_cat)
    print("->", out)


if __name__ == "__main__":
    main()
