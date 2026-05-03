#!/usr/bin/env python3
"""
Write reports/pii_category_report.json (sklearn-shaped demo for merge + ML dashboard).

Committed copy lives at reports/pii_category_report.json — overwrite with real eval output.
Regenerate this demo file anytime with: python tools/gen_demo_pii_category_report.py

Does NOT run a real model — use your own eval to produce y_true/y_pred, then:

  from sklearn.metrics import classification_report
  import json
  from pathlib import Path
  report = classification_report(y_true, y_pred, target_names=[...], output_dict=True)
  Path("reports/pii_category_report.json").write_text(json.dumps(report, indent=2))

Then: python tools/update_model_records_per_category.py reports/pii_category_report.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
OUT = _ROOT / "reports" / "pii_category_report.json"

# Valid sklearn output_dict shape (f1-score keys) for merge tool + ML dashboard.
DEMO = {
    "CONTACT": {"precision": 0.91, "recall": 0.935, "f1-score": 0.922, "support": 120},
    "ID": {"precision": 0.88, "recall": 0.905, "f1-score": 0.892, "support": 95},
    "OTHER_PII": {"precision": 0.74, "recall": 0.795, "f1-score": 0.767, "support": 60},
    "NAME": {"precision": 0.22, "recall": 0.35, "f1-score": 0.277, "support": 200},
    "FINANCIAL": {"precision": 0.18, "recall": 0.26, "f1-score": 0.216, "support": 45},
    "LOCATION": {"precision": 0.14, "recall": 0.2, "f1-score": 0.166, "support": 80},
    "AUTH": {"precision": 0.11, "recall": 0.17, "f1-score": 0.135, "support": 30},
    "HEALTH": {"precision": 0.01, "recall": 0.03, "f1-score": 0.017, "support": 12},
    "accuracy": 0.72,
    "macro avg": {"precision": 0.36, "recall": 0.38, "f1-score": 0.37, "support": 642},
    "weighted avg": {"precision": 0.41, "recall": 0.45, "f1-score": 0.43, "support": 642},
}


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(DEMO, indent=2) + "\n", encoding="utf-8")
    print("Wrote", OUT)
    print("Merge: python tools/update_model_records_per_category.py", OUT.relative_to(_ROOT))


if __name__ == "__main__":
    main()
