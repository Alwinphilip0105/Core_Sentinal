"""
Map sklearn classification_report(..., output_dict=True) into
docs/data/model_records.json → metrics.per_category (for ML dashboard).

No torch/transformers imports — safe for lightweight CLI use.
"""

from __future__ import annotations

import json
from pathlib import Path

_SKIP = frozenset({"accuracy", "macro avg", "weighted avg"})


def per_category_f1_from_classification_report(report: dict) -> dict[str, dict[str, float]]:
    """Extract per-label F1 only; keys match classification_report rows (e.g. CONTACT, low)."""
    out: dict[str, dict[str, float]] = {}
    for cat, row in report.items():
        if cat in _SKIP:
            continue
        if not isinstance(row, dict):
            continue
        raw = row.get("f1-score")
        if raw is None:
            continue
        try:
            out[str(cat)] = {"f1": round(float(raw), 3)}
        except (TypeError, ValueError):
            continue
    return out


def merge_per_category_into_model_records(
    per_category: dict[str, dict[str, float]],
    *,
    mr_path: Path | None = None,
) -> Path:
    """Read model_records.json, set metrics.per_category, write back (indent=2)."""
    repo_root = Path(__file__).resolve().parent.parent
    path = mr_path or (repo_root / "docs" / "data" / "model_records.json")
    mr = json.loads(path.read_text(encoding="utf-8-sig"))
    mr.setdefault("metrics", {})
    mr["metrics"]["per_category"] = per_category
    path.write_text(json.dumps(mr, indent=2) + "\n", encoding="utf-8")
    return path
