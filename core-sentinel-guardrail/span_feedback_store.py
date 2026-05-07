"""
Span-level and document-level corrections from the Layer Inspector for active learning.

Flow:
  1. User submits corrections → logs/span_corrections_pending.jsonl
  2. Periodic review → tools/review_span_corrections.py (copy/approve lines)
  3. Approved lines live in logs/span_corrections_approved.jsonl
  4. tools/merge_span_corrections_to_training.py → data/user_feedback/export_span_corrections.jsonl
  5. feedback_store.load_user_feedback_export_rows() merges export.jsonl + export_span_corrections.jsonl into training.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent
SPAN_PENDING_PATH = _ROOT / "logs" / "span_corrections_pending.jsonl"
SPAN_APPROVED_PATH = _ROOT / "logs" / "span_corrections_approved.jsonl"
SPAN_TRAINING_EXPORT_PATH = _ROOT / "data" / "user_feedback" / "export_span_corrections.jsonl"


def _normalize_risk(raw: str) -> str:
    s = str(raw or "").strip().lower()
    if s == "medium":
        return "med"
    if s in ("low", "med", "high"):
        return s
    return "low"


def phi_class_to_training_risk(phi_class: str) -> str:
    """Map collapsed PHI label to 3-class training risk for snippet rows."""
    c = str(phi_class or "").strip().upper()
    if c in ("", "O", "SAFE", "NONE", "NEGATIVE"):
        return "low"
    if c in ("FINANCIAL", "HEALTH", "AUTH", "ID"):
        return "high"
    return "med"


def record_pending_span_correction(
    text: str,
    predicted_risk: str,
    corrected_risk: str,
    span_fixes: list[dict[str, Any]],
    *,
    risk_score: int | None = None,
    decision: str | None = None,
    source: str = "layer_inspector",
) -> str:
    """
    Append one pending review record. span_fixes entries: start, end, was_class, correct_class (optional notes).

    Returns record id (uuid4).
    """
    SPAN_PENDING_PATH.parent.mkdir(parents=True, exist_ok=True)
    rid = str(uuid.uuid4())
    row: dict[str, Any] = {
        "id": rid,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "text": text.strip(),
        "predicted_risk": _normalize_risk(predicted_risk),
        "corrected_risk": _normalize_risk(corrected_risk),
        "span_fixes": list(span_fixes or []),
        "review_status": "pending",
        "source": str(source),
    }
    if risk_score is not None:
        row["risk_score"] = int(risk_score)
    if decision:
        row["decision"] = str(decision)
    with open(SPAN_PENDING_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rid


def count_pending_lines() -> int:
    if not SPAN_PENDING_PATH.is_file():
        return 0
    n = 0
    with open(SPAN_PENDING_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def count_approved_lines() -> int:
    if not SPAN_APPROVED_PATH.is_file():
        return 0
    n = 0
    with open(SPAN_APPROVED_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def read_pending_records() -> list[dict[str, Any]]:
    if not SPAN_PENDING_PATH.is_file():
        return []
    out: list[dict[str, Any]] = []
    with open(SPAN_PENDING_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def append_approved_record(row: dict[str, Any]) -> None:
    """Append a curated row to the approved ledger (typically after human review)."""
    SPAN_APPROVED_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = dict(row)
    row["review_status"] = "approved"
    if "approved_at" not in row:
        row["approved_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(SPAN_APPROVED_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def approve_pending_by_line_number(line_1_based: int) -> dict[str, Any] | None:
    """Copy one line from pending to approved by 1-based line index. Returns appended row or None."""
    pending = read_pending_records()
    if line_1_based < 1 or line_1_based > len(pending):
        return None
    row = pending[line_1_based - 1]
    append_approved_record(row)
    return row


def export_approved_to_training_jsonl(
    out_path: Path | None = None,
    *,
    snippet_pad: int = 96,
    dedupe: bool = True,
) -> dict[str, Any]:
    """
    Read span_corrections_approved.jsonl and write training rows:
      - One row per record: full text + corrected_risk
      - Optional extra rows: context windows around each span_fix (risk from correct_class)

    Returns stats dict.
    """
    out_path = Path(out_path or SPAN_TRAINING_EXPORT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not SPAN_APPROVED_PATH.is_file():
        out_path.write_text("", encoding="utf-8")
        return {"written": 0, "skipped": 0, "path": str(out_path.resolve())}

    try:
        from data import _normalize_for_leakage, MAX_TEXT_LEN, MIN_TEXT_LEN
    except ImportError:
        _normalize_for_leakage = lambda t: " ".join(str(t).lower().split())  # type: ignore
        MIN_TEXT_LEN, MAX_TEXT_LEN = 20, 1000

    rows_out: list[dict[str, str]] = []
    seen: set[str] = set()
    skipped_bad_json = 0
    skipped_dup_or_short = 0

    def _add(text: str, risk: str) -> None:
        nonlocal skipped_dup_or_short
        t = str(text or "").strip()
        if len(t) < MIN_TEXT_LEN or len(t) > MAX_TEXT_LEN:
            skipped_dup_or_short += 1
            return
        key = _normalize_for_leakage(t) + "|" + risk
        if dedupe and key in seen:
            skipped_dup_or_short += 1
            return
        seen.add(key)
        rows_out.append({"text": t, "risk": risk})

    with open(SPAN_APPROVED_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped_bad_json += 1
                continue
            text = str(rec.get("text") or "").strip()
            risk = _normalize_risk(str(rec.get("corrected_risk") or ""))
            if text and risk in ("low", "med", "high"):
                _add(text, risk)

            for fix in rec.get("span_fixes") or []:
                if not isinstance(fix, dict):
                    continue
                try:
                    s = int(fix.get("start", -1))
                    e = int(fix.get("end", -1))
                except (TypeError, ValueError):
                    continue
                if s < 0 or e <= s or s >= len(text):
                    continue
                cc = str(fix.get("correct_class") or "").strip()
                snippet_risk = phi_class_to_training_risk(cc)
                lo = max(0, s - snippet_pad)
                hi = min(len(text), e + snippet_pad)
                snippet = text[lo:hi].strip()
                if snippet:
                    _add(snippet, snippet_risk)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps({"text": r["text"], "risk": r["risk"]}, ensure_ascii=False) + "\n")

    return {
        "written": len(rows_out),
        "skipped_bad_json": skipped_bad_json,
        "skipped_dup_or_short": skipped_dup_or_short,
        "path": str(out_path.resolve()),
    }
