"""
Store user corrections for guardrail labels; supports retraining workflows and stats.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
FEEDBACK_PATH = _ROOT / "logs" / "feedback_store.jsonl"
HASH_INDEX_PATH = _ROOT / "logs" / "hash_index.jsonl"
# Full pasted text per hash (append-only; last line per hash wins). Used for training export.
FEEDBACK_FULLTEXT_PATH = _ROOT / "logs" / "feedback_fulltext.jsonl"
# Merged from bubble feedback for data.py → train.py (see merge_feedback_to_training.py)
USER_FEEDBACK_EXPORT_PATH = _ROOT / "data" / "user_feedback" / "export.jsonl"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def record_feedback(
    original_text: str,
    predicted_label: str,
    correct_label: str,
    source: str = "user",
) -> None:
    """
    Save one feedback row. Writes hash_index (500-char preview) and feedback_fulltext
    (full string) so training export can recover complete pastes.
    """
    FEEDBACK_PATH.parent.mkdir(exist_ok=True)
    text_hash = _sha256(original_text)

    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "text_hash": text_hash,
        "predicted": predicted_label,
        "correct": correct_label,
        "source": source,
        "used_for_training": False,
    }
    with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

    # Write hash → text mapping (truncated to 500 chars)
    idx_row = {
        "text_hash": text_hash,
        "text_preview": original_text[:500],
    }
    with open(HASH_INDEX_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(idx_row) + "\n")

    if os.environ.get("GUARDRAIL_FEEDBACK_NO_FULLTEXT", "").strip().lower() not in (
        "1",
        "true",
        "yes",
    ):
        ft_row = {"text_hash": text_hash, "text": original_text}
        with open(FEEDBACK_FULLTEXT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(ft_row, ensure_ascii=False) + "\n")


def get_pending_feedback(min_count: int = 30) -> list:
    """
    Returns feedback rows not yet used for training.
    Returns empty list if fewer than min_count pending.
    """
    if not FEEDBACK_PATH.exists():
        return []
    rows = []
    with open(FEEDBACK_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    if not r.get("used_for_training"):
                        rows.append(r)
                except json.JSONDecodeError:
                    continue
    if len(rows) < min_count:
        return []
    return rows


def mark_feedback_used(text_hashes: list) -> None:
    """Mark rows as used so they are not retrained on twice."""
    if not FEEDBACK_PATH.exists():
        return
    updated = []
    hash_set = set(text_hashes)
    with open(FEEDBACK_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("text_hash") in hash_set:
                    r["used_for_training"] = True
                updated.append(json.dumps(r))
            except json.JSONDecodeError:
                updated.append(line)
    with open(FEEDBACK_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(updated) + "\n")


def should_trigger_retrain(min_count: int = 30) -> bool:
    return len(get_pending_feedback(min_count)) >= min_count


def _normalize_feedback_risk(raw: str) -> str | None:
    s = str(raw or "").strip().lower()
    if s in ("low", "med", "high"):
        return s
    if s in ("medium",):
        return "med"
    return None


def _hash_to_fulltext_map() -> dict[str, str]:
    """Last full text per hash from feedback_fulltext.jsonl (append-only; last line wins)."""
    if not FEEDBACK_FULLTEXT_PATH.exists():
        return {}
    m: dict[str, str] = {}
    with open(FEEDBACK_FULLTEXT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
                h = o.get("text_hash")
                if h:
                    m[str(h)] = str(o.get("text") or "")
            except (json.JSONDecodeError, TypeError):
                continue
    return m


def _hash_to_preview_map() -> dict[str, str]:
    """Last text_preview per hash (previews are up to 500 chars; see record_feedback)."""
    if not HASH_INDEX_PATH.exists():
        return {}
    m: dict[str, str] = {}
    with open(HASH_INDEX_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
                h = o.get("text_hash")
                if h:
                    m[h] = str(o.get("text_preview") or "")
            except (json.JSONDecodeError, TypeError):
                continue
    return m


def _latest_feedback_by_hash() -> dict[str, dict]:
    """Last row per text_hash wins."""
    if not FEEDBACK_PATH.exists():
        return {}
    latest: dict[str, dict] = {}
    with open(FEEDBACK_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            h = r.get("text_hash")
            if h:
                latest[str(h)] = r
    return latest


def export_feedback_to_training_jsonl(
    out_path: Path | None = None,
) -> dict[str, object]:
    """
    Join feedback rows with full text (preferred) or hash_index previews and write JSONL.

    Each line: {"text": str, "risk": "low"|"med"|"high"}
    Resolution order: ``logs/feedback_fulltext.jsonl`` (full paste), else
    ``logs/hash_index.jsonl`` (500-char preview; older data only).

    Returns stats dict: written, skipped_no_text, skipped_bad_label, from_fulltext, hashes, path.
    """
    out_path = Path(out_path or USER_FEEDBACK_EXPORT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fulltext = _hash_to_fulltext_map()
    previews = _hash_to_preview_map()
    latest = _latest_feedback_by_hash()
    written = 0
    skipped_no_text = 0
    skipped_bad_label = 0
    from_fulltext = 0
    hashes_out: list[str] = []
    lines_out: list[str] = []

    for h, r in latest.items():
        risk = _normalize_feedback_risk(str(r.get("correct", "")))
        if risk is None:
            skipped_bad_label += 1
            continue
        raw = (fulltext.get(h) or "").strip()
        if raw:
            text = raw
            from_fulltext += 1
        else:
            text = (previews.get(h) or "").strip()
        if not text:
            skipped_no_text += 1
            continue
        row = {"text": text, "risk": risk}
        lines_out.append(json.dumps(row, ensure_ascii=False))
        written += 1
        hashes_out.append(h)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out))
        if lines_out:
            f.write("\n")

    return {
        "written": written,
        "skipped_no_text": skipped_no_text,
        "skipped_no_preview": skipped_no_text,
        "skipped_bad_label": skipped_bad_label,
        "from_fulltext": from_fulltext,
        "hashes": hashes_out,
        "path": str(out_path.resolve()),
    }


def load_user_feedback_export_rows(
    path: str | Path | None = None,
) -> list[dict[str, str]]:
    """
    Load rows produced by export_feedback_to_training_jsonl for merging into data.py
    (multi_real_synthetic). Returns [{"text", "risk"}, ...].
    """
    raw = os.environ.get("GUARDRAIL_USER_FEEDBACK_JSONL", "").strip()
    p = Path(path) if path else (Path(raw) if raw else USER_FEEDBACK_EXPORT_PATH)
    if not p.exists():
        return []
    out: list[dict[str, str]] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = str(r.get("text") or r.get("content") or "").strip()
            risk = _normalize_feedback_risk(str(r.get("risk") or r.get("label") or ""))
            if text and risk:
                out.append({"text": text, "risk": risk})
    return out


def get_feedback_stats() -> dict:
    """Summary stats for the ML report."""
    if not FEEDBACK_PATH.exists():
        return {"total": 0, "pending": 0, "used": 0, "corrections": {}}
    total = pending = used = 0
    corrections = Counter()
    with open(FEEDBACK_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                total += 1
                if r.get("used_for_training"):
                    used += 1
                else:
                    pending += 1
                key = f"{r['predicted']}->{r['correct']}"
                corrections[key] += 1
            except json.JSONDecodeError:
                continue
    return {
        "total": total,
        "pending": pending,
        "used": used,
        "corrections": dict(corrections),
    }


if __name__ == "__main__":
    # test with dummy data
    record_feedback(
        "My SSN is 123-45-6789",
        predicted_label="high",
        correct_label="high",
        source="test",
    )
    record_feedback(
        "Call me at 555-1234",
        predicted_label="high",
        correct_label="med",
        source="test",
    )
    stats = get_feedback_stats()
    print("Feedback stats:", stats)
    print("Should retrain (threshold 30):", should_trigger_retrain(30))
    print("Should retrain (threshold 1):", should_trigger_retrain(1))
