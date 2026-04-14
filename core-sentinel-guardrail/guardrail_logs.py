"""
Structured SQLite audit log for Core Sentinel (clipboard guardrail).

Stores:
  - scoring_events: same information as the legacy events.csv queue (hash-only, no raw text)
  - clipboard_events: preprocess skips, duplicate skips, etc.
  - remediation_events: user actions from the remediation panel (snooze, proceed, …)

Database path: logs/guardrail.db (WAL mode). Thread-safe for the infer background writer + UI thread.
Optional legacy CSV: set GUARDRAIL_LEGACY_EVENTS_CSV=1 to also append logs/events.csv

Risk telemetry (JSONL + optional webhook + optional Supabase):
  - logs/risk_telemetry.jsonl — one JSON object per score (risk, risk_score, hash, …).
  - GUARDRAIL_RISK_TELEMETRY=0 disables JSONL append.
  - GUARDRAIL_TELEMETRY_WEBHOOK_URL — POST each record as JSON (async) for your own server / Zapier / etc.
  - GUARDRAIL_TELEMETRY_SUPABASE=1 — insert each record into Supabase (see docs/sql/risk_telemetry.sql).
    Uses SUPABASE_URL + SUPABASE_ANON_KEY; table name from GUARDRAIL_TELEMETRY_SUPABASE_TABLE (default risk_telemetry).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
LOGS_DIR = _ROOT / "logs"
DB_PATH = LOGS_DIR / "guardrail.db"
LEGACY_EVENTS_CSV = LOGS_DIR / "events.csv"
RISK_TELEMETRY_JSONL = LOGS_DIR / "risk_telemetry.jsonl"

_db_lock = threading.Lock()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scoring_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            text_hash TEXT NOT NULL,
            pii_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            action TEXT NOT NULL,
            context_json TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS clipboard_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            event_type TEXT NOT NULL,
            meta_json TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS remediation_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            text_hash TEXT NOT NULL,
            event_type TEXT NOT NULL,
            detail_json TEXT
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_scoring_created ON scoring_events(created_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_scoring_action ON scoring_events(action)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_clipboard_created ON clipboard_events(created_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_remediation_created ON remediation_events(created_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_remediation_type ON remediation_events(event_type)"
    )


def _legacy_csv_enabled() -> bool:
    v = os.environ.get("GUARDRAIL_LEGACY_EVENTS_CSV", "").strip().lower()
    return v in ("1", "true", "yes")


def _append_legacy_csv_row(row: list) -> None:
    """row: [timestamp, text_hash, pii_class, confidence, action]"""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = LEGACY_EVENTS_CSV.exists()
    with open(LEGACY_EVENTS_CSV, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["timestamp", "text_hash", "pii_class", "confidence", "action"])
        w.writerow(row)


def write_scoring_event(
    row: list,
    context_json: str | None = None,
) -> None:
    """
    Persist one scoring row (same shape as legacy CSV).

    row: [timestamp, text_hash, pii_class, confidence_str, action]
    """
    if len(row) < 5:
        return
    ts, text_hash, pii_class, conf_s, action = row[0], row[1], row[2], row[3], row[4]
    try:
        confidence = float(conf_s)
    except (TypeError, ValueError):
        confidence = 0.0
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with _db_lock:
        conn = sqlite3.connect(DB_PATH, timeout=30.0)
        try:
            _ensure_schema(conn)
            conn.execute(
                """
                INSERT INTO scoring_events
                    (created_at, text_hash, pii_class, confidence, action, context_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (ts, text_hash, pii_class, confidence, action, context_json),
            )
            conn.commit()
        finally:
            conn.close()
    if _legacy_csv_enabled():
        try:
            _append_legacy_csv_row(row)
        except Exception:
            pass
    try:
        _append_risk_telemetry_line(row, context_json)
    except Exception:
        pass


def _risk_telemetry_enabled() -> bool:
    v = os.environ.get("GUARDRAIL_RISK_TELEMETRY", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _append_risk_telemetry_line(row: list, context_json: str | None) -> None:
    """
    Append one line to logs/risk_telemetry.jsonl for dashboard / batch review.
    Expects context_json to include risk, risk_score, score_context when provided by infer.py.
    """
    if not _risk_telemetry_enabled():
        return
    if len(row) < 5:
        return
    ts, text_hash, pii_class, conf_s, action = row[0], row[1], row[2], row[3], row[4]
    ctx: dict = {}
    if context_json:
        try:
            ctx = json.loads(context_json)
        except json.JSONDecodeError:
            ctx = {}
    try:
        confidence = float(conf_s)
    except (TypeError, ValueError):
        confidence = 0.0
    rs = ctx.get("risk_score", 0)
    try:
        risk_score = int(rs) if rs is not None else 0
    except (TypeError, ValueError):
        risk_score = 0
    rec = {
        "timestamp": ts,
        "text_hash": text_hash,
        "risk": str(ctx.get("risk", "unknown")).strip().lower(),
        "risk_score": risk_score,
        "pii_class": pii_class,
        "confidence": confidence,
        "action": action,
        "score_context": str(ctx.get("score_context", "")),
    }
    if ctx.get("critical_secret"):
        rec["critical_secret"] = True
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RISK_TELEMETRY_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    _maybe_post_telemetry_webhook(rec)
    _maybe_supabase_telemetry(rec)


def _maybe_supabase_telemetry(rec: dict) -> None:
    """Insert telemetry row into Supabase when GUARDRAIL_TELEMETRY_SUPABASE=1 (async thread)."""
    v = os.environ.get("GUARDRAIL_TELEMETRY_SUPABASE", "").strip().lower()
    if v not in ("1", "true", "yes"):
        return
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_ANON_KEY", "").strip()
    if not url or not key:
        return
    table = (os.environ.get("GUARDRAIL_TELEMETRY_SUPABASE_TABLE") or "risk_telemetry").strip()
    if not table:
        return

    def _run() -> None:
        try:
            from supabase import create_client

            sb = create_client(url, key)
            sb.table(table).insert({"payload": rec}).execute()
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True, name="guardrail-telemetry-supabase").start()


def _maybe_post_telemetry_webhook(rec: dict) -> None:
    url = (os.environ.get("GUARDRAIL_TELEMETRY_WEBHOOK_URL") or "").strip()
    if not url:
        return

    def _run() -> None:
        try:
            body = json.dumps(rec).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=12) as resp:
                resp.read()
        except (urllib.error.URLError, TimeoutError, OSError):
            pass

    threading.Thread(target=_run, daemon=True, name="guardrail-telemetry-webhook").start()


def log_clipboard_event(event_type: str, meta: dict | None = None) -> None:
    """Structured log for paste pipeline (preprocess skip, duplicate, etc.)."""
    if not event_type:
        return
    payload = dict(meta or {})
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with _db_lock:
        conn = sqlite3.connect(DB_PATH, timeout=30.0)
        try:
            _ensure_schema(conn)
            conn.execute(
                """
                INSERT INTO clipboard_events (created_at, event_type, meta_json)
                VALUES (?, ?, ?)
                """,
                (_utc_now_iso(), event_type, json.dumps(payload, ensure_ascii=False)),
            )
            conn.commit()
        finally:
            conn.close()


def _text_hash(text: str) -> str:
    raw = text if isinstance(text, str) else ""
    if not raw.strip():
        return hashlib.sha256(b"").hexdigest()
    return hashlib.sha256(raw.strip().encode("utf-8")).hexdigest()


def log_remediation_event(
    event_type: str,
    detail: dict | None = None,
    *,
    text: str | None = None,
) -> None:
    """User actions from RemediationDialog (snooze, proceed, redact copy, …)."""
    if not event_type:
        return
    th = _text_hash(text or "")
    dj = json.dumps(detail or {}, ensure_ascii=False)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with _db_lock:
        conn = sqlite3.connect(DB_PATH, timeout=30.0)
        try:
            _ensure_schema(conn)
            conn.execute(
                """
                INSERT INTO remediation_events (created_at, text_hash, event_type, detail_json)
                VALUES (?, ?, ?, ?)
                """,
                (_utc_now_iso(), th, event_type, dj),
            )
            conn.commit()
        finally:
            conn.close()


def query_recent_scoring(limit: int = 100) -> list[dict]:
    """Return recent scoring rows as dicts (for debugging / CLI)."""
    if limit < 1:
        limit = 1
    if limit > 10_000:
        limit = 10_000
    if not DB_PATH.exists():
        return []
    with _db_lock:
        conn = sqlite3.connect(DB_PATH, timeout=30.0)
        try:
            _ensure_schema(conn)
            cur = conn.execute(
                """
                SELECT id, created_at, text_hash, pii_class, confidence, action, context_json
                FROM scoring_events
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cur.fetchall()
        finally:
            conn.close()
    out: list[dict] = []
    for r in rows:
        out.append(
            {
                "id": r[0],
                "created_at": r[1],
                "text_hash": r[2],
                "pii_class": r[3],
                "confidence": r[4],
                "action": r[5],
                "context_json": r[6],
            }
        )
    return out


def query_recent_clipboard(limit: int = 50) -> list[dict]:
    if limit < 1:
        limit = 1
    if not DB_PATH.exists():
        return []
    with _db_lock:
        conn = sqlite3.connect(DB_PATH, timeout=30.0)
        try:
            _ensure_schema(conn)
            cur = conn.execute(
                """
                SELECT id, created_at, event_type, meta_json
                FROM clipboard_events
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cur.fetchall()
        finally:
            conn.close()
    return [
        {"id": r[0], "created_at": r[1], "event_type": r[2], "meta_json": r[3]}
        for r in rows
    ]


def export_scoring_events_csv(
    dest: Path | None = None,
    *,
    limit: int = 5000,
) -> Path:
    """
    Write recent scoring rows to a CSV file (for Notepad / Excel).
    Default: logs/guardrail_scoring_export.csv
    """
    dest = dest or (LOGS_DIR / "guardrail_scoring_export.csv")
    rows = query_recent_scoring(limit=limit)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "id",
                "created_at",
                "text_hash",
                "pii_class",
                "confidence",
                "action",
                "context_json",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r["id"],
                    r["created_at"],
                    r["text_hash"],
                    r["pii_class"],
                    r["confidence"],
                    r["action"],
                    r["context_json"] or "",
                ]
            )
    return dest


def query_recent_remediation(limit: int = 50) -> list[dict]:
    if limit < 1:
        limit = 1
    if not DB_PATH.exists():
        return []
    with _db_lock:
        conn = sqlite3.connect(DB_PATH, timeout=30.0)
        try:
            _ensure_schema(conn)
            cur = conn.execute(
                """
                SELECT id, created_at, text_hash, event_type, detail_json
                FROM remediation_events
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cur.fetchall()
        finally:
            conn.close()
    return [
        {"id": r[0], "created_at": r[1], "text_hash": r[2], "event_type": r[3], "detail_json": r[4]}
        for r in rows
    ]
