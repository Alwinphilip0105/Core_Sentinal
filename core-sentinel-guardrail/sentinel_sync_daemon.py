"""
Background sync: SQLite scoring_events → Supabase guardrail_events every SYNC_INTERVAL seconds.

Run alongside main.py (see start_sync_thread in main) or standalone:
  python sentinel_sync_daemon.py

Requires: SUPABASE_URL, SUPABASE_ANON_KEY
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
SYNC_INTERVAL = 60  # seconds
DB_PATH = _ROOT / "logs" / "guardrail.db"
STATE_FILE = _ROOT / "logs" / "supabase_sync_state.json"


def get_last_synced_id() -> int:
    try:
        if STATE_FILE.exists():
            return int(json.loads(STATE_FILE.read_text(encoding="utf-8")).get("last_id", 0))
    except Exception:
        pass
    return 0


def save_last_synced_id(last_id: int) -> None:
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(
            json.dumps(
                {
                    "last_id": last_id,
                    "last_sync": datetime.now(timezone.utc).isoformat(),
                }
            ),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"[sync] state save error: {e}")


def get_supabase_client():
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_ANON_KEY", "").strip()
    if not url or not key:
        return None
    try:
        from supabase import create_client

        return create_client(url, key)
    except Exception as e:
        print(f"[sync] supabase client error: {e}")
        return None


def _risk_score_from_row(d: dict) -> int:
    if "risk_score" in d and d.get("risk_score") is not None:
        try:
            return max(0, min(100, int(d.get("risk_score") or 0)))
        except (TypeError, ValueError):
            pass
    raw = d.get("context_json")
    if not raw:
        return 0
    try:
        ctx = json.loads(raw)
        rs = ctx.get("risk_score", 0)
        return max(0, min(100, int(rs) if rs is not None else 0))
    except (json.JSONDecodeError, TypeError, ValueError):
        return 0


def _row_to_payload(d: dict, target: str) -> dict:
    """Map SQLite row to guardrail_events columns."""
    if target == "scoring_events":
        ctx: dict = {}
        raw = d.get("context_json")
        if raw:
            try:
                ctx = json.loads(raw)
            except json.JSONDecodeError:
                pass
        return {
            "timestamp": str(d.get("created_at") or ""),
            "text_hash": str(d.get("text_hash") or ""),
            "action": str(d.get("action") or ""),
            "risk_score": _risk_score_from_row(d),
            "pii_classes": str(d.get("pii_class") or ""),
            "llm_name": str(ctx.get("llm_name") or d.get("llm_name") or ""),
        }
    # legacy / generic
    return {
        "timestamp": str(
            d.get("timestamp") or d.get("scored_at") or d.get("created_at") or ""
        ),
        "text_hash": str(d.get("text_hash") or ""),
        "action": str(d.get("action") or ""),
        "risk_score": _risk_score_from_row(d),
        "pii_classes": str(d.get("pii_class") or d.get("pii_classes") or ""),
        "llm_name": str(d.get("llm_name") or d.get("window_title") or ""),
    }


def sync_once() -> None:
    if not DB_PATH.exists():
        return
    sb = get_supabase_client()
    if not sb:
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    tables = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    ]

    target = None
    for t in ("scoring_events", "events", "clipboard_events"):
        if t in tables:
            target = t
            break
    if not target:
        conn.close()
        return

    cols = [c[1] for c in conn.execute(f"PRAGMA table_info({target})").fetchall()]

    last_id = get_last_synced_id()

    id_col = "id" if "id" in cols else "rowid"
    try:
        rows = conn.execute(
            f"SELECT * FROM {target} WHERE {id_col} > ? ORDER BY {id_col} ASC LIMIT 200",
            (last_id,),
        ).fetchall()
    except sqlite3.OperationalError:
        conn.close()
        return
    conn.close()

    if not rows:
        return

    synced = 0
    new_last_id = last_id

    for row in rows:
        d = dict(row)
        try:
            row_id = int(d.get("id", 0))
        except (TypeError, ValueError):
            row_id = last_id

        payload = _row_to_payload(d, target)
        try:
            sb.table("guardrail_events").insert(payload).execute()
            synced += 1
            new_last_id = max(new_last_id, row_id)
        except Exception as e:
            err = str(e).lower()
            if (
                "duplicate" in err
                or "23505" in err
                or "unique" in err
                or "already exists" in err
            ):
                new_last_id = max(new_last_id, row_id)
            else:
                print(f"[sync] insert error: {e}")

    if synced > 0:
        save_last_synced_id(new_last_id)
        print(
            f"[sync] {synced} new rows synced to Supabase "
            f"(last_id={new_last_id}) "
            f"at {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC"
        )


def run_daemon() -> None:
    print(f"[sync] Daemon started — syncing every {SYNC_INTERVAL}s")
    while True:
        try:
            sync_once()
        except Exception as e:
            print(f"[sync] error: {e}")
        time.sleep(SYNC_INTERVAL)


def start_sync_thread() -> threading.Thread:
    """Run sync in a daemon thread (call from main.py)."""
    t = threading.Thread(target=run_daemon, daemon=True, name="sentinel-supabase-sync")
    t.start()
    print("[sync] Background sync thread started", flush=True)
    return t


if __name__ == "__main__":
    run_daemon()
