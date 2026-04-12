"""One-shot: copy recent scoring rows from logs/guardrail.db to Supabase guardrail_events."""
from __future__ import annotations

import csv
import json
import os
import sqlite3
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def main() -> int:
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_ANON_KEY", "")
    if not url or not key:
        print("ERROR: SUPABASE_URL and SUPABASE_ANON_KEY must be set")
        return 1

    from supabase import create_client

    sb = create_client(url, key)

    db_path = _ROOT / "logs" / "guardrail.db"
    if db_path.exists():
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT created_at, text_hash, pii_class, confidence, action, context_json
                FROM scoring_events
                ORDER BY id DESC
                LIMIT 100
                """
            ).fetchall()
        except sqlite3.OperationalError as e:
            print(f"SQLite error: {e}")
            conn.close()
            return 1

        print(f"Found {len(rows)} events in scoring_events")
        for row in rows:
            ctx: dict = {}
            raw_ctx = row["context_json"]
            if raw_ctx:
                try:
                    ctx = json.loads(raw_ctx)
                except json.JSONDecodeError:
                    pass
            rs = ctx.get("risk_score", 0)
            try:
                risk_score = int(rs) if rs is not None else 0
            except (TypeError, ValueError):
                risk_score = 0
            try:
                sb.table("guardrail_events").insert(
                    {
                        "timestamp": str(row["created_at"] or ""),
                        "text_hash": str(row["text_hash"] or ""),
                        "action": str(row["action"] or ""),
                        "risk_score": risk_score,
                        "pii_classes": str(row["pii_class"] or ""),
                        "llm_name": str(ctx.get("llm_name", "") or ""),
                    }
                ).execute()
            except Exception as e:
                print(f"  skip: {e}")

        print("Sync done")
        conn.close()
        return 0

    csv_path = _ROOT / "logs" / "events.csv"
    if csv_path.exists():
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        print(f"Found {len(rows)} events in CSV")
        for row in rows[:50]:
            try:
                sb.table("guardrail_events").insert(
                    {
                        "timestamp": row.get("timestamp", ""),
                        "text_hash": row.get("text_hash", ""),
                        "action": row.get("action", ""),
                        "risk_score": int(row.get("risk_score", 0) or 0),
                        "pii_classes": row.get("pii_class", ""),
                        "llm_name": row.get("llm_name", ""),
                    }
                ).execute()
            except Exception as e:
                print(f"  skip: {e}")
        print("CSV sync done")
        return 0

    print("No local data found — need logs/guardrail.db or logs/events.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
