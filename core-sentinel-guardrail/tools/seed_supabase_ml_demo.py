#!/usr/bin/env python3
"""
Insert demo rows into Supabase for the ML Health Dashboard (feedback_corrections + guardrail_events).

Requires:
  SUPABASE_URL
  SUPABASE_SERVICE_ROLE_KEY   (recommended; RLS may block inserts with anon key)

Usage:
  set SUPABASE_URL=https://xxxx.supabase.co
  set SUPABASE_SERVICE_ROLE_KEY=eyJ...
  python tools/seed_supabase_ml_demo.py

Optional:
  set ML_DEMO_CLEAR=1   # DELETE existing demo rows (text_hash prefix demo_ml_) before insert
"""

from __future__ import annotations

import hashlib
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone

try:
    from supabase import create_client
except ImportError:
    print("Install supabase: pip install supabase", file=sys.stderr)
    sys.exit(1)

DEMO_PREFIX = "demo_ml_"
ML_RETRAIN_TARGET = 100

# Matches docs/ml/index.html demo layout (47 wrong corrections; FP UI may show a separate 20-row view).
PATTERN_SPECS: list[tuple[str, str, str, int]] = [
    ("Email (work)", "med", "low", 12),
    ("Phone (office)", "high", "med", 8),
    ("Name + Context", "low", "high", 6),
    ("Address fragment", "high", "low", 5),
    ("Date of birth", "high", "med", 4),
]
# Remaining rows to reach 47
MISC_FILL = 12


def _iso(dt: datetime) -> str:
    return dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def _hash_part(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:24]


def build_feedback_rows(now: datetime) -> list[dict]:
    rng = random.Random(7)
    specs: list[tuple[str, str, str]] = []
    for pii, pred, corr, count in PATTERN_SPECS:
        for _ in range(count):
            specs.append((pii, pred, corr))
    for j in range(MISC_FILL):
        pred = rng.choice(["low", "med", "high"])
        corr = rng.choice(["low", "med", "high"])
        while corr == pred:
            corr = rng.choice(["low", "med", "high"])
        specs.append((f"Misc item {j + 1}", pred, corr))
    if len(specs) != 47:
        raise RuntimeError(f"expected 47 correction specs, got {len(specs)}")

    rows: list[dict] = []
    for i, (pii, pred, corr) in enumerate(specs):
        days_ago = 56 - (i % 8) * 7 - rng.randint(0, 4)
        ts = now - timedelta(
            days=days_ago,
            hours=rng.randint(0, 23),
            minutes=rng.randint(0, 59),
        )
        key = f"{DEMO_PREFIX}fb{i}"
        text_hash = DEMO_PREFIX + _hash_part(key)
        score = 30 + (hash(key) % 45)
        rows.append(
            {
                "timestamp": _iso(ts),
                "recorded_at": _iso(ts),
                "text_hash": text_hash,
                "predicted": pred,
                "correct": corr,
                "source": "demo_seed",
                "feedback_type": "wrong",
                "used_for_training": False,
                "pii_classes": pii,
                "risk_score": score,
            }
        )

    recent_offsets_min = [
        2,
        18,
        60,
        120,
        180,
        300,
        360,
        480,
        30 * 60,
        31 * 60,
        32 * 60,
        33 * 60,
        48 * 60,
        49 * 60,
        50 * 60,
        72 * 60,
        73 * 60,
        96 * 60,
        4 * 24 * 60,
        5 * 24 * 60,
        6 * 24 * 60,
    ]
    for i, off in enumerate(recent_offsets_min):
        if i >= len(rows):
            break
        ts = now - timedelta(minutes=off)
        rows[i]["timestamp"] = _iso(ts)
        rows[i]["recorded_at"] = _iso(ts)

    rows.sort(key=lambda x: x["timestamp"], reverse=True)
    return rows


def build_guardrail_events(now: datetime, n: int = 1200) -> list[dict]:
    out: list[dict] = []
    rng = random.Random(42)
    for i in range(n):
        days_ago = rng.uniform(0, 14)
        ts = now - timedelta(days=days_ago, minutes=rng.randint(0, 1200))
        score = rng.randint(0, 99)
        if score >= 70:
            action = "block"
        elif score >= 40:
            action = "warn"
        else:
            action = "silent"
        key = f"{DEMO_PREFIX}ev{i}"
        text_hash = DEMO_PREFIX + _hash_part(key)
        pii = rng.choice(
            [
                "Email (work)",
                "Phone (office)",
                "Full name",
                "Address fragment",
                "SSN (partial)",
                "API key (obfuscated)",
            ]
        )
        out.append(
            {
                "timestamp": _iso(ts),
                "text_hash": text_hash,
                "action": action,
                "risk_score": score,
                "pii_classes": pii,
                "llm_name": "demo",
            }
        )
    return out


def main() -> int:
    url = (os.environ.get("SUPABASE_URL") or "").strip().rstrip("/")
    key = (os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_ANON_KEY") or "").strip()
    if not url or not key:
        print("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_ANON_KEY).", file=sys.stderr)
        return 1

    clear = os.environ.get("ML_DEMO_CLEAR", "").strip() in ("1", "true", "yes")
    now = datetime.now(timezone.utc)
    sb = create_client(url, key)

    if clear:
        try:
            sb.table("feedback_corrections").delete().eq("source", "demo_seed").execute()
        except Exception as e:
            print(f"[warn] delete feedback_corrections by source: {e}")
        try:
            sb.table("guardrail_events").delete().eq("llm_name", "demo").execute()
        except Exception as e:
            print(f"[warn] delete guardrail_events by llm_name: {e}")

    fb_rows = build_feedback_rows(now)
    ev_rows = build_guardrail_events(now)

    # Batch insert (PostgREST prefers smaller chunks)
    chunk = 100

    def insert_feedback(rows: list[dict]) -> None:
        for i in range(0, len(rows), chunk):
            part = rows[i : i + chunk]
            try:
                sb.table("feedback_corrections").insert(part).execute()
            except Exception as e1:
                slim = [
                    {
                        k: v
                        for k, v in r.items()
                        if k
                        in (
                            "timestamp",
                            "recorded_at",
                            "text_hash",
                            "predicted",
                            "correct",
                            "source",
                            "feedback_type",
                            "used_for_training",
                        )
                    }
                    for r in part
                ]
                try:
                    sb.table("feedback_corrections").insert(slim).execute()
                except Exception as e2:
                    raise e2 from e1
            time.sleep(0.05)

    insert_feedback(fb_rows)
    print(f"Inserted {len(fb_rows)} feedback_corrections (demo). Pending wrong toward retrain: {len(fb_rows)}/{ML_RETRAIN_TARGET}")

    for i in range(0, len(ev_rows), chunk):
        part = ev_rows[i : i + chunk]
        sb.table("guardrail_events").insert(part).execute()
        time.sleep(0.05)
    print(f"Inserted {len(ev_rows)} guardrail_events (demo).")

    print("Done. Open docs/ml/index.html with your Supabase URL/key, or rely on dashboard demo fallback if tables stay empty.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
