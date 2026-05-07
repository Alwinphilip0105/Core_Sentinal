"""
Background sync: SQLite scoring_events → Supabase guardrail_events every SYNC_INTERVAL seconds.

Run alongside main.py (see start_sync_thread in main) or standalone:
  python sentinel_sync_daemon.py

Requires: SUPABASE_URL, SUPABASE_ANON_KEY
"""

from __future__ import annotations

import json
import os
import socket
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def _load_env_files() -> None:
    """Later paths override earlier; override=True lets .env beat stale OS env."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for path in (_ROOT.parent / ".env", _ROOT / ".env"):
        if path.is_file():
            load_dotenv(path, override=True)


_load_env_files()

SYNC_INTERVAL = 60  # seconds
DB_PATH = _ROOT / "logs" / "guardrail.db"
STATE_FILE = _ROOT / "logs" / "supabase_sync_state.json"

# Throttle repeated "offline / bad URL" logs (seconds between messages).
_NETWORK_WARN_INTERVAL_SEC = 300.0
_last_network_warn_ts = 0.0
_logged_placeholder_url = False


def is_placeholder_supabase_url(url: str | None) -> bool:
    """True when SUPABASE_URL is empty or still a docs/template hostname (not a real project)."""
    raw = (url or "").strip()
    if not raw:
        return True
    try:
        from urllib.parse import urlparse

        host = (urlparse(raw).hostname or "").lower()
    except Exception:
        return True
    if not host:
        return True
    if "<" in host or ">" in host:
        return True
    if host in ("your-project.supabase.co", "xxxx.supabase.co", "example.supabase.co"):
        return True
    if "your-project" in host:
        return True
    return False


def _is_dns_or_network_error(exc: BaseException) -> bool:
    """True when the host cannot be reached (DNS, offline, bad SUPABASE_URL)."""
    chain: list[BaseException] = []
    e: BaseException | None = exc
    seen: set[int] = set()
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        chain.append(e)
        e = e.__cause__ or e.__context__

    for err in chain:
        if isinstance(err, socket.gaierror):
            return True
        if isinstance(err, (TimeoutError, ConnectionError, OSError)):
            errno = getattr(err, "errno", None)
            if errno in (11001, 11002, 101, 113, -2, -3):
                return True
            if errno is not None and "getaddrinfo" in str(err).lower():
                return True
        msg = str(err).lower()
        if "getaddrinfo" in msg or "name or service not known" in msg or "nodename nor servname" in msg:
            return True
    return False


def _maybe_log_network_hint(exc: BaseException) -> None:
    global _last_network_warn_ts
    now = time.time()
    if now - _last_network_warn_ts < _NETWORK_WARN_INTERVAL_SEC:
        return
    _last_network_warn_ts = now
    url = os.environ.get("SUPABASE_URL", "").strip()
    host = ""
    if url:
        try:
            from urllib.parse import urlparse

            host = urlparse(url).netloc or ""
        except Exception:
            pass
    print(
        "[sync] Cannot reach Supabase (DNS/network). "
        f"Error: {exc}\n"
        "      Fix: set SUPABASE_URL to https://<project-ref>.supabase.co, "
        "check internet/VPN/DNS, then restart the app.\n"
        f"      Current host from env: {host or '(parse failed)'}",
        flush=True,
    )


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
    global _logged_placeholder_url
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_ANON_KEY", "").strip()
    if not url or not key:
        return None
    if is_placeholder_supabase_url(url):
        if not _logged_placeholder_url:
            _logged_placeholder_url = True
            print(
                "[sync] SUPABASE_URL is still a template (e.g. <your-project>.supabase.co). "
                "Replace it with your project URL from Supabase Settings -> API, then restart. "
                "Sync disabled until then.",
                flush=True,
            )
        return None
    try:
        from urllib.parse import urlparse

        host = (urlparse(url).hostname or "").strip()
        if not host or "." not in host:
            print(f"[sync] SUPABASE_URL must be like https://xxxx.supabase.co (bad host in URL).", flush=True)
            return None
    except Exception:
        pass
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
            if _is_dns_or_network_error(e):
                _maybe_log_network_hint(e)
                break
            err = str(e).lower()
            if (
                "duplicate" in err
                or "23505" in err
                or "unique" in err
                or "already exists" in err
            ):
                new_last_id = max(new_last_id, row_id)
            else:
                print(f"[sync] insert error: {e}", flush=True)

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
