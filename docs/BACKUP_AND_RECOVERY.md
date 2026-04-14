# Backup, GitHub sync, and recovery

This project uses **Git** (typically **GitHub**) as the source of truth for **code**, **documentation**, **SQL migrations**, and **versioned config** that is safe to share. Runtime data that can be large or sensitive stays **out of the repository** by default (see `.gitignore`).

## What is stored in Git

| In repo | Purpose |
|--------|---------|
| Application source | Restored by `git clone` / `git pull` |
| `docs/sql/*.sql` | Supabase schema you run in the SQL Editor |
| `core-sentinel-guardrail/config/risk_policy.json` | Calibrated thresholds (regenerate with `evaluate_test.py` + `calibrate_thresholds.py` if you change data) |
| `core-sentinel-guardrail/pyproject.toml`, `uv.lock` | Reproducible Python dependencies |

## What is not in Git (protect or copy separately)

| Not in repo | Why | How to avoid data loss |
|-------------|-----|------------------------|
| `logs/guardrail.db`, `*.jsonl` telemetry | Local audit trail; can be large | **Supabase**: scoring sync to `guardrail_events` + optional `risk_telemetry` table (see `RISK_TELEMETRY.md`) |
| `models/tinybert_guardrail/` | Large; rebuilt with `train.py` | Retrain from `data.py` output, or keep your own backup of the folder |
| `arrow_datasets/` | Dataset cache | Regenerate with `data.py` |
| `.env` | Secrets | Store keys in a password manager; recreate `.env` after clone |
| `config/user_settings.json` | Per-machine UI prefs | Gitignored; optional manual export |

## Routine sync (developers)

1. **Before work:** `git pull` on `main` (or your branch).
2. **After meaningful changes:** `git add`, `git commit` with a clear message, `git push`.
3. Prefer **small, focused commits** so history is easy to bisect and restore.

## Recovering after disk loss or a new machine

1. **Clone** the repository and install Python dependencies (`README.md` / **`docs/WINDOWS_SETUP.md`**).
2. **Recreate `.env`** with `GEMINI_API_KEY`, `SUPABASE_*`, and any optional guardrail env vars.
3. **Run Supabase SQL** from `docs/sql/` if your cloud tables are empty or new.
4. **Rehydrate models:** run `python data.py` then `python train.py` in `core-sentinel-guardrail`, or restore a saved `models/` backup.
5. **Telemetry:** if you used Supabase, historical rows remain in the project; local JSONL/SQLite are not recovered from Git.

## Cloud copies of logs (summary)

- **Webhook:** `GUARDRAIL_TELEMETRY_WEBHOOK_URL` — POST each telemetry JSON to your endpoint.
- **Supabase telemetry table:** `GUARDRAIL_TELEMETRY_SUPABASE=1` after applying `docs/sql/risk_telemetry.sql`.
- **Supabase scoring sync:** `sentinel_sync_daemon` → `guardrail_events` when `SUPABASE_URL` / `SUPABASE_ANON_KEY` are set.

See **`core-sentinel-guardrail/RISK_TELEMETRY.md`** for details.
