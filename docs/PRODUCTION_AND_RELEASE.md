# Production workflow, release discipline, and documentation

This document describes a **professional, repeatable** way to develop, validate, and operate Core Sentinel. It complements the technical guides under `core-sentinel-guardrail/` and **[BACKUP_AND_RECOVERY.md](BACKUP_AND_RECOVERY.md)**.

---

## 1. Source control (GitHub)

| Practice | Why |
|----------|-----|
| **Always work from a `git clone`** | Ensures `.git` exists, `git pull` works, and you get the latest `run_guardrail.py` launcher and fixes. ZIP/USB copies drift and break updates. |
| **Use a branch for work** (`feature/…`, `fix/…`) | Keeps `main` stable; use pull requests to review before merge. |
| **Commit messages** | Short imperative subject (e.g. `Fix DLL path for torch on Windows`); body optional for context. |
| **Never commit secrets** | `.env` is gitignored; use **`.env.example`** as the template only. |

**Clone (example):**

```powershell
git clone https://github.com/Alwinphilip0105/Core_Sentinal.git
cd Core_Sentinal
```

---

## 2. Secrets and configuration

| Item | Location | Rule |
|------|----------|------|
| API keys (Gemini, Supabase, webhooks) | Repo root **`.env`** | Create locally; never push. Copy from **`.env.example`**. |
| Per-user UI settings | `config/user_settings.json` | Gitignored; not for production baselines. |
| Calibrated policy | `config/risk_policy.json` | May be committed after `calibrate_thresholds.py`; contains no raw PII. |

**Production habit:** Rotate keys if exposed; restrict Supabase **RLS** policies so anon keys cannot read sensitive data beyond what you intend.

---

## 3. Reproducible Python environment

| Step | Command / note |
|------|----------------|
| Create venv | `python -m venv .venv` (repo root) or `uv venv` under `core-sentinel-guardrail` per your layout. |
| Install deps | `pip install -r core-sentinel-guardrail/requirements.txt` (full desktop stack). |
| Lockfile | `core-sentinel-guardrail/uv.lock` + `pyproject.toml` support **`uv sync`** for ML tooling; align with `requirements.txt` if you standardize on one tool. |
| Document path quirks | Paths with **spaces** need quotes in PowerShell; see **[WINDOWS_SETUP.md](WINDOWS_SETUP.md)**. |

---

## 4. Quality gates before you ship changes

Run from **`core-sentinel-guardrail`** with the project venv activated.

| Gate | Command | Purpose |
|------|---------|---------|
| Adversarial smoke | `python adversarial_tests.py` | Required regex/model integration checks; exit 0 to pass. |
| Test suite (extended) | `python test_sentinel.py` | Broader cases; review `logs/test_report.html` for failures. |
| Model eval (if ML changed) | `python evaluate_test.py` | Test-split metrics; refresh `reports/` and PR curve. |
| Thresholds (if eval changed) | `python calibrate_thresholds.py` | Writes `config/risk_policy.json` from PR data. |

**If you only changed UI or docs:** run at least **`adversarial_tests.py`** when inference paths touch `infer.py`.

---

## 5. ML release sequence (when the model or data pipeline changes)

1. **`python data.py`** (with the intended data source argument if not default).  
2. **`python train.py`**  
3. **`python evaluate_test.py`**  
4. **`python calibrate_thresholds.py`**  
5. Commit updated **`config/risk_policy.json`** and any report snippets you policy-allow; **do not** commit `models/` or `arrow_datasets/` (see `.gitignore`).  
6. Document in the PR what data source and metrics you used.

---

## 6. Cloud and telemetry (operations)

| Component | Documentation |
|-----------|----------------|
| SQLite → Supabase `guardrail_events` | Daemon in `sentinel_sync_daemon.py`; needs `SUPABASE_URL`, `SUPABASE_ANON_KEY`. |
| Optional telemetry table | **`docs/sql/risk_telemetry.sql`**, **`GUARDRAIL_TELEMETRY_SUPABASE=1`** — see **`core-sentinel-guardrail/RISK_TELEMETRY.md`**. |
| Webhook | **`GUARDRAIL_TELEMETRY_WEBHOOK_URL`** for external pipelines. |

**Production:** Apply SQL migrations in a named Supabase project; tighten **RLS** for production (avoid open `SELECT` for anon if data is sensitive).

---

## 7. Desktop app launch (validated path)

From **repository root** (where `run_guardrail.py` lives):

```powershell
$env:GUARDRAIL_BLOCKING_PRELOAD = "0"
& ".\.venv\Scripts\python.exe" ".\run_guardrail.py"
```

If the venv lives only under `core-sentinel-guardrail`, adjust the path to `python.exe` accordingly. **Do not** rely on ZIP extracts without replacing **`run_guardrail.py`** from `main` — see **`docs/WINDOWS_SETUP.md`**.

---

## 8. Release checklist (before demo / class / stakeholder handoff)

- [ ] Repository is a **Git clone** at the tagged commit or `main`.  
- [ ] **`.env`** present locally (not in Git); keys valid.  
- [ ] **Dependencies installed** in the venv used for the demo.  
- [ ] **`adversarial_tests.py`** passes.  
- [ ] **`run_guardrail.py`** starts the UI without import errors.  
- [ ] Supabase (if used): migrations applied; sync or telemetry verified in dashboard.  
- [ ] **Backup path** understood: see **BACKUP_AND_RECOVERY.md** for what is recoverable from Git vs cloud.

---

## 9. Documentation map

| Document | Role |
|----------|------|
| [README.md](../README.md) | Install, configure, run, training overview. |
| [REPO_LAYOUT.md](../REPO_LAYOUT.md) | Top-level folders and entry points. |
| [WINDOWS_SETUP.md](WINDOWS_SETUP.md) | PyTorch DLL issues, paths with spaces, non-git copies. |
| [BACKUP_AND_RECOVERY.md](BACKUP_AND_RECOVERY.md) | Git vs local data, recovery steps. |
| [core-sentinel-guardrail/RISK_TELEMETRY.md](../core-sentinel-guardrail/RISK_TELEMETRY.md) | Telemetry JSONL, webhook, Supabase. |
| [docs/sql/](sql/) | Supabase DDL to run in order. |

---

## 10. Incident recovery (short)

1. Re-clone or `git pull` to a clean directory.  
2. Restore **`.env`** from secure storage.  
3. Reinstall venv and `requirements.txt`.  
4. Re-run Supabase SQL if tables missing.  
5. Retrain or restore **`models/`** from your own backup if needed (not in Git).

For detail, see **BACKUP_AND_RECOVERY.md**.
