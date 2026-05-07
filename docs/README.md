# Core Sentinel — documentation

All guides live next to the application under **`core-sentinel-guardrail/`**. Use the links below (valid on GitHub and locally).

## Writing reports

| Document | Description |
|----------|-------------|
| [Capstone / thesis Results chapter](CAPSTONE_RESULTS_CHAPTER.md) | **§6 Results** skeleton: model benchmark table, PR/ROC figures, system screenshots (ML dashboard, landing, hub, overlays). Figure checklist and paths. |
| [Report figures folder](report-figures/FIGURES_README.txt) | PNG placeholders + **`overlay-three-states-capture.html`** for overlay triptych |
| ML screenshots | [`ml/screenshots/`](ml/screenshots/README.txt) — regenerate with **`python scripts/capture_ml_dashboard_screenshots.py`** from repo root (Playwright; writes dashboard PNGs + `report-figures/ui-overlay-states.png`) |

## Getting started

| Document | Description |
|----------|-------------|
| [Repository layout](../REPO_LAYOUT.md) | Top-level folders, entry points, what is production vs legacy |
| [Production and release](PRODUCTION_AND_RELEASE.md) | Git workflow, secrets, quality gates, ML release order, launch checklist |
| [Windows setup](WINDOWS_SETUP.md) | PyTorch DLL issues, PowerShell paths with spaces, non-git copies |
| [Backup and recovery](BACKUP_AND_RECOVERY.md) | What is in Git vs local-only, Supabase/webhook logging, how to recover |
| [Guardrail README](../core-sentinel-guardrail/README.md) | Overview, label modes (3-class / 9-class), `data.py` / `train.py` |
| [Clipboard guardrail](../core-sentinel-guardrail/CLIPBOARD_GUARDRAIL.md) | How paste monitoring works, env vars, full training / calibration loop |

## Feedback, GitHub, and retraining

| Document | Description |
|----------|-------------|
| [GitHub feedback sync](../core-sentinel-guardrail/GITHUB_FEEDBACK_SYNC.md) | Which files to sync (`feedback_store.jsonl`, `feedback_fulltext.jsonl`, …), `git pull` / `push` workflow |
| [Feedback training loop](../core-sentinel-guardrail/FEEDBACK_TRAINING_LOOP.md) | `merge_feedback_to_training.py` → `data.py` → `train.py`, full-text vs preview |

## Data directories

Dataset placeholders and READMEs under `core-sentinel-guardrail/data/` (e.g. Patronus, ai4privacy Kaggle) — see each folder’s `README.md` after you add data locally.

## Root README

The [repository root README](../README.md) is the main landing page for GitHub. It documents **GitHub Pages** assets under `docs/` (hub, ML health **Summary + Precision–Recall** UI, privacy dashboard, admin), **`docs/data/*.json`** published metrics, **`ml/screenshots/`** capture tooling, and **Supabase** table/query expectations for live dashboards.
