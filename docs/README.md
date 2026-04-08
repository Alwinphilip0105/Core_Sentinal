# Core Sentinel — documentation

All guides live next to the application under **`core-sentinel-guardrail/`**. Use the links below (valid on GitHub and locally).

## Getting started

| Document | Description |
|----------|-------------|
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

The [repository root README](../README.md) is the main landing page for GitHub.
