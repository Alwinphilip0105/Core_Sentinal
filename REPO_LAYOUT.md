# Repository layout

| Path | Purpose |
|------|---------|
| **`run_guardrail.py`** (root) | Thin wrapper — runs **`launchers/run_guardrail.py`**. Same command as before: `python run_guardrail.py` |
| **`launchers/`** | Real launcher scripts (PyTorch DLL paths, `importlib` load of `core-sentinel-guardrail/main.py`) |
| **`core-sentinel-guardrail/`** | Full **Risk-Aware Assistant**: inference (`infer.py`), UI (`main.py`), training, config, optional `tools/` (e.g. telemetry dashboard server) |
| **`scripts/`** | Dev / evaluation helpers (`manual_probe.py`, dataset checks) — run from repo root |
| **`docs/`** | Guides: production workflow, backup, Windows setup, static HTML hub, **`docs/sql/`** for Supabase |
| **`legacy/rutgers_demo/`** | Older small clipboard demo (PyQt nudge) — **not** the production guardrail |

## Entry points (quick reference)

| Goal | Command |
|------|---------|
| Full guardrail UI | `python run_guardrail.py` from repo root (venv active) |
| Alternate clipboard app | `python run_clipboard_guardrail.py` |
| ML / training | `cd core-sentinel-guardrail` then `python data.py`, `train.py`, … |
| Probe scoring | `python scripts/manual_probe.py` from repo root |

Secrets: **`.env`** at repo root (copy from **`.env.example`**). See **`README.md`**.
