# Rutgers demo (legacy standalone UI)

Small **PyQt6** demo that monitors the clipboard for simple patterns (e.g. Rutgers ID, SAP-style IDs) and shows a nudge window. It is **separate** from the full guardrail in **`core-sentinel-guardrail/`** (TinyBERT, remediation panel, LLM detection).

## Run

From this folder, with the project venv activated and dependencies installed (`PyQt6`, `pywin32`, `pyperclip`):

```powershell
cd legacy/rutgers_demo
python main.py
```

Or from repo root:

```powershell
.\core-sentinel-guardrail\.venv\Scripts\python.exe legacy\rutgers_demo\main.py
```

## Production app

Use **`run_guardrail.py`** at the repository root for the full Risk-Aware Assistant — see **`README.md`** and **`REPO_LAYOUT.md`**.
