#!/usr/bin/env python3
"""Run the clipboard guardrail app from project root. Forwards all args."""
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve().parent
    return here.parent if here.name == "launchers" else here


guardrail_dir = _repo_root() / "core-sentinel-guardrail"
script = guardrail_dir / "windows_clipboard_app.py"
if not script.exists():
    print(f"Not found: {script}", file=sys.stderr)
    sys.exit(1)
sys.exit(subprocess.call([sys.executable, str(script)] + sys.argv[1:]))
