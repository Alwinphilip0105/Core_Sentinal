#!/usr/bin/env python3
"""Run the clipboard guardrail app from project root. Forwards all args."""
import subprocess
import sys
from pathlib import Path

guardrail_dir = Path(__file__).resolve().parent / "core-sentinel-guardrail"
script = guardrail_dir / "windows_clipboard_app.py"
if not script.exists():
    print(f"Not found: {script}", file=sys.stderr)
    sys.exit(1)
sys.exit(subprocess.call([sys.executable, str(script)] + sys.argv[1:]))
