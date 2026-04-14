#!/usr/bin/env python3
"""Entry point at repo root; implementation lives in launchers/ (see REPO_LAYOUT.md)."""
import os
import subprocess
import sys

_root = os.path.dirname(os.path.abspath(__file__))
launcher = os.path.join(_root, "launchers", "run_guardrail.py")
sys.exit(subprocess.call([sys.executable, launcher] + sys.argv[1:]))
