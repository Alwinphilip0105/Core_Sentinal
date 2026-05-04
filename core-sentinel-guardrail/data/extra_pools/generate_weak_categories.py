#!/usr/bin/env python3
"""
Thin entrypoint: same as tools/gen_weak_category_examples.py
Run from core-sentinel-guardrail/:
  python data/extra_pools/generate_weak_categories.py
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    target = _ROOT / "tools" / "gen_weak_category_examples.py"
    if not target.is_file():
        print(f"Missing implementation: {target}", file=sys.stderr)
        sys.exit(1)
    runpy.run_path(str(target), run_name="__main__")
