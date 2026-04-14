import os
import sys

# Add torch DLL directory before any imports (Windows fix)
_here = os.path.dirname(os.path.abspath(__file__))
for _rel in (
    (".venv", "Lib", "site-packages", "torch", "lib"),
    ("core-sentinel-guardrail", ".venv", "Lib", "site-packages", "torch", "lib"),
):
    _dll = os.path.join(_here, *_rel)
    if os.path.isdir(_dll) and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_dll)
        break

_guardrail = os.path.join(_here, "core-sentinel-guardrail")
_main_py = os.path.join(_guardrail, "main.py")
if not os.path.isfile(_main_py):
    print(f"Missing main.py: {_main_py}", file=sys.stderr)
    sys.exit(1)

# Working directory for relative paths inside the app; absolute path for runpy avoids broken __file__ on Windows.
os.chdir(_guardrail)
sys.path.insert(0, _guardrail)

import runpy

runpy.run_path(_main_py, run_name="__main__")
