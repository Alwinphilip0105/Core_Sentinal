import os
import sys

# Add torch DLL directory before any imports (Windows fix)
_here = os.path.dirname(os.path.abspath(__file__))
_dll = os.path.join(_here, ".venv", "Lib", "site-packages", "torch", "lib")
if os.path.isdir(_dll) and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(_dll)

# Set working directory and path to the guardrail package
os.chdir(os.path.join(_here, "core-sentinel-guardrail"))
sys.path.insert(0, os.path.join(_here, "core-sentinel-guardrail"))

# Launch main.py as __main__
import runpy
runpy.run_path("main.py", run_name="__main__")
