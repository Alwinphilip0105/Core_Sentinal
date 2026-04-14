import importlib.util
import os
import sys


def _repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here) if os.path.basename(here) == "launchers" else here


# Add torch DLL directory before any imports (Windows fix)
_here = _repo_root()
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

os.chdir(_guardrail)
sys.path.insert(0, _guardrail)

_spec = importlib.util.spec_from_file_location("__main__", _main_py)
if _spec is None or _spec.loader is None:
    print(f"Could not load: {_main_py}", file=sys.stderr)
    sys.exit(1)
_main_mod = importlib.util.module_from_spec(_spec)
sys.modules["__main__"] = _main_mod
_spec.loader.exec_module(_main_mod)
