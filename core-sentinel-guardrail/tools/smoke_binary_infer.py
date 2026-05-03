import contextlib
import io
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from infer import score_clipboard_with_pii

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    cases = [
        ("Hello, how are you today?", "expect: allow"),
        ("My email is john@example.com", "expect: warn or block"),
        ("SSN: 123-45-6789, Password: abc123!", "expect: block"),
    ]
    for text, note in cases:
        r = score_clipboard_with_pii(text)
        print(f"\n{note}")
        print(f"  decision   : {r.get('decision')}")
        pr = r.get("prob_risky")
        print(f"  prob_risky : {pr:.3f}" if pr is not None else "  prob_risky : None")
        print(f"  t_warn     : {r.get('t_warn')}")
        print(f"  t_block    : {r.get('t_block')}")
        print(f"  layer_summary: {r.get('layer_summary')}")
print(buf.getvalue())
