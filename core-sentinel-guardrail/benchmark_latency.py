"""
End-to-end latency: tokenize → model forward → regex / PII overrides → policy decision.
Uses score_clipboard_with_pii() to match production. Each iteration uses a unique suffix so the
60s inference hash cache does not collapse timings.

Reports p50 / p95 / p99 in milliseconds (Wall clock). Optional warmup via GUARDRAIL_BENCH_WARMUP (default 10).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from infer import preload_guardrail_model, score_clipboard_with_pii

_DEFAULT_ITERS = 200
_BASE_TEXT = (
    "Wire transfer confirmation: beneficiary IBAN GB82WEST12345698765432, "
    "amount USD 45000, ref SAP-998877. Contact finance@acme.example.com for questions."
)


def _percentile_ms(samples: np.ndarray, q: float) -> float:
    return float(np.percentile(samples, q) * 1000.0)


def main() -> None:
    warmup = int(os.environ.get("GUARDRAIL_BENCH_WARMUP", "10"))
    iters = int(os.environ.get("GUARDRAIL_BENCH_ITERS", str(_DEFAULT_ITERS)))

    preload_guardrail_model()

    # Warmup: prime CUDA/cudnn and allocator
    for i in range(max(0, warmup)):
        score_clipboard_with_pii(f"{_BASE_TEXT} [{i}]")

    times = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        t0 = time.perf_counter()
        score_clipboard_with_pii(f"{_BASE_TEXT} [bench-{i}]")
        times[i] = time.perf_counter() - t0

    p50 = _percentile_ms(times, 50)
    p95 = _percentile_ms(times, 95)
    p99 = _percentile_ms(times, 99)
    mean_ms = float(np.mean(times) * 1000.0)

    print("\n" + "=" * 60)
    print("Latency benchmark (score_clipboard_with_pii)")
    print("=" * 60)
    print(f"  iterations: {iters}  warmup: {warmup}")
    print(f"  mean:   {mean_ms:.3f} ms")
    print(f"  p50:    {p50:.3f} ms")
    print(f"  p95:    {p95:.3f} ms")
    print(f"  p99:    {p99:.3f} ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
