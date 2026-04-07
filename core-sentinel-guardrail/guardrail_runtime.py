"""
Shared runtime state for Smart Silencing (main thread + RemediationDialog).

- Recent paste text hashes (last 3) for duplicate suppression (critical-secret pastes are not skipped; see infer.detect_critical_secret_leak).
- Duplicate bypass set: texts that were analyzed as risk \"high\" OR risk_score at/above GUARDRAIL_DUPLICATE_BYPASS_MIN_SCORE (default 80) are never skipped as duplicates or inference-cache hits for the session.
- Snooze-until timestamp (monotonic clock) for optional 10-minute bubble silence.
- Inference dedup: SHA-256 of stripped text -> monotonic time of last model run (60s TTL).
"""

from __future__ import annotations

import hashlib
import os
import time
from collections import deque
from typing import Deque

# Last 3 distinct clipboard strings (Python hash(), session-randomized salt per process — OK for dedup).
_recent_text_hashes: Deque[int] = deque(maxlen=3)

# Bubble snooze: no UI until this monotonic time.
_snooze_until_mono: float = 0.0

# Skip model if same normalized-text SHA-256 was scored within this many seconds.
_INFERENCE_HASH_TTL_SEC: float = 60.0
_inference_hash_ts: dict[str, float] = {}

# python hash(text): session text that was high risk or high risk_score (never skip duplicate / TTL)
_never_skip_duplicate_clipboard_hashes: set[int] = set()


def duplicate_bypass_min_risk_score() -> int:
    """Pastes at or above this score always re-run analysis (duplicates + 60s inference TTL bypass)."""
    raw = os.environ.get("GUARDRAIL_DUPLICATE_BYPASS_MIN_SCORE", "80").strip()
    try:
        v = int(raw)
    except ValueError:
        return 80
    return max(0, min(100, v))


def record_scored_clipboard_risk_score(text: str, risk_score: int, *, risk_level: str | None = None) -> None:
    """After an analysis, remember clipboard text that must always be re-analyzed on repeat."""
    t = (text or "").strip()
    if not t:
        return
    r = (risk_level or "").strip().lower()
    if r == "high":
        _never_skip_duplicate_clipboard_hashes.add(hash(t))
        return
    thr = duplicate_bypass_min_risk_score()
    if thr > 0 and risk_score >= thr:
        _never_skip_duplicate_clipboard_hashes.add(hash(t))


def is_high_scoring_clipboard_repeat(text: str) -> bool:
    """True if this string was previously high risk or hit the duplicate-bypass score threshold."""
    return hash(text) in _never_skip_duplicate_clipboard_hashes


def _normalized_text_sha256_hex(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def should_skip_inference_for_recent_hash(text: str, ttl_sec: float = _INFERENCE_HASH_TTL_SEC) -> bool:
    """True if this text's hash was recorded within ttl_sec — caller should not run the model."""
    t = (text or "").strip()
    if not t:
        return False
    if is_high_scoring_clipboard_repeat(t):
        return False
    h = _normalized_text_sha256_hex(t)
    now = time.monotonic()
    ts = _inference_hash_ts.get(h)
    if ts is None:
        return False
    if now - ts >= ttl_sec:
        del _inference_hash_ts[h]
        return False
    return True


def record_inference_scored_text(text: str) -> None:
    """Call after a forward pass through the classifier so duplicates within TTL skip inference."""
    t = (text or "").strip()
    if not t:
        return
    h = _normalized_text_sha256_hex(t)
    _inference_hash_ts[h] = time.monotonic()


def is_recent_duplicate(text: str) -> bool:
    """True if this exact string was among the last 3 analyzed pastes."""
    h = hash(text)
    return h in _recent_text_hashes


def record_recent_text(text: str) -> None:
    """Call after scoring a non-duplicate paste so repeats are suppressed."""
    _recent_text_hashes.append(hash(text))


def is_guard_snoozed() -> bool:
    """True if user enabled snooze and the window has not expired."""
    return time.monotonic() < _snooze_until_mono


def snooze_guard_10_minutes() -> None:
    """Silence RiskBubble UI for 10 minutes from now."""
    global _snooze_until_mono
    _snooze_until_mono = time.monotonic() + 600.0
