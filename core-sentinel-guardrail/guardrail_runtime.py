"""
Shared runtime state for Smart Silencing (main thread + RemediationDialog).

- Recent paste text hashes (last 3) for duplicate suppression (critical-secret pastes are not skipped; see infer.detect_critical_secret_leak).
- Duplicate bypass set: texts that were analyzed as risk \"high\" OR risk_score at/above GUARDRAIL_DUPLICATE_BYPASS_MIN_SCORE (default 80) are never skipped as duplicates or inference-cache hits for the session.
- Snooze-until timestamp (monotonic clock) for optional 10-minute bubble silence.
- Inference cache: SHA-256 of stripped text + kind ("pii"|"plain") -> last result dict and monotonic time (60s TTL).
"""

from __future__ import annotations

import copy
import hashlib
import os
import threading
import time
from collections import deque
from typing import Deque

# Last 3 distinct clipboard strings (Python hash(), session-randomized salt per process — OK for dedup).
_recent_text_hashes: Deque[int] = deque(maxlen=3)

# Bubble snooze: no UI until this monotonic time.
_snooze_until_mono: float = 0.0

# When True, only Ctrl+V / clipboard poll while an LLM window is focused (see user_settings).
_monitor_llm_only: bool = True

# When True, listener and clipboard poll do not enqueue analysis.
_monitoring_paused: bool = False

# Skip model if same normalized-text SHA-256 was scored within this many seconds (per cache kind).
_INFERENCE_CACHE_TTL_SEC: float = 60.0
_inference_cache_ts: dict[str, float] = {}
_score_cache: dict[str, dict] = {}
_inference_cache_lock = threading.Lock()

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


def _inference_cache_key(text_stripped: str, cache_kind: str) -> str:
    h = _normalized_text_sha256_hex(text_stripped)
    k = (cache_kind or "pii").strip().lower()
    if k not in ("pii", "plain"):
        k = "pii"
    return f"{h}|{k}"


def get_cached_inference_result(
    text: str,
    *,
    cache_kind: str = "pii",
    ttl_sec: float = _INFERENCE_CACHE_TTL_SEC,
) -> dict | None:
    """
    If this text was scored within ttl_sec for the same cache_kind, return a deep copy of the
    last result dict so the model can be skipped. Returns None if cache miss or bypass applies.
    """
    t = (text or "").strip()
    if not t:
        return None
    if is_high_scoring_clipboard_repeat(t):
        return None
    key = _inference_cache_key(t, cache_kind)
    with _inference_cache_lock:
        ts = _inference_cache_ts.get(key)
        if ts is None:
            return None
        if time.monotonic() - ts >= ttl_sec:
            del _inference_cache_ts[key]
            _score_cache.pop(key, None)
            return None
        cached = _score_cache.get(key)
        return copy.deepcopy(cached) if cached is not None else None


def record_inference_scored_text(
    text: str,
    result: dict | None = None,
    *,
    cache_kind: str = "pii",
) -> None:
    """Call after scoring so the same text can reuse the stored result within the TTL."""
    t = (text or "").strip()
    if not t:
        return
    key = _inference_cache_key(t, cache_kind)
    with _inference_cache_lock:
        _inference_cache_ts[key] = time.monotonic()
        if result is not None:
            _score_cache[key] = copy.deepcopy(result)


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


def snooze_guard_minutes(minutes: float) -> None:
    """Silence RiskBubble UI for the given duration from now."""
    global _snooze_until_mono
    _snooze_until_mono = time.monotonic() + max(0.0, float(minutes)) * 60.0


def snooze_guard_10_minutes() -> None:
    """Silence RiskBubble UI for 10 minutes from now."""
    snooze_guard_minutes(10.0)


def clear_guard_snooze() -> None:
    """End snooze immediately so the guardrail UI responds again."""
    global _snooze_until_mono
    _snooze_until_mono = 0.0


def get_monitor_llm_only() -> bool:
    return _monitor_llm_only


def set_monitor_llm_only(v: bool) -> None:
    global _monitor_llm_only
    _monitor_llm_only = bool(v)


def is_monitoring_paused() -> bool:
    return _monitoring_paused


def set_monitoring_paused(v: bool) -> None:
    global _monitoring_paused
    _monitoring_paused = bool(v)
