"""
Merge default inference thresholds with config/risk_policy.json.

All tunable numeric thresholds for 3-class inference, strict decision rules, and
critical-secret heuristics live in JSON; this module deep-merges defaults with
the file and exposes typed getters. Call clear_risk_policy_cache() in tests or
after editing the JSON at runtime.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

# Full defaults when risk_policy.json is missing or partial.
DEFAULT_RISK_POLICY: dict[str, Any] = {
    "min_prob_high_block": 0.80,
    "min_prob_high_warn": 0.50,
    "min_prob_med_warn": 0.60,
    # 3-class label mapping from softmax (same semantics as legacy infer.py)
    "inference_3class_labels": {
        "prob_threshold_high": 0.80,
        "prob_threshold_med": 0.40,
    },
    # Stricter branch inside decision_from_probs when risk == "high"
    "strict_decision_rules": {
        "prob_high_block": 0.35,
        "prob_high_warn": 0.20,
        "pii_high_block_prob": 0.30,
    },
    # 9-class: count non-O classes with prob >= this
    "pii_count_probability_threshold": 0.1,
    # High-entropy token heuristic for critical_secret_detected (bits per char)
    "critical_secret": {
        "min_token_entropy_bits": 4.2,
    },
    # Kept for 9-class models; merged from file
    "per_class_thresholds": {},
}

_policy_cache: dict[str, Any] | None = None
_policy_path_used: Path | None = None


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_merged_risk_policy(config_path: Path | None = None) -> dict[str, Any]:
    """
    Load config/risk_policy.json and merge with DEFAULT_RISK_POLICY.
    Does not use process-global cache; use cached_load_risk_policy for infer.py.
    """
    path = config_path or Path(__file__).resolve().parent / "config" / "risk_policy.json"
    merged = copy.deepcopy(DEFAULT_RISK_POLICY)
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                user = json.load(f)
            if isinstance(user, dict):
                merged = _deep_merge(merged, user)
        except (json.JSONDecodeError, OSError):
            pass
    return merged


def cached_load_risk_policy(config_path: Path | None = None) -> dict[str, Any]:
    """Single-process cache for merged policy (infer hot path)."""
    global _policy_cache, _policy_path_used
    path = (config_path or Path(__file__).resolve().parent / "config" / "risk_policy.json").resolve()
    if _policy_cache is not None and _policy_path_used == path:
        return _policy_cache
    _policy_cache = load_merged_risk_policy(path)
    _policy_path_used = path
    return _policy_cache


def clear_risk_policy_cache() -> None:
    """Invalidate cache (tests, calibration script reloading JSON)."""
    global _policy_cache, _policy_path_used
    _policy_cache = None
    _policy_path_used = None


def get_inference_3class_label_thresholds(policy: dict[str, Any]) -> tuple[float, float]:
    """Returns (prob_threshold_high, prob_threshold_med) for mapping softmax → high/med/low."""
    t = policy.get("inference_3class_labels") or {}
    th = float(t.get("prob_threshold_high", DEFAULT_RISK_POLICY["inference_3class_labels"]["prob_threshold_high"]))
    tm = float(t.get("prob_threshold_med", DEFAULT_RISK_POLICY["inference_3class_labels"]["prob_threshold_med"]))
    return th, tm


def get_strict_decision_rules(policy: dict[str, Any]) -> dict[str, float]:
    d = policy.get("strict_decision_rules") or {}
    base = DEFAULT_RISK_POLICY["strict_decision_rules"]
    return {
        "prob_high_block": float(d.get("prob_high_block", base["prob_high_block"])),
        "prob_high_warn": float(d.get("prob_high_warn", base["prob_high_warn"])),
        "pii_high_block_prob": float(d.get("pii_high_block_prob", base["pii_high_block_prob"])),
    }


def get_pii_count_probability_threshold(policy: dict[str, Any]) -> float:
    v = policy.get("pii_count_probability_threshold")
    if v is None:
        return float(DEFAULT_RISK_POLICY["pii_count_probability_threshold"])
    return float(v)


def get_critical_secret_entropy_min(policy: dict[str, Any]) -> float:
    c = policy.get("critical_secret") or {}
    return float(c.get("min_token_entropy_bits", DEFAULT_RISK_POLICY["critical_secret"]["min_token_entropy_bits"]))
