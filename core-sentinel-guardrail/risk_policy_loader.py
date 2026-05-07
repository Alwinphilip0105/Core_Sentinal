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
    # If risk_score >= this (1..100), force block despite allow_warn_instead (infer.py). 0 = disabled.
    "block_threshold": 95,
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
    # 1–100 risk_score ceiling when paste is contact-only (email/phone/IP) and would otherwise score higher.
    "contact_only_risk_cap": 70,
    # P1-D: downgrades block→warn when ML-only NAME/LOCATION windows have no Regex/NER overlap
    "gate_geo_name_ml_without_ner_regex": True,
    # Kept for 9-class models; merged from file
    "per_class_thresholds": {},
}

_policy_cache: dict[str, Any] | None = None
_policy_path_used: Path | None = None

_TIER_KEYS = frozenset({"low", "med", "high"})


def _tier_warn_block(th: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    """Warn/block pairs from thresholds.low / .med / .high (defaults if missing)."""
    low = th.get("low") if isinstance(th.get("low"), dict) else {}
    med = th.get("med") if isinstance(th.get("med"), dict) else {}
    hi = th.get("high") if isinstance(th.get("high"), dict) else {}
    wl = float(low.get("warn", 1.0))
    bl = float(low.get("block", 1.0))
    wm = float(med.get("warn", DEFAULT_RISK_POLICY["min_prob_med_warn"]))
    bm = float(med.get("block", 1.0))
    wh = float(hi.get("warn", DEFAULT_RISK_POLICY["min_prob_high_warn"]))
    bh = float(hi.get("block", DEFAULT_RISK_POLICY["min_prob_high_block"]))
    return wl, bl, wm, bm, wh, bh


def _apply_tier_thresholds_to_min_prob(merged: dict[str, Any]) -> None:
    """Map thresholds.low/med/high → min_prob_high_block, min_prob_high_warn, min_prob_med_warn."""
    th = merged.get("thresholds")
    if not isinstance(th, dict):
        return
    if not all(k in th for k in ("low", "med", "high")):
        return
    _, _, wm, _, wh, bh = _tier_warn_block(th)
    merged["min_prob_high_block"] = bh
    merged["min_prob_high_warn"] = wh
    merged["min_prob_med_warn"] = wm


def _expand_per_class_thresholds(merged: dict[str, Any]) -> None:
    """
    Build per_class_thresholds from class_overrides (max_action × tier warn/block) and
    overlay calibrated entries under thresholds.<CLASS> (not low/med/high).
    """
    th = merged.get("thresholds")
    if not isinstance(th, dict):
        th = {}
    wl, bl, wm, bm, wh, bh = _tier_warn_block(th)

    pct: dict[str, dict[str, float]] = {}
    co = merged.get("class_overrides")
    if isinstance(co, dict) and co:
        for cls, spec in co.items():
            key = str(cls).strip().upper()
            if not isinstance(spec, dict):
                continue
            ma = str(spec.get("max_action", "warn")).lower()
            if ma == "silent":
                pct[key] = {"warn": wl, "block": bl}
            elif ma == "warn":
                pct[key] = {"warn": wm, "block": bm}
            elif ma == "block":
                pct[key] = {"warn": wh, "block": bh}
            else:
                pct[key] = {"warn": wm, "block": bm}

    for k, v in th.items():
        if k in _TIER_KEYS or not isinstance(v, dict):
            continue
        if "warn" not in v and "block" not in v:
            continue
        ck = str(k).strip().upper()
        pct[ck] = {
            "warn": float(v.get("warn", 1.0)),
            "block": float(v.get("block", 1.0)),
        }

    if pct:
        merged["per_class_thresholds"] = pct


def _normalize_merged_policy(merged: dict[str, Any]) -> None:
    _apply_tier_thresholds_to_min_prob(merged)
    _expand_per_class_thresholds(merged)


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
    _normalize_merged_policy(merged)
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


def get_binary_two_thresholds(policy: dict[str, Any]) -> tuple[float, float]:
    """
    Returns (t_warn, t_block) from policy.
    Zones: prob < t_warn -> safe; t_warn <= prob < t_block -> warn; prob >= t_block -> block.
    Falls back from inference_binary_labels.prob_threshold_risky if two-threshold section absent.
    """
    two = policy.get("inference_binary_two_threshold") or {}
    t_warn = two.get("prob_threshold_warn")
    t_block = two.get("prob_threshold_block")
    if t_warn is not None and t_block is not None:
        return float(t_warn), float(t_block)
    single = float(
        (policy.get("inference_binary_labels") or {}).get("prob_threshold_risky", 0.5)
    )
    t_block = single
    t_warn = max(0.05, float(single) - 0.20)
    print(
        f"[policy] two-threshold absent — derived: t_warn={t_warn:.3f}, t_block={t_block:.3f}"
    )
    return t_warn, t_block


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


def get_contact_only_risk_cap(policy: dict[str, Any]) -> int:
    """
    Max risk_score (1–100) for contact-only spans after capping elevated model scores.
    Used in infer.score_clipboard_with_pii — must align with clipboard_ui_action / block thresholds.
    """
    v = policy.get("contact_only_risk_cap")
    if v is None:
        return int(DEFAULT_RISK_POLICY["contact_only_risk_cap"])
    try:
        n = int(float(v))
    except (TypeError, ValueError):
        return int(DEFAULT_RISK_POLICY["contact_only_risk_cap"])
    return max(1, min(100, n))
