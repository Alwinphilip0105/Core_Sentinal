"""
Load/save user preferences to config/user_settings.json and apply env/runtime mirrors.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import guardrail_runtime

_GUARDRAIL_ROOT = Path(__file__).resolve().parent
USER_SETTINGS_PATH = _GUARDRAIL_ROOT / "config" / "user_settings.json"

_DEFAULTS: dict = {
    "monitor_llm_only": True,
    "show_badge_on_issues": True,
    "block_high_risk_pastes": True,
    "play_sound_on_block": False,
    "auto_redact_on_block": False,
    "sensitivity": "medium",
    "font_size": 11,
    "bubble_x": None,
    "bubble_y": None,
    "panel_x": None,
    "panel_y": None,
    # Optional PNG/SVG/JPG path for the pill “idle” slot (when no score/critical). Env overrides.
    "bubble_idle_image": None,
    # Daily streak for high-risk character messages (YYYY-MM-DD in last_date).
    "high_pastes_today": 0,
    "last_date": "",
    # Safe-paste streak (persisted; reset if idle >48h on load — see RiskBubble._load_streak).
    "streak_count": 0,
    "streak_best": 0,
    "streak_last_date": "",
    "total_safe": 0,
    "total_risky": 0,
}


def _merge_defaults(data: dict | None) -> dict:
    out = dict(_DEFAULTS)
    if isinstance(data, dict):
        for k, v in data.items():
            if k in _DEFAULTS:
                out[k] = v
    if out["sensitivity"] not in ("low", "medium", "high", "strict"):
        out["sensitivity"] = "medium"
    out["monitor_llm_only"] = bool(out["monitor_llm_only"])
    out["show_badge_on_issues"] = bool(out["show_badge_on_issues"])
    out["block_high_risk_pastes"] = bool(out["block_high_risk_pastes"])
    out["play_sound_on_block"] = bool(out["play_sound_on_block"])
    out["auto_redact_on_block"] = bool(out.get("auto_redact_on_block", False))
    try:
        fs = int(out.get("font_size", _DEFAULTS["font_size"]))
    except (TypeError, ValueError):
        fs = int(_DEFAULTS["font_size"])
    out["font_size"] = max(7, min(fs, 72))
    for k in ("bubble_x", "bubble_y", "panel_x", "panel_y"):
        v = out.get(k)
        if v is None:
            continue
        try:
            out[k] = int(v)
        except (TypeError, ValueError):
            out[k] = None
    img = out.get("bubble_idle_image")
    if isinstance(img, str) and img.strip():
        out["bubble_idle_image"] = str(img).strip()
    else:
        bundled = _GUARDRAIL_ROOT / "assets" / "coresentinel_idle.png"
        out["bubble_idle_image"] = (
            str(bundled.resolve()) if bundled.is_file() else None
        )
    for sk in (
        "streak_count",
        "streak_best",
        "total_safe",
        "total_risky",
    ):
        try:
            out[sk] = max(0, int(out.get(sk, 0) or 0))
        except (TypeError, ValueError):
            out[sk] = int(_DEFAULTS[sk])
    sld = out.get("streak_last_date")
    out["streak_last_date"] = str(sld).strip() if isinstance(sld, str) else ""
    return out


def load() -> dict:
    try:
        if USER_SETTINGS_PATH.is_file():
            with open(USER_SETTINGS_PATH, encoding="utf-8") as f:
                raw = json.load(f)
            return _merge_defaults(raw if isinstance(raw, dict) else {})
    except Exception:
        pass
    return _merge_defaults({})


def record_high_risk_paste() -> int:
    """
    Increment today's high-risk paste count (resets when `last_date` != today).
    Persists to user_settings.json. Returns the new count (>= 1).
    """
    from datetime import date

    today = date.today().isoformat()
    d = load()
    last = str(d.get("last_date") or "")
    if last != today:
        d["high_pastes_today"] = 0
        d["last_date"] = today
    n = int(d.get("high_pastes_today") or 0) + 1
    d["high_pastes_today"] = n
    save(d)
    return n


def save_bubble_position(x: int, y: int) -> None:
    """Persist floating pill position (merged into user_settings.json)."""
    d = load()
    d["bubble_x"] = int(x)
    d["bubble_y"] = int(y)
    save(d)


def save(data: dict) -> None:
    combined = {**load(), **(data or {})}
    merged = _merge_defaults(combined)
    USER_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(USER_SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    apply_to_environment_and_runtime(merged)


def apply_to_environment_and_runtime(data: dict | None = None) -> None:
    """Set os.environ knobs used by infer + guardrail_runtime mirrors."""
    d = _merge_defaults(data) if data is not None else load()
    guardrail_runtime.set_monitor_llm_only(d.get("monitor_llm_only", True))

    # GUARDRAIL_BLOCK_RISK_SCORE_MIN: 0 = disabled (see infer._risk_score_force_block_min)
    if not d.get("block_high_risk_pastes", True):
        os.environ["GUARDRAIL_BLOCK_RISK_SCORE_MIN"] = "0"
    else:
        sens = str(d.get("sensitivity", "medium"))
        score_map = {"low": 98, "medium": 95, "high": 90, "strict": 85}
        os.environ["GUARDRAIL_BLOCK_RISK_SCORE_MIN"] = str(score_map.get(sens, 95))
