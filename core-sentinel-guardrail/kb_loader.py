"""
Load company-specific knowledge bases (YAML) from local disk or GitHub,
merge into guardrail policy thresholds, and expose compiled regex patterns.
"""

from __future__ import annotations

import copy
import json
import os
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parent
_CONFIG_DIR = _ROOT / "config"
_KB_CACHE_PATH = _CONFIG_DIR / "kb_cache.yaml"

# Map KB sensitivity override actions to per-class probability thresholds (warn / block).
_SENSITIVITY_ACTION_TO_THRESHOLDS: dict[str, dict[str, float]] = {
    "block": {"warn": 0.35, "block": 0.55},
    "warn": {"warn": 0.65, "block": 0.92},
}


def _http_get_text(url: str, token: str | None, timeout: float = 5.0) -> str | None:
    try:
        headers: dict[str, str] = {}
        if token:
            headers["Authorization"] = f"token {token}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8")
    except (urllib.error.URLError, OSError, TimeoutError, ValueError) as e:
        print(f"[kb_loader] warning: HTTP GET failed for {url!r}: {e}")
        return None


def _normalize_github_repo(repo_url: str) -> str | None:
    u = repo_url.strip().rstrip("/")
    if "raw.githubusercontent.com" in u:
        return None
    if "github.com" in u:
        tail = u.split("github.com/", 1)[-1]
        tail = tail.replace(".git", "")
        parts = tail.split("/")
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
    if "/" in u and not u.startswith("http"):
        return u
    return None


def fetch_kb_from_github(repo_url: str, token: str, file_path: str) -> str | None:
    """
    Fetch KB YAML text from GitHub.
    If repo_url is a raw.githubusercontent.com URL, GET it directly.
    Otherwise build: https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}
    """
    u = repo_url.strip().rstrip("/")
    if "raw.githubusercontent.com" in u and u.startswith("http"):
        out = _http_get_text(u, token or None)
        if out is None:
            print("[kb_loader] warning: fetch_kb_from_github raw URL returned no content")
        return out

    owner_repo = _normalize_github_repo(repo_url)
    if not owner_repo:
        print(f"[kb_loader] warning: could not parse GitHub repo URL from {repo_url!r}")
        return None

    branch = os.environ.get("GUARDRAIL_KB_BRANCH", "main")
    path = file_path.lstrip("/")
    raw_url = f"https://raw.githubusercontent.com/{owner_repo}/{branch}/{path}"
    out = _http_get_text(raw_url, token or None)
    if out is None:
        print("[kb_loader] warning: fetch_kb_from_github built URL fetch failed")
    return out


def fetch_kb_from_local(path: str | Path) -> str | None:
    p = Path(path)
    if not p.is_file():
        print(f"[kb_loader] warning: local KB file not found: {p}")
        return None
    try:
        return p.read_text(encoding="utf-8")
    except OSError as e:
        print(f"[kb_loader] warning: could not read local KB {p}: {e}")
        return None


def _write_kb_cache(data: dict[str, Any]) -> None:
    try:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(_KB_CACHE_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
        print(f"[kb_loader] wrote cache {_KB_CACHE_PATH}")
    except OSError as e:
        print(f"[kb_loader] warning: could not write KB cache: {e}")


def _load_kb_cache_file() -> dict[str, Any] | None:
    if not _KB_CACHE_PATH.is_file():
        return None
    try:
        with open(_KB_CACHE_PATH, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if isinstance(raw, dict):
            return raw
    except (yaml.YAMLError, OSError) as e:
        print(f"[kb_loader] warning: could not read KB cache: {e}")
    return None


def load_kb(source: str = "local", **kwargs: Any) -> dict[str, Any]:
    """
    Load and parse KB YAML. Caches successful loads to config/kb_cache.yaml with timestamp.
    On fetch failure, falls back to cached file.
    """
    text: str | None = None
    if source == "github":
        print("[kb_loader] loading KB from GitHub")
        text = fetch_kb_from_github(
            kwargs.get("repo_url", ""),
            kwargs.get("token", ""),
            kwargs.get("file_path", "kb.yaml"),
        )
    elif source == "local":
        p = kwargs.get("path")
        print(f"[kb_loader] loading KB from local path: {p!r}")
        if p:
            text = fetch_kb_from_local(p)
        else:
            print("[kb_loader] warning: local source requires path=...")
    else:
        print(f"[kb_loader] warning: unknown source {source!r}")
        text = None

    parsed: dict[str, Any] | None = None
    if text:
        try:
            loaded = yaml.safe_load(text)
            parsed = loaded if isinstance(loaded, dict) else None
            if parsed is None:
                print("[kb_loader] warning: YAML did not parse to a dict")
        except yaml.YAMLError as e:
            print(f"[kb_loader] warning: YAML parse error: {e}")

    if parsed:
        ts = datetime.now(timezone.utc).isoformat()
        to_cache = copy.deepcopy(parsed)
        to_cache["timestamp"] = ts
        print(f"[kb_loader] KB parsed OK (company={parsed.get('company', '?')!r})")
        _write_kb_cache(to_cache)
        return parsed

    print("[kb_loader] fetch/parse failed; trying config/kb_cache.yaml")
    cached = _load_kb_cache_file()
    if cached:
        # Return KB without cache-only metadata if present
        out = {k: v for k, v in cached.items() if k != "timestamp"}
        print("[kb_loader] using cached KB from disk")
        return out

    print("[kb_loader] total failure: returning empty KB dict")
    return {}


def get_kb_compiled_patterns(kb: dict) -> list[tuple[str, re.Pattern[str], str]]:
    """Return (name, compiled_regex, risk) for custom_pii_patterns and forbidden_terms."""
    out: list[tuple[str, re.Pattern[str], str]] = []

    for entry in kb.get("custom_pii_patterns") or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "unnamed"))
        pattern = entry.get("regex") or entry.get("pattern")
        risk = str(entry.get("risk", "med"))
        if not pattern:
            print(f"[kb_loader] warning: skipping pattern {name!r} (no regex)")
            continue
        try:
            cre = re.compile(pattern)
            out.append((name, cre, risk))
        except re.error as e:
            print(f"[kb_loader] warning: regex compile failed for {name!r}: {e}")

    for term in kb.get("forbidden_terms") or []:
        if not isinstance(term, str) or not term.strip():
            continue
        t = term.strip()
        name = f"forbidden:{t[:64]}"
        try:
            cre = re.compile(re.escape(t), re.IGNORECASE)
            out.append((name, cre, "high"))
        except re.error as e:
            print(f"[kb_loader] warning: forbidden term compile failed for {t!r}: {e}")

    return out


def merge_kb_into_policy(kb: dict, policy: dict) -> dict:
    """
    Apply kb['sensitivity_overrides'] to policy['per_class_thresholds'].
    Returns a new dict; does not mutate the input policy.
    """
    merged = copy.deepcopy(policy)
    overrides = kb.get("sensitivity_overrides") or {}
    if not overrides:
        return merged

    pct = merged.get("per_class_thresholds")
    if not isinstance(pct, dict):
        pct = {}
        merged["per_class_thresholds"] = pct

    for class_name, action in overrides.items():
        key = str(class_name).upper()
        act = str(action).lower().strip()
        thresh = _SENSITIVITY_ACTION_TO_THRESHOLDS.get(act)
        if thresh is None:
            print(f"[kb_loader] warning: unknown sensitivity action {action!r} for {key}")
            continue
        pct[key] = copy.deepcopy(thresh)
        print(f"[kb_loader] sensitivity override {key} -> {act} ({thresh})")

    return merged


def get_active_kb() -> dict:
    """
    Resolve KB from env:
      GUARDRAIL_KB_PATH — local file
      GUARDRAIL_KB_URL + GUARDRAIL_KB_TOKEN — GitHub (repo or raw URL; optional GUARDRAIL_KB_FILE)
    """
    local = os.environ.get("GUARDRAIL_KB_PATH")
    if local:
        return load_kb("local", path=local)

    url = os.environ.get("GUARDRAIL_KB_URL")
    if url:
        token = os.environ.get("GUARDRAIL_KB_TOKEN", "")
        file_path = os.environ.get("GUARDRAIL_KB_FILE", "kb.yaml")
        return load_kb("github", repo_url=url, token=token, file_path=file_path)

    return {}


if __name__ == "__main__":
    _sample = _ROOT / "config" / "sample_kb.yaml"
    if _sample.is_file():
        os.environ.setdefault("GUARDRAIL_KB_PATH", str(_sample))
    kb = get_active_kb()
    print("Loaded KB:", kb.get("company", "no company set"))
    patterns = get_kb_compiled_patterns(kb)
    print(f"Compiled {len(patterns)} patterns")
