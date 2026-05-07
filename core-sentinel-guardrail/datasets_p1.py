"""
P1 supplemental training sources (opt-in via environment variables).

Integrations are gated: place corpora locally or configure JSONL / Hugging Face —
see docs/P1_ROADMAP.md.

- HEALTH: i2b2 2014 de-identification XML (distribution via i2b2.org; user accepts DUA).
- AUTH: rotated/public secret-format lines (JSONL) or optional HF dataset id.
- FINANCIAL: balanced risky/safe JSONL (generator: tools/gen_financial_balanced_p1.py).

All loaders return rows: {"text", "risk", "sap_id", "source"} compatible with multi_real_synthetic.
"""

from __future__ import annotations

import html
import json
import os
import re
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent
_DEFAULT_FIN_BALANCED = _ROOT / "data" / "extra_pools" / "financial_balanced_p1.jsonl"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


def _norm_ws(s: str) -> str:
    return " ".join(str(s or "").split()).strip()


def row_from_text(text: str, risk: str, sap_id: str, source: str) -> dict[str, Any]:
    """Shared row schema for dataset merging."""
    return {
        "text": text.strip(),
        "risk": risk if risk in ("low", "med", "high") else "low",
        "is_sap": 0,
        "sap_id": sap_id,
        "source": source,
    }


def load_financial_balanced_jsonl(max_rows: int | None = None) -> list[dict[str, Any]]:
    """
    GUARDRAIL_FINANCIAL_BALANCED_JSONL — JSONL {"text":"...","risk":"low|med|high"}

    Typical content: risky account/wire snippets + safe earnings/budget prose.

    If the env var is unset, loads ``data/extra_pools/financial_balanced_p1.jsonl`` when
    that file exists (built-in synthetic pool). Set ``GUARDRAIL_FINANCIAL_BALANCED_USE_DEFAULT_POOL``
    to 0/false/no to skip only that default path.
    """
    path_raw = os.environ.get("GUARDRAIL_FINANCIAL_BALANCED_JSONL", "").strip()
    if path_raw:
        path = Path(path_raw)
        if not path.is_file():
            print(f"[p1-financial_balanced] file not found: {path}")
            return []
    else:
        use_def = (
            str(os.environ.get("GUARDRAIL_FINANCIAL_BALANCED_USE_DEFAULT_POOL", "1")).strip().lower()
            not in ("0", "false", "no")
        )
        path = _DEFAULT_FIN_BALANCED
        if not use_def or not path.is_file():
            return []
    cap = max_rows if max_rows is not None else _env_int("GUARDRAIL_FIN_BALANCED_MAX_ROWS", 12000)
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8-sig") as f:
        for i, line in enumerate(f):
            if len(out) >= cap:
                print(f"[p1-financial_balanced] capped at {cap}")
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = str(obj.get("text") or obj.get("content") or "").strip()
            rk = str(obj.get("risk") or obj.get("label") or "").strip().lower()
            if rk == "medium":
                rk = "med"
            if not t or rk not in ("low", "med", "high"):
                continue
            if len(t) < 12 or len(t) > 32000:
                continue
            out.append(row_from_text(t, rk, f"finbal-{len(out)}", "financial_balanced_p1"))
    if out:
        src = (
            ""
            if path_raw
            else " (tracked default)"
        )
        print(f"[p1-financial_balanced] merged {len(out)} rows from {path}{src}")
    return out


def load_i2b2_2014_xml_dir(max_rows: int | None = None) -> list[dict[str, Any]]:
    """
    GUARDRAIL_I2B2_2014_DIR — directory containing i2b2 2014 de-id training XML notes.

    Each document with any <PHI> (or PHI tag) yields risk ``high`` (clinical PHI-heavy).
    Extraction unwraps PHI tags while keeping inner text — suitable for PHI-aware training tone.
    """
    raw = os.environ.get("GUARDRAIL_I2B2_2014_DIR", "").strip()
    if not raw:
        return []
    base = Path(raw)
    if not base.is_dir():
        print(f"[p1-i2b2] directory not found: {base}")
        return []
    cap = max_rows if max_rows is not None else _env_int("GUARDRAIL_I2B2_MAX_ROWS", 9000)

    paths = sorted(base.rglob("*.xml"))
    seen_norm: set[str] = set()
    out: list[dict[str, Any]] = []

    phi_re = re.compile(
        r"<\s*PHI\b[^>]*>([\s\S]*?)<\s*/\s*PHI\s*>",
        re.IGNORECASE,
    )

    for xp in paths:
        if len(out) >= cap:
            print(f"[p1-i2b2] capped at {cap}")
            break
        try:
            body = xp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        has_phi = bool(phi_re.search(body)) or "<phi" in body.lower()
        tmp = phi_re.sub(r"\1", body)
        tmp = re.sub(r"<[^>]+>", " ", tmp)
        plaintext = html.unescape(_norm_ws(tmp))
        if not plaintext:
            continue
        key = plaintext[:4096].lower()
        if key in seen_norm:
            continue
        seen_norm.add(key)

        risk = "high" if has_phi else "med"

        if len(plaintext) < 20:
            continue
        if len(plaintext) > MAX_TEXT_HARD():
            plaintext = plaintext[: MAX_TEXT_HARD()]

        out.append(row_from_text(plaintext, risk, f"i2b2-{len(out)}", "i2b2_2014_deid"))

    if out:
        print(f"[p1-i2b2] merged {len(out)} documents from {base}")
    return out


def MAX_TEXT_HARD() -> int:
    return _env_int("GUARDRAIL_P1_MAX_CHARS", 8000)


def load_auth_secret_patterns_jsonl(max_rows: int | None = None) -> list[dict[str, Any]]:
    """
    GUARDRAIL_AUTH_SECRETS_JSONL — lines like:
      {"text":"export AWS_SECRET_ACCESS_KEY=wJal...", "risk":"high","is_positive":true}
      {"text":"# safe config","risk":"low","is_positive":false}

    Rotated / synthetic placeholders only recommended for redistribution.
    """
    path_raw = os.environ.get("GUARDRAIL_AUTH_SECRETS_JSONL", "").strip()
    if not path_raw:
        return []
    path = Path(path_raw)
    if not path.is_file():
        print(f"[p1-auth-jsonl] not found: {path}")
        return []
    cap = max_rows if max_rows is not None else _env_int("GUARDRAIL_AUTH_SECRET_MAX_ROWS", 25000)
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            if len(out) >= cap:
                break
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = str(o.get("text") or o.get("snippet") or o.get("code") or "").strip()
            pos = o.get("is_positive", o.get("positive", o.get("is_secret")))
            if pos is not None:
                try:
                    is_pos = bool(pos)
                except (TypeError, ValueError):
                    is_pos = False
                risk = "high" if is_pos else "low"
            else:
                r = str(o.get("risk") or "").lower()
                risk = r if r in ("low", "med", "high") else ""
            if not text or risk not in ("low", "med", "high"):
                continue
            tl = MAX_TEXT_HARD()
            if len(text) > tl:
                text = text[:tl]
            out.append(row_from_text(text, risk, f"auths-{len(out)}", "auth_secrets_jsonl"))
    if out:
        print(f"[p1-auth-jsonl] merged {len(out)} rows from {path}")
    return out


def load_auth_hf_optional(max_rows: int | None = None) -> list[dict[str, Any]]:
    """
    Optional HF dataset via GUARDRAIL_AUTH_HF_DATASET=id (split train by default).

    Column resolution is best-effort: text/snippet/content + label/leak/binary.
    """
    ds_spec = os.environ.get("GUARDRAIL_AUTH_HF_DATASET", "").strip()
    if not ds_spec:
        return []
    split = os.environ.get("GUARDRAIL_AUTH_HF_SPLIT", "train").strip() or "train"
    try:
        from datasets import load_dataset
    except ImportError:
        print("[p1-auth-hf] pip install datasets to use GUARDRAIL_AUTH_HF_DATASET")
        return []
    cap = max_rows if max_rows is not None else _env_int("GUARDRAIL_AUTH_HF_MAX_ROWS", 15000)

    ds_id = ds_spec
    subset = ""
    if ":" in ds_spec:
        ds_id, subset = ds_spec.split(":", 1)
    try:
        if subset:
            ds = load_dataset(ds_id, subset, split=split)
        else:
            ds = load_dataset(ds_id, split=split)
    except Exception as ex:
        print(f"[p1-auth-hf] load_dataset failed ({ds_spec}): {ex}")
        return []

    cols = getattr(ds, "column_names", [])
    txt_col = next(
        (
            c
            for c in ("text", "snippet", "code", "content", "statement", "line")
            if c in cols
        ),
        cols[0] if cols else None,
    )
    if txt_col is None:
        print("[p1-auth-hf] no recognizable text column")
        return []

    label_col = next(
        (c for c in ("label", "leak", "is_secret", "in_clean_set", "vulnerable") if c in cols),
        None,
    )

    out: list[dict[str, Any]] = []
    tl = MAX_TEXT_HARD()
    for i, row in enumerate(ds):
        if len(out) >= cap:
            break
        text = str(row.get(txt_col) or "").strip()
        if len(text) < 8:
            continue
        if label_col:
            lv = row[label_col]
            if isinstance(lv, bool):
                risk = "high" if lv else "low"
            elif isinstance(lv, (int, float)):
                risk = "high" if int(lv) != 0 else "low"
            else:
                s = str(lv).strip().upper()
                risk = (
                    "low"
                    if s in ("0", "FALSE", "CLEAN", "NEGATIVE", "SAFE", "BENIGN")
                    else "high"
                )
        else:
            risk = "high"
        if len(text) > tl:
            text = text[:tl]
        out.append(row_from_text(text, risk, f"authhf-{i}", "auth_hf_p1"))

    if out:
        print(f"[p1-auth-hf] merged {len(out)} rows from {ds_spec}")
    return out


def merge_p1_optional_rows_into(*, financial: bool = True, i2b2: bool = True, auth: bool = True) -> list[dict[str, Any]]:
    """Gather all configured P1 sources (no duplicates across calls — caller merges into pool)."""
    merged: list[dict[str, Any]] = []
    if financial:
        merged.extend(load_financial_balanced_jsonl())
    if i2b2:
        merged.extend(load_i2b2_2014_xml_dir())
    if auth:
        merged.extend(load_auth_secret_patterns_jsonl())
        merged.extend(load_auth_hf_optional())
    return merged
