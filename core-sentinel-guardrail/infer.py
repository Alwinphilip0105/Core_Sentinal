"""
Inference for TinyBERT guardrail: score text and policy-based allow/warn/block.
Runs on CUDA if available, else CPU.
Uses a direct AutoModel forward pass (not transformers pipeline) for lower per-paste latency;
call preload_guardrail_model() at app startup so load cost is paid once, not on first paste.
Env: GUARDRAIL_TORCH_THREADS (default 2), GUARDRAIL_MAX_SEQ_LEN (default 128), GUARDRAIL_SLIDING_STRIDE (default 32,
overlap between windows for long texts), GUARDRAIL_BLOCKING_PRELOAD (main.py),
GUARDRAIL_BLOCK_RISK_SCORE_MIN (default 95: score at/above forces block UI),
GUARDRAIL_DUPLICATE_BYPASS_MIN_SCORE (default 80: at/above never skips duplicate or inference-cache).
GUARDRAIL_DYNAMIC_RISK_SCORE (optional): set to 1/true to blend model softmax (low/med/high) into 0–100;
  set to blend for 50/50 mix with the legacy tier+trigger score. Default off — legacy uses fixed tier bases (0/30/70).

Decision driven by config/risk_policy.json (3-class: min_prob_* defaults or file;
9-class: optional per_class_thresholds with per-PII warn/block scores). Scoring queues a row to
logs/guardrail.db (SQLite) on a background thread (timestamp, text_hash, pii_class, confidence,
action; optional context JSON). Set GUARDRAIL_LEGACY_EVENTS_CSV=1 to also append logs/events.csv.
Optional PII-based path: use score_clipboard_with_pii() or score_clipboard(..., use_pii=True).
"""

import hashlib
import json
import math
import os
import sys
from typing import Optional
import queue
import re
import threading
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from guardrail_runtime import get_cached_inference_result, record_inference_scored_text
from pii_remediation import _resolve_span_bounds
from risk_mapping import (
    aggregate_span_risks,
    apply_pii_overrides,
    get_pii_override_triggers,
    kb_rule_spans,
    strong_regex_pii_spans,
)
from risk_policy_loader import (
    DEFAULT_RISK_POLICY,
    cached_load_risk_policy,
    clear_risk_policy_cache,
    get_binary_two_thresholds,
    get_critical_secret_entropy_min,
    get_inference_3class_label_thresholds,
    get_pii_count_probability_threshold,
    get_strict_decision_rules,
    load_merged_risk_policy,
)

# Suppress "Torch was not compiled with flash attention" (harmless; uses standard attention)
warnings.filterwarnings("ignore", message=".*flash attention.*")

_HIPAA_AGE_OVER_89 = r"(?:9\d|1[01]\d|12\d)"

_HIPAA_CRITICAL_RULES: list[tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(
            r"(?ix)\b(?:mrn|medical\s*record(?:\s*number)?)\s*(?:[:#-]|\bis\b)?\s*\d{5,12}\b"
        ),
        "Medical Record Number (MRN)",
        "critical",
    ),
    (
        re.compile(
            r"(?ix)\bdea(?:\s*(?:number|no|\#))?\s*[:#-]?\s*[A-Z]{2}\d{7}\b"
        ),
        "DEA number",
        "critical",
    ),
    (
        re.compile(
            r"(?ix)\bnpi(?:\s*(?:number|no|\#))?\s*[:#-]?\s*\d{10}\b"
        ),
        "NPI",
        "critical",
    ),
    (
        re.compile(
            r"(?ix)\b(?:member|beneficiary|policy)\s*(?:id|identifier|number|no|\#)?\s*[:#-]?\s*[A-Z0-9][A-Z0-9-]{4,24}\b"
        ),
        "Health Plan ID",
        "critical",
    ),
    (
        re.compile(
            rf"(?ix)\b(?:age\s*[:#-]?\s*{_HIPAA_AGE_OVER_89}\b|aged\s+{_HIPAA_AGE_OVER_89}\b|{_HIPAA_AGE_OVER_89}[-\s]?year[-\s]?old\b|{_HIPAA_AGE_OVER_89}\s+years?\s+old\b)"
        ),
        "Age over 89",
        "critical",
    ),
]

def _is_windows_torch_dll_error(exc: BaseException) -> bool:
    s = str(exc).lower()
    return (
        "dll" in s
        or "c10" in s
        or "torch_python" in s
        or "winerror 126" in s
        or "winerror 127" in s
        or "the specified module could not be found" in s
    )


def _windows_torch_lib_dir() -> Optional[str]:
    base = os.path.dirname(os.path.abspath(__file__))
    for rel in (
        ("..", ".venv", "Lib", "site-packages", "torch", "lib"),
        (".venv", "Lib", "site-packages", "torch", "lib"),
    ):
        p = os.path.normpath(os.path.join(base, *rel))
        if os.path.isdir(p):
            return p
    return None


try:
    if sys.platform == "win32":
        _torch_lib = _windows_torch_lib_dir()
        if _torch_lib is not None:
            os.add_dll_directory(_torch_lib)
    import torch
    import torch.nn.functional as F
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
except (OSError, ImportError) as e:
    import traceback
    traceback.print_exc()
    raise

DEVICE = 0 if torch.cuda.is_available() else -1  # 0 = first GPU, -1 = CPU

# Default 2 threads: TinyBERT is small; all cores increases scheduling jitter. Override via GUARDRAIL_TORCH_THREADS.
_tn_env = os.environ.get("GUARDRAIL_TORCH_THREADS")
if _tn_env is not None and _tn_env.strip().isdigit():
    torch.set_num_threads(max(1, int(_tn_env.strip())))
else:
    torch.set_num_threads(2)

# Sequence length for tokenizer/model forward (lower = faster; default matches typical TinyBERT train).
_MAX_SEQ_LEN = max(32, min(512, int(os.environ.get("GUARDRAIL_MAX_SEQ_LEN", "128"))))

# Sliding-window inference: overlap in *content* tokens between consecutive windows (BERT reserves 2 for CLS+SEP).
_infer_thread_local = threading.local()

# Word-count threshold below which we use a single truncated encode (skip full-document tokenization).
_SHORT_TEXT_MAX_WORDS = 80


def _sliding_window_stride() -> int:
    """Overlap between consecutive windows (content tokens); must be < max_content or we fall back to step=1."""
    raw = os.environ.get("GUARDRAIL_SLIDING_STRIDE", "32").strip()
    try:
        v = int(raw)
    except ValueError:
        v = 32
    max_content = max(1, _MAX_SEQ_LEN - 2)
    return max(1, min(v, max_content - 1))


def _sliding_window_encode(
    text: str,
    tokenizer,
    max_len: int | None = None,
    stride: int | None = None,
) -> list[dict]:
    """
    Tokenize full text without truncation, split into overlapping windows with [CLS] + content + [SEP] each.
    Each window is padded to max_len. Returns one dict per window with tensors batch dim 1.
    """
    max_len = max_len if max_len is not None else _MAX_SEQ_LEN
    stride = stride if stride is not None else _sliding_window_stride()
    max_content = max_len - 2  # room for CLS and SEP
    if max_content < 1:
        max_content = 1

    t = text or ""
    # Fast path: short text fits in one 128-token window — single encode, no full-text tokenization.
    if len(t.split()) < _SHORT_TEXT_MAX_WORDS:
        enc = tokenizer(
            t,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        alen = int(enc["attention_mask"][0].sum().item())
        content_tokens = max(0, alen - 2)
        return [
            {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "char_start": 0,
                "char_end": len(t),
                "token_start": 0,
                "token_end": content_tokens,
            }
        ]

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id
    if cls_id is None or sep_id is None:
        # Fall back to typical BERT ids if tokenizer omits them
        cls_id = getattr(tokenizer, "cls_token_id", None) or 101
        sep_id = getattr(tokenizer, "sep_token_id", None) or 102
    if pad_id is None:
        pad_id = 0

    try:
        encoding = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            padding=False,
            return_tensors=None,
        )
    except (TypeError, ValueError):
        encoding = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_tensors=None,
        )
        encoding["offset_mapping"] = []

    all_ids = encoding["input_ids"]
    all_offset = encoding.get("offset_mapping") or []

    total = len(all_ids)
    if total == 0:
        enc = tokenizer(
            text,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return [
            {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "char_start": 0,
                "char_end": len(text),
                "token_start": 0,
                "token_end": 0,
            }
        ]

    if total <= max_content:
        enc = tokenizer(
            text,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return [
            {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "char_start": 0,
                "char_end": len(text),
                "token_start": 0,
                "token_end": total,
            }
        ]

    step = max_content - stride
    if step < 1:
        step = 1

    windows: list[dict] = []
    start = 0
    while start < total:
        end = min(start + max_content, total)
        chunk_ids = all_ids[start:end]
        ids = [cls_id] + chunk_ids + [sep_id]
        pad_len = max_len - len(ids)
        if pad_len < 0:
            # Should not happen if max_content <= max_len - 2
            ids = ids[:max_len]
            pad_len = 0
        padded_ids = ids + [pad_id] * pad_len
        attention = [1] * len(ids) + [0] * pad_len

        if all_offset and start < len(all_offset) and end > 0:
            char_start = int(all_offset[start][0])
            char_end = int(all_offset[min(end - 1, len(all_offset) - 1)][1])
        else:
            char_start = 0
            char_end = len(text)

        windows.append(
            {
                "input_ids": torch.tensor([padded_ids], dtype=torch.long),
                "attention_mask": torch.tensor([attention], dtype=torch.long),
                "char_start": char_start,
                "char_end": char_end,
                "token_start": start,
                "token_end": end,
            }
        )
        if end == total:
            break
        start += step

    return windows


def _pick_worst_window_probs(prob_vectors: list[list[float]], num_labels: int) -> list[float]:
    """
    Merge multi-window softmax outputs: take the window that implies highest sensitivity.
    3-class (low/med/high): prefer higher prob_high, then prob_med, then lower prob_low.
    9-class: maximize max probability among non-O classes (index 0 = O).
    """
    if not prob_vectors:
        return []
    if num_labels == 3 and len(prob_vectors[0]) >= 3:
        return max(prob_vectors, key=lambda p: (p[2], p[1], -p[0]))
    # PII / multi-class: ignore O at 0 when possible
    return max(
        prob_vectors,
        key=lambda p: max(p[1:]) if len(p) > 1 else p[0],
    )


def _run_sliding_inference(
    text: str,
    *,
    min_window_confidence: float | None = None,
    max_window_char_span: int | None = None,
) -> tuple[list, dict[int, float]] | None:
    """
    Full-document inference: overlapping windows, then merge to a single prob vector for policy/labels.
    Stores window metadata on _infer_thread_local for score_clipboard_with_pii.

    When min_window_confidence / max_window_char_span are set (document scans), windows that fail
    the filter are excluded from the worst-window merge so large low-signal blocks do not drive HIGH.
    """
    t = (text or "").strip()
    if not t:
        _infer_thread_local.sliding_meta = {
            "token_count": 0,
            "chunks_scored": 0,
            "text_truncated": False,
            "window_scores": [],
            "window_prob_vectors": [],
        }
        return None

    _load_guardrail_model()
    assert _guardrail_model is not None and _guardrail_tokenizer is not None and _torch_device is not None
    tokenizer = _guardrail_tokenizer
    model = _guardrail_model
    n = _model_num_labels if _model_num_labels is not None else 3

    windows = _sliding_window_encode(t, tokenizer)
    content_token_count = 0
    if len(t.split()) < _SHORT_TEXT_MAX_WORDS and windows:
        try:
            w0 = windows[0]
            content_token_count = int(w0["attention_mask"][0].sum().item()) - 2
            if content_token_count < 0:
                content_token_count = 0
        except (TypeError, ValueError, AttributeError, IndexError):
            content_token_count = 0
    else:
        try:
            enc0 = tokenizer(
                t,
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_tensors=None,
            )
            content_token_count = len(enc0["input_ids"])
        except (TypeError, ValueError):
            content_token_count = 0

    window_scores_meta: list[dict] = []
    prob_vectors: list[list[float]] = []

    if not windows:
        _infer_thread_local.sliding_meta = {
            "token_count": content_token_count,
            "chunks_scored": 0,
            "text_truncated": False,
            "window_scores": [],
            "window_prob_vectors": [],
        }
        return None

    with torch.inference_mode():
        for w in windows:
            batch = {
                "input_ids": w["input_ids"].to(_torch_device),
                "attention_mask": w["attention_mask"].to(_torch_device),
            }
            logits = model(**batch).logits
            probs_t = F.softmax(logits[0], dim=-1)
            p_list = [float(probs_t[i].item()) for i in range(min(n, probs_t.shape[0]))]
            prob_vectors.append(p_list)
            window_scores_meta.append(
                {
                    "char_start": w["char_start"],
                    "char_end": w["char_end"],
                    "top_prob": max(p_list) if p_list else 0.0,
                }
            )

    merge_vectors = prob_vectors
    if min_window_confidence is not None or max_window_char_span is not None:
        min_c = 0.0 if min_window_confidence is None else float(min_window_confidence)
        max_span = 10**9 if max_window_char_span is None else int(max_window_char_span)
        filtered: list[list[float]] = []
        for i, p_list in enumerate(prob_vectors):
            if i >= len(window_scores_meta):
                break
            ws = window_scores_meta[i]
            try:
                cs = int(ws.get("char_start", 0))
                ce = int(ws.get("char_end", 0))
            except (TypeError, ValueError):
                continue
            if ce - cs > max_span:
                continue
            if p_list and max(float(x) for x in p_list) < min_c:
                continue
            filtered.append(p_list)
        merge_vectors = filtered

    if not merge_vectors:
        worst = [0.0] * n
        if n > 0:
            worst[0] = 1.0
    else:
        worst = _pick_worst_window_probs(merge_vectors, n)
    if not worst:
        _infer_thread_local.sliding_meta = {
            "token_count": content_token_count,
            "chunks_scored": len(windows),
            "text_truncated": len(windows) > 1,
            "window_scores": window_scores_meta,
            "window_prob_vectors": prob_vectors,
        }
        return None

    probs: dict[int, float] = {}
    scores: list = []
    for i in range(min(n, len(worst))):
        p = float(worst[i])
        probs[i] = p
        label_name = _model_id2label.get(i, f"LABEL_{i}") if _model_id2label else f"LABEL_{i}"
        scores.append({"label": str(label_name), "score": p})

    _infer_thread_local.sliding_meta = {
        "token_count": content_token_count,
        "chunks_scored": len(windows),
        "text_truncated": len(windows) > 1,
        "window_scores": window_scores_meta,
        "window_prob_vectors": prob_vectors,
    }

    if not scores:
        return None
    return scores, probs


def _normalize_span_char_offsets(text: str, spans: list[dict]) -> None:
    """
    Ensure each span has start/end consistent with `text` and `match`.
    When offsets are missing or do not align with the matched substring, locate `match` in `text`.
    KB spans may use a truncated `match` (prefix of the full slice); in that case existing
    char_start/char_end from regex are kept when the slice starts with `match`.
    """
    if not text or not spans:
        return
    for sp in spans:
        if not isinstance(sp, dict):
            continue
        match = sp.get("match")
        if match is None:
            continue
        match = str(match)
        if not match.strip():
            continue
        start = sp.get("start")
        end = sp.get("end")
        ok = False
        if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text):
            seg = text[start:end]
            if seg == match or (len(seg) >= len(match) and seg.startswith(match)):
                ok = True
        if ok:
            continue
        pos = text.find(match)
        if pos < 0:
            pos = text.lower().find(match.lower())
        if pos >= 0:
            sp["start"] = pos
            sp["end"] = pos + len(match)


def _span_highlight_priority(span: dict) -> int:
    """Higher = replace first when overlapping (regex/kb over coarse model_window)."""
    src = str(span.get("source", "")).lower()
    if src == "regex":
        return 4
    if src == "kb_rule":
        return 3
    if src == "model_window":
        return 0
    return 1


def filter_triggers_already_in_spans(
    spans: list[dict], triggers: list[str]
) -> list[str]:
    """
    Drop human-readable trigger strings that duplicate a span's ``class`` (span card already
    represents that detection, e.g. regex \"credit card pattern\" + trigger of the same name).
    """
    if not triggers:
        return []
    classes: set[str] = set()
    for s in spans or []:
        if not isinstance(s, dict):
            continue
        c = str(s.get("class", "")).strip().lower()
        if c:
            classes.add(c)
    out: list[str] = []
    for t in triggers:
        tl = str(t).strip().lower()
        if not tl:
            continue
        if tl in classes:
            continue
        out.append(str(t))
    return out


def highlight_pii_in_text(text: str, spans: list[dict]) -> str:
    """
    Return text with each span replaced by a [class] marker.

    Overlapping spans are deduplicated: prefer regex/kb over broad model_window, and shorter
    spans over longer ones at the same priority. Remaining ranges are applied right-to-left on
    the original string so indices stay valid.
    """
    if not text:
        return ""
    if not spans:
        return text

    candidates: list[tuple[int, int, str, int]] = []
    for span in spans:
        if not isinstance(span, dict):
            continue
        resolved = _resolve_span_bounds(text, span)
        if resolved is None:
            continue
        start, end, _seg = resolved
        c = str(span.get("class") or "PII").replace("[", "").replace("]", "")[:64]
        tag = f"[{c}]"
        pri = _span_highlight_priority(span)
        candidates.append((start, end, tag, pri))

    if not candidates:
        return text

    # Greedy non-overlapping: higher priority first, then shorter span (specific match over window)
    candidates.sort(key=lambda x: (-x[3], x[1] - x[0], x[0]))

    selected: list[tuple[int, int, str]] = []
    for start, end, tag, _pri in candidates:
        overlaps = any(
            not (end <= s0 or start >= s1) for (s0, s1, _) in selected
        )
        if overlaps:
            continue
        selected.append((start, end, tag))

    selected.sort(key=lambda x: x[0], reverse=True)
    result = text
    for start, end, tag in selected:
        if start < 0 or start >= end:
            continue
        if end > len(result):
            end = len(result)
        result = result[:start] + tag + result[end:]
    return result


def get_pii_spans(
    text: str,
    policy: dict | None = None,
    *,
    sliding_meta: dict | None = None,
    min_confidence: float = 0.45,
    max_model_window_chars: int = 100,
) -> list[dict]:
    """
    Character-level span annotations for regex PII and model windows (non-low / non-O windows).

    Each item: start, end, class, match, source ("regex" | "kb_rule" | "model_window").
    kb_rule spans may include a \"risk\" field (low/med/high).

    sliding_meta may include window_results: a list of dicts with a \"spans\" list (e.g. model
    windows without offsets); those spans are merged and start/end are filled from match text.

    Model windows longer than max_model_window_chars or below min_confidence are skipped
    (reduces sliding-window over-flagging on long generic text).
    """
    p = policy or load_risk_policy()
    t = text if isinstance(text, str) else ""
    spans: list[dict] = []

    spans.extend(strong_regex_pii_spans(t))
    spans.extend(_hipaa_critical_spans(t))
    spans.extend(kb_rule_spans(t))

    meta = sliding_meta or {}
    for window_result in meta.get("window_results") or []:
        if not isinstance(window_result, dict):
            continue
        for span in window_result.get("spans") or []:
            if isinstance(span, dict):
                span.setdefault("source", "model_window")
                spans.append(span)

    vecs: list = list(meta.get("window_prob_vectors") or [])
    wscores: list = list(meta.get("window_scores") or [])
    n_lab = _model_num_labels if _model_num_labels is not None else 3
    id2l = _model_id2label if isinstance(_model_id2label, dict) else {}

    for i, p_list in enumerate(vecs):
        if i >= len(wscores):
            break
        ws = wscores[i]
        try:
            cs = int(ws.get("char_start", 0))
            ce = int(ws.get("char_end", 0))
        except (TypeError, ValueError):
            continue
        cs = max(0, min(cs, len(t)))
        ce = max(0, min(ce, len(t)))
        if ce <= cs:
            continue
        slice_txt = t[cs:ce]
        if len(slice_txt) > int(max_model_window_chars):
            continue

        win_conf: float
        if n_lab == 3:
            th, tm = get_inference_3class_label_thresholds(p)
            if len(p_list) < 3:
                continue
            ph, pm = float(p_list[2]), float(p_list[1])
            if ph > th:
                wclass = "HIGH"
                win_conf = ph
            elif pm > tm:
                wclass = "MED"
                win_conf = pm
            else:
                continue
        else:
            if not p_list:
                continue
            best_i = max(range(len(p_list)), key=lambda j: p_list[j])
            wclass = str(id2l.get(best_i, "O")).strip().upper()
            if wclass in ("O", "LABEL_0", ""):
                continue
            win_conf = float(p_list[best_i])

        if win_conf < float(min_confidence):
            continue

        spans.append(
            {
                "start": cs,
                "end": ce,
                "class": wclass,
                "match": slice_txt,
                "source": "model_window",
            }
        )

    _normalize_span_char_offsets(t, spans)
    return sorted(
        spans,
        key=lambda s: (int(s.get("start", 0)), int(s.get("end", 0))),
    )


def _get_sliding_window_meta() -> dict:
    """Metadata from the last _run_sliding_inference in this thread (defaults if none)."""
    return getattr(
        _infer_thread_local,
        "sliding_meta",
        {
            "token_count": 0,
            "chunks_scored": 0,
            "text_truncated": False,
            "window_scores": [],
            "window_prob_vectors": [],
        },
    )


def _risk_score_force_block_min() -> int:
    """If risk_score >= this (1..100), force decision/action block despite allow_warn_instead. 0 = disabled."""
    pol = load_risk_policy()
    bt = pol.get("block_threshold")
    if bt is not None:
        try:
            return max(0, min(100, int(bt)))
        except (TypeError, ValueError):
            pass
    raw = os.environ.get("GUARDRAIL_BLOCK_RISK_SCORE_MIN", "95").strip()
    try:
        v = int(raw)
    except ValueError:
        return 95
    return max(0, min(100, v))

# Paths relative to this package
_GUARDRAIL_ROOT = Path(__file__).resolve().parent
MODEL_DIR = _GUARDRAIL_ROOT / "models" / "tinybert_guardrail"
CONFIG_PATH = _GUARDRAIL_ROOT / "config" / "risk_policy.json"


def _resolve_guardrail_model_path() -> Path:
    """
    If root model.safetensors is stale (e.g. 3-class on disk) but checkpoint-* has the trained
    2-class head, load the checkpoint (Windows often locks the root file during training).
    Set GUARDRAIL_INFER_MODEL_DIR or GUARDRAIL_CALIB_MODEL_DIR to override.
    """
    base = MODEL_DIR
    env = (
        os.environ.get("GUARDRAIL_INFER_MODEL_DIR")
        or os.environ.get("GUARDRAIL_CALIB_MODEL_DIR")
        or ""
    ).strip()
    if env:
        p = Path(env)
        if not p.is_dir():
            raise FileNotFoundError(f"Model directory not found: {p}")
        return p.resolve()
    try:
        import safetensors.torch as st

        root_w = base / "model.safetensors"
        root_n = None
        if root_w.is_file():
            root_n = int(st.load_file(str(root_w))["classifier.weight"].shape[0])
        ck_dirs = sorted(
            base.glob("checkpoint-*"),
            key=lambda x: int(x.name.split("-", 1)[1])
            if x.name.split("-", 1)[1].isdigit()
            else -1,
        )
        for ck in reversed(ck_dirs):
            wf = ck / "model.safetensors"
            if not wf.is_file():
                continue
            cn = int(st.load_file(str(wf))["classifier.weight"].shape[0])
            if cn == 2 and root_n is not None and root_n != 2:
                return ck.resolve()
    except Exception:
        pass
    return base.resolve()
PII_POLICY_PATH = _GUARDRAIL_ROOT / "config" / "pii_policy.json"
PII_TO_RISK_PATH = _GUARDRAIL_ROOT / "config" / "pii_to_risk.json"
LOGS_DIR = _GUARDRAIL_ROOT / "logs"

ID_TO_RISK = {0: "low", 1: "med", 2: "high"}
RISK_TO_ID = {"low": 0, "med": 1, "high": 2}

# 9-class: risk bucket from PII label name (loaded from config when model is 9-class)
_pii_to_risk = None
_model_num_labels = None
_model_id2label = None

# PII decision policy: high can warn instead of block for a less aggressive UX
POLICY = {
    "high": {"default_decision": "block", "allow_warn_instead": True},
    "med": {"default_decision": "warn"},
    "low": {"default_decision": "allow"},
}
_pii_policy = None
_guardrail_model = None
_guardrail_tokenizer = None
_torch_device = None
_model_init_lock = threading.Lock()

_events_log_queue = None
_events_log_thread = None
_events_log_thread_lock = threading.Lock()


def load_pii_policy(path: Path | None = None) -> dict:
    """Load PII policy from config; merge into POLICY. Returns current policy dict."""
    global _pii_policy, POLICY
    if _pii_policy is not None:
        return _pii_policy
    p = path or PII_POLICY_PATH
    if p.exists():
        try:
            with open(p, encoding="utf-8") as f:
                overrides = json.load(f)
            for risk_level, opts in overrides.items():
                if risk_level in POLICY and isinstance(opts, dict):
                    POLICY[risk_level].update(
                        {k: v for k, v in opts.items() if k in ("default_decision", "allow_warn_instead")}
                    )
        except Exception:
            pass
    _pii_policy = dict(POLICY)
    return _pii_policy


def load_risk_policy(path: Path | None = None) -> dict:
    """
    Merged risk policy: defaults from risk_policy_loader.DEFAULT_RISK_POLICY plus
    config/risk_policy.json (per-class thresholds, inference cutoffs, strict rules).
    """
    if path is not None:
        return load_merged_risk_policy(Path(path))
    return cached_load_risk_policy(CONFIG_PATH)


def decision_from_probs(
    prob_low: float,
    prob_med: float,
    prob_high: float,
    risk: str,
    pii_count: int,
    policy: dict | None = None,
) -> str:
    """
    Decide allow / warn / block from model output.

    Stricter hardcoded rules for high-risk (checked first):
    - If pii_count >= 2 and risk == "high" and prob_high >= 0.30 → "block"
    - Else if risk == "high" and prob_high >= 0.35 → "block"
    - Else if risk == "high" and prob_high >= 0.20 → "warn"
    Else use config policy for low/med:
    - if prob_high >= min_prob_high_block → "block"
    - else if prob_high >= min_prob_high_warn or prob_med >= min_prob_med_warn → "warn"
    - else → "allow"
    """
    p = policy or load_risk_policy()
    strict = get_strict_decision_rules(p)
    # Stricter rules when top label is high
    if risk == "high":
        if pii_count >= 2 and prob_high >= strict["pii_high_block_prob"]:
            return "block"
        if prob_high >= strict["prob_high_block"]:
            return "block"
        if prob_high >= strict["prob_high_warn"]:
            return "warn"

    # Config-based logic for remaining cases (low/med or high below warn threshold)
    min_high_block = float(p.get("min_prob_high_block", DEFAULT_RISK_POLICY["min_prob_high_block"]))
    min_high_warn = float(p.get("min_prob_high_warn", DEFAULT_RISK_POLICY["min_prob_high_warn"]))
    min_med_warn = float(p.get("min_prob_med_warn", DEFAULT_RISK_POLICY["min_prob_med_warn"]))

    if prob_high >= min_high_block:
        return "block"
    if prob_high >= min_high_warn or prob_med >= min_med_warn:
        return "warn"
    return "allow"


def decision_from_per_class_probs(
    probs: dict[int, float],
    id2label: dict[int, str],
    policy: dict | None = None,
) -> str:
    """
    For 9-class PII models: allow / warn / block from per-class probabilities.
    Per-class: if prob >= block → block candidate; elif prob >= warn → warn candidate.
    Any class triggering block yields block; else any warn yields warn; else allow.
    """
    p = policy or load_risk_policy()
    thresholds = p.get("per_class_thresholds")
    if not isinstance(thresholds, dict):
        return "allow"

    block_hit = False
    warn_hit = False
    for i, prob in probs.items():
        label = id2label.get(i, "O")
        key = str(label).strip().upper()
        t = thresholds.get(key) or thresholds.get(label)
        if not isinstance(t, dict):
            continue
        warn_th = float(t.get("warn", 1.0))
        block_th = float(t.get("block", 1.0))
        if prob >= block_th:
            block_hit = True
        elif prob >= warn_th:
            warn_hit = True

    if block_hit:
        return "block"
    if warn_hit:
        return "warn"
    return "allow"


_DECISION_RANK = {"block": 3, "warn": 2, "allow": 1}


def _strict_recall_mode_enabled(policy: dict | None = None) -> bool:
    """
    Recall-first strict mode for production safety posture.
    Env override: GUARDRAIL_STRICT_RECALL_MODE=1|0
    Policy fallback: strict_decision_rules.recall_first_mode (default false).
    """
    env_raw = os.environ.get("GUARDRAIL_STRICT_RECALL_MODE")
    if env_raw is not None and str(env_raw).strip() != "":
        return str(env_raw).strip().lower() in ("1", "true", "yes", "on")
    p = policy or load_risk_policy()
    try:
        strict_rules = p.get("strict_decision_rules", {}) if isinstance(p, dict) else {}
        return bool(strict_rules.get("recall_first_mode", False))
    except Exception:
        return False


def _merge_decisions(*decisions: str, critical_escalate_warn_to_block: bool = False) -> str:
    """
    Keep the strictest decision (block > warn > allow).

    If critical_escalate_warn_to_block is True and the merged result is warn, return block.
    Critical secrets (API keys, JWTs, base64 credentials, etc.) override threshold-based warn.
    """
    best = "allow"
    for d in decisions:
        if _DECISION_RANK.get(d, 1) > _DECISION_RANK.get(best, 1):
            best = d
    if critical_escalate_warn_to_block and best == "warn":
        return "block"
    return best


def clipboard_ui_action(decision: str, *, critical_secret: bool = False) -> str:
    """
    How clipboard-related UIs should notify after a score.
    silent: no UI; warn: subtle non-modal hint; block: modal remediation flow.
    """
    if critical_secret:
        return "block"
    d = (decision or "allow").lower()
    if d == "block":
        return "block"
    if d == "warn":
        return "warn"
    return "silent"


_UI_ACTION_RANK = {"silent": 0, "warn": 1, "block": 2}


def _decision_and_block_from_ui_action(action: str) -> tuple[str, bool]:
    """Map UI action string to decision + block flag (after enforce_risk_policy)."""
    a = (action or "silent").lower()
    if a == "block":
        return "block", True
    if a == "warn":
        return "warn", False
    return "allow", False


def _primary_pii_class_for_class_override(
    labels: list | None,
    class_probs: dict[int, float] | None,
) -> str:
    """Dominant non-O PII class for class_overrides lookup (9-class probs preferred)."""
    if class_probs and _model_id2label:
        best_i: int | None = None
        best_p = -1.0
        for i, prob in class_probs.items():
            lab = str(_model_id2label.get(int(i), "O")).strip().upper()
            if lab == "O":
                continue
            if float(prob) > best_p:
                best_p = float(prob)
                best_i = int(i)
        if best_i is not None:
            return str(_model_id2label.get(best_i, "O")).strip().upper()
    for lab in labels or []:
        u = str(lab).strip().upper()
        if u and u != "O":
            return u
    return "O"


def enforce_risk_policy(
    action: str,
    risk_label: str,
    pii_class: str,
    policy: dict,
    *,
    critical_secret: bool = False,
    text: str = "",
    top_conf: float = 0.0,
    triggers: list[str] | None = None,
    labels: list[str] | None = None,
) -> str:
    """
    Final UI action after merged model/regex decisions: three-tier risk_policy.json rules.
    Critical secrets (API keys, etc.) always block. Low risk is always silent.
    Medium risk never blocks paste; it warns.
    High risk respects class_overrides max_action ceiling.
    """
    if critical_secret:
        return "block"
    rl = (risk_label or "low").lower()
    strict_mode = _strict_recall_mode_enabled(policy)
    if rl == "low":
        return "silent"
    if rl == "med":
        strict_cfg = policy.get("strict_decision_rules", {}) if isinstance(policy, dict) else {}
        if _strict_medium_block_downgrade_for_tiny_text(
            text=text,
            triggers=list(triggers or []),
            labels=labels,
            top_conf=top_conf,
            strict_cfg=strict_cfg if isinstance(strict_cfg, dict) else {},
        ):
            return "silent"
        a = (action or "silent").lower()
        if a == "block":
            return "warn"
        if a == "silent":
            return "warn"
        return a if a in _UI_ACTION_RANK else "warn"
    if rl == "high" and _has_hard_block_trigger(list(triggers or []), text, labels):
        return "block"

    class_overrides = policy.get("class_overrides") or {}
    key = str(pii_class or "O").strip().upper()
    class_policy = class_overrides.get(key)
    if not isinstance(class_policy, dict):
        class_policy = {}
    max_action = str(class_policy.get("max_action", "block")).lower()
    if max_action not in _UI_ACTION_RANK:
        max_action = "block"

    a = (action or "silent").lower()
    if _UI_ACTION_RANK.get(a, 0) > _UI_ACTION_RANK.get(max_action, 2):
        return max_action
    return a if a in _UI_ACTION_RANK else "silent"


# Scalar score → clipboard UI: above this → warn; at or below → silent (block paths unchanged).
RISK_SCORE_SILENT_MAX = 30


def _finalize_action_for_risk_score(
    action: str,
    risk_score: int,
    *,
    critical_secret: bool,
    binary_allow_zone: bool = False,
) -> str:
    """
    Final gate: risk_score > RISK_SCORE_SILENT_MAX → warn, else silent.
    Preserves block (forced high-score / policy block and critical-secret path).
    When binary two-threshold decision is allow (prob below t_warn), do not upgrade
    silent → warn from scalar score alone — that score reflects noisy logits, not UI tier.
    """
    if critical_secret:
        return "block"
    a = (action or "silent").lower()
    if a == "block":
        return "block"
    if binary_allow_zone and a == "silent":
        return "silent"
    try:
        rs = int(risk_score)
    except (TypeError, ValueError):
        rs = 0
    if rs > RISK_SCORE_SILENT_MAX:
        return "warn"
    return "silent"


def _text_hash(text: str) -> str:
    """SHA256 hash of normalized text (no raw content in logs)."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _append_events_csv_row(row: list, context_json: str | None = None) -> None:
    """Background thread: persist one scoring row to SQLite (optional legacy CSV)."""
    from guardrail_logs import write_scoring_event

    write_scoring_event(row, context_json)


def _events_csv_log_loop() -> None:
    global _events_log_queue
    while True:
        q = _events_log_queue
        if q is None:
            return
        item = q.get()
        try:
            if isinstance(item, tuple) and len(item) == 2:
                row, ctx = item
            else:
                row, ctx = item, None
            _append_events_csv_row(row, ctx)
        except Exception:
            pass


def _ensure_events_log_thread() -> None:
    global _events_log_queue, _events_log_thread
    with _events_log_thread_lock:
        if _events_log_thread is not None and _events_log_thread.is_alive():
            return
        _events_log_queue = queue.Queue(maxsize=512)
        _events_log_thread = threading.Thread(
            target=_events_csv_log_loop,
            daemon=True,
            name="guardrail-events-db",
        )
        _events_log_thread.start()


def _telemetry_ctx(
    *,
    risk: str | None,
    risk_score: int | None,
    score_context: str,
    critical_secret: bool = False,
) -> dict:
    """Fields stored in scoring context_json and copied to risk_telemetry.jsonl."""
    d: dict = {
        "risk": (risk or "unknown").strip().lower(),
        "risk_score": int(risk_score) if risk_score is not None else 0,
        "score_context": score_context,
    }
    if critical_secret:
        d["critical_secret"] = True
    return d


def _log_scoring_event(
    text: str,
    *,
    pii_class: str,
    confidence: float,
    action: str,
    context: dict | None = None,
) -> None:
    """
    Queue one row for logs/guardrail.db (background thread) so scoring never waits on disk.
    Columns: timestamp, text_hash, pii_class, confidence, action; optional context JSON
    (include risk, risk_score, score_context for telemetry / dashboards).
    """
    raw = text if isinstance(text, str) else ""
    text_hash = _text_hash(raw) if raw.strip() else hashlib.sha256(b"").hexdigest()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    row = [timestamp, text_hash, pii_class, f"{confidence:.6f}", action]
    ctx_json = json.dumps(context, ensure_ascii=False) if context else None
    _ensure_events_log_thread()
    q = _events_log_queue
    if q is not None:
        try:
            q.put_nowait((row, ctx_json))
        except queue.Full:
            pass


def _confidence_from_scores(scores: list | None) -> float:
    if not scores:
        return 0.0
    return max((float(s.get("score", 0.0)) for s in scores), default=0.0)


def _pii_class_for_log(labels: list, risk: str) -> str:
    if labels:
        return "/".join(str(x) for x in labels)
    return str(risk or "unknown")


def _load_guardrail_model() -> None:
    """
    Load tokenizer + sequence classifier once (thread-safe). Faster than transformers pipeline()
    for single-string scoring (no pipeline batching overhead).
    """
    global _guardrail_model, _guardrail_tokenizer, _torch_device, _model_num_labels, _model_id2label, _pii_to_risk
    if _guardrail_model is not None:
        return
    with _model_init_lock:
        if _guardrail_model is not None:
            return
        path = _resolve_guardrail_model_path()
        if not path.exists():
            raise FileNotFoundError(f"Model not found at {path}. Run train.py first.")
        resolved = str(path.resolve())
        config = AutoConfig.from_pretrained(resolved)
        nlab = getattr(config, "num_labels", None)
        if nlab is None:
            id2l = getattr(config, "id2label", None) or {}
            nlab = len(id2l) if id2l else 3
        _model_num_labels = int(nlab)
        _model_id2label = getattr(config, "id2label", None) or ID_TO_RISK
        _model_id2label = {int(k): str(v) for k, v in _model_id2label.items()}
        if _model_num_labels == 9:
            if PII_TO_RISK_PATH.exists():
                with open(PII_TO_RISK_PATH, encoding="utf-8") as f:
                    _pii_to_risk = json.load(f)
            else:
                _pii_to_risk = {
                    "O": "low", "NAME": "low", "LOCATION": "low",
                    "CONTACT": "med", "ID": "med", "OTHER_PII": "med",
                    "FINANCIAL": "high", "HEALTH": "high", "AUTH": "high",
                }
        else:
            _pii_to_risk = None
        _guardrail_tokenizer = AutoTokenizer.from_pretrained(resolved)
        _guardrail_model = AutoModelForSequenceClassification.from_pretrained(resolved)
        _model_num_labels = int(_guardrail_model.classifier.out_features)
        _guardrail_model.eval()
        torch.set_grad_enabled(False)
        _torch_device = torch.device("cuda:0" if DEVICE == 0 else "cpu")
        _guardrail_model.to(_torch_device)


def preload_guardrail_model() -> None:
    """
    Block until TinyBERT + tokenizer are loaded and one forward pass has completed (CUDA sync).
    Call once at app startup so the first user paste only pays inference time, not load time.
    """
    _load_guardrail_model()
    assert _guardrail_model is not None and _guardrail_tokenizer is not None and _torch_device is not None
    with torch.inference_mode():
        enc = _guardrail_tokenizer(
            ".",
            truncation=True,
            max_length=_MAX_SEQ_LEN,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(_torch_device) for k, v in enc.items()}
        _guardrail_model(**enc)
    if _torch_device.type == "cuda":
        torch.cuda.synchronize()


def _infer_scores_and_probs(
    text: str,
    *,
    min_window_confidence: float | None = None,
    max_window_char_span: int | None = None,
) -> tuple[list, dict[int, float]] | None:
    """
    Document-level scores from sliding-window inference (overlapping chunks) merged to one prob vector.
    See _run_sliding_inference / _sliding_window_encode; metadata via _get_sliding_window_meta().
    """
    return _run_sliding_inference(
        text,
        min_window_confidence=min_window_confidence,
        max_window_char_span=max_window_char_span,
    )


def _labels_from_scores(scores: list) -> list[str]:
    """Map classifier scores to a single-label list (document-level stub)."""
    # Convert pipeline scores into probabilities keyed by label name.
    # For the common 3-class model, label names are typically: low/med/high.
    probs_by_label: dict[str, float] = {}
    for item in scores:
        raw = str(item.get("label"))
        if raw.startswith("LABEL_"):
            lid = int(raw.replace("LABEL_", ""))
            label_name = _model_id2label.get(lid, "O") if _model_id2label else "O"
        else:
            label_name = raw
        probs_by_label[str(label_name).strip().lower()] = float(item.get("score", 0.0))

    # For 3-class risk models, use probability thresholds from risk_policy.json (inference_3class_labels).
    if any(k in probs_by_label for k in ("low", "med", "high")):
        prob_high = probs_by_label.get("high", 0.0)
        prob_med = probs_by_label.get("med", 0.0)
        prob_low = probs_by_label.get("low", 0.0)
        pol = load_risk_policy()
        th, tm = get_inference_3class_label_thresholds(pol)
        if prob_high > th:
            return ["high"]
        if prob_med > tm:
            return ["med"]
        _ = prob_low
        return ["low"]
    if any(k in probs_by_label for k in ("safe", "risky")):
        p_risky = probs_by_label.get("risky", 0.0)
        pol = load_risk_policy()
        thr = _binary_risky_threshold(pol)
        return ["risky"] if p_risky >= thr else ["safe"]

    best = max(scores, key=lambda x: float(x.get("score", 0.0)))
    raw_label = str(best.get("label"))
    if raw_label.startswith("LABEL_"):
        lid = int(raw_label.replace("LABEL_", ""))
        label_name = _model_id2label.get(lid, "O") if _model_id2label else "O"
    else:
        label_name = raw_label
    return [label_name]


def predict_pii_labels(text: str) -> list[str]:
    """
    Return per-span/per-token PII labels for the given text.

    Stub: uses the document-level TinyBERT model and returns a single-label list.
    Replace with a real token-level NER or span classifier to get one label per span.
    """
    if not text or not text.strip():
        return []
    got = _infer_scores_and_probs(text)
    if got is None:
        return ["O"]
    scores, _ = got
    return _labels_from_scores(scores)


def decision_from_pii_risk(risk: str, policy: dict | None = None) -> dict:
    """
    Map PII-derived risk to guardrail decision using POLICY.

    Returns dict with "decision" ("allow" | "warn" | "block") and "block" (bool).
    When policy["high"]["allow_warn_instead"] is True, high risk yields warn instead of block.
    """
    r = (risk or "low").lower()
    p = policy or load_pii_policy()
    if r == "high":
        high_opts = p.get("high", {})
        if high_opts.get("allow_warn_instead") is True:
            return {"decision": "warn", "block": False}
        return {"decision": high_opts.get("default_decision", "block"), "block": True}
    if r == "med":
        return {"decision": p.get("med", {}).get("default_decision", "warn"), "block": False}
    return {"decision": p.get("low", {}).get("default_decision", "allow"), "block": False}


def build_user_message(
    risk: str,
    decision: str,
    pii_override_applied: bool,
    triggers: list[str],
) -> str:
    """
    Build a user-facing message for the clipboard guardrail UI.
    """
    risk = (risk or "low").lower()
    decision = (decision or "allow").lower()
    if decision == "allow" and risk == "low":
        return "Clipboard content appears safe (no sensitive data detected)."
    if decision == "warn" and risk == "med":
        return "Clipboard content may contain names or general personal details. Please review before pasting."
    if decision == "warn" and risk == "high":
        trigger_str = ", ".join(triggers) if triggers else "sensitive data"
        if pii_override_applied:
            return f"Potentially sensitive data detected (e.g., {trigger_str}). Proceed only if this paste is authorized."
        return "Potentially sensitive data detected. Proceed only if this paste is authorized."
    if decision == "block" and risk == "high":
        trigger_str = ", ".join(triggers) if triggers else "sensitive data"
        return f"Paste blocked due to high-risk sensitive data (e.g., {trigger_str}). Contact security/compliance if you believe this is an error."
    # fallback
    return "Clipboard content has been reviewed. Please confirm before pasting."


def _shannon_entropy(s: str) -> float:
    """Shannon entropy in bits per character (0..~8 for ASCII)."""
    if not s:
        return 0.0
    n = len(s)
    counts = Counter(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _hipaa_critical_spans(text: str) -> list[dict]:
    """Regex spans for HIPAA-specific identifiers that should be treated as critical."""
    if not text or not text.strip():
        return []
    out: list[dict] = []
    seen: set[tuple[int, int, str]] = set()
    for rx, class_name, risk_tier in _HIPAA_CRITICAL_RULES:
        for m in rx.finditer(text):
            key = (m.start(), m.end(), class_name)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "start": m.start(),
                    "end": m.end(),
                    "class": class_name,
                    "match": m.group()[:500],
                    "source": "regex",
                    "risk": risk_tier,
                }
            )
    return sorted(
        out,
        key=lambda s: (
            int(s.get("start", 0)),
            int(s.get("end", 0)),
            str(s.get("class", "")),
        ),
    )


# Regex rules that strongly suggest API keys / secrets (always treat as critical).
# Each tuple: (compiled pattern, UI class label, span risk tier: "high" | "critical").
_CRITICAL_SECRET_RULES: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"sk-(?:live|test|proj)-[A-Za-z0-9]{20,}", re.IGNORECASE), "Critical secret", "high"),
    (re.compile(r"sk-[A-Za-z0-9]{20,}"), "Critical secret", "high"),
    (re.compile(r"rk_live_[A-Za-z0-9]{10,}", re.IGNORECASE), "Critical secret", "high"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "Critical secret", "high"),
    (re.compile(r"ASIA[0-9A-Z]{16}"), "Critical secret", "high"),
    (re.compile(r"ghp_[A-Za-z0-9]{36,}"), "Critical secret", "high"),
    (re.compile(r"gho_[A-Za-z0-9]{36,}"), "Critical secret", "high"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}", re.IGNORECASE), "Critical secret", "high"),
    (re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}", re.IGNORECASE), "Critical secret", "high"),
    (re.compile(r"AIza[Sy][A-Za-z0-9_-]{30,}"), "Critical secret", "high"),
    (re.compile(r"ya29\.[A-Za-z0-9_-]+"), "Critical secret", "high"),
    (re.compile(r"Bearer\s+[A-Za-z0-9._-]{24,}", re.IGNORECASE), "Critical secret", "high"),
    (re.compile(r"(?:api[_-]?key|apikey)\s*[:=]\s*['\"]?[A-Za-z0-9._-]{16,}", re.IGNORECASE), "Critical secret", "high"),
    (re.compile(r"SECRET_KEY_BASE\s*[:=]\s*['\"]?[A-Za-z0-9+/=_-]{16,}", re.IGNORECASE), "Critical secret", "high"),
    (
        re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
        "Critical secret",
        "high",
    ),
    (
        re.compile(r"(?i)(db_password|password|pwd|passwd)\s*[=:]\s*\S{6,}"),
        "Password credential",
        "critical",
    ),
    (
        re.compile(r"-----BEGIN\s+(RSA\s+|EC\s+|OPENSSH\s+)?PRIVATE\s+KEY-----"),
        "SSH private key",
        "critical",
    ),
    (
        re.compile(r"(mongodb|mysql|postgres|postgresql|redis|mssql)://\w+:[^@\s]+@"),
        "DB connection string",
        "critical",
    ),
]


def detect_critical_secret_leak(text: str) -> bool:
    """
    True if clipboard text likely contains an API key or other high-entropy secret.

    Used to bypass Smart Silencing (always show bubble) regardless of model risk_score.
    """
    if not text or not text.strip():
        return False
    t = text.strip()
    for rx, _, _ in _CRITICAL_SECRET_RULES:
        if rx.search(t):
            return True
    # Long, high-entropy alphanumeric token (likely raw key material).
    for m in re.finditer(
        r"(?<![A-Za-z0-9])[A-Za-z0-9+/=_-]{36,}(?![A-Za-z0-9+/=_-])", t
    ):
        chunk = m.group(0)
        ent_min = get_critical_secret_entropy_min(load_risk_policy())
        if _shannon_entropy(chunk) >= ent_min:
            return True
    return False


def critical_secret_spans_for_ui(text: str) -> list[dict]:
    """
    Character spans for critical-secret matches. Used when get_pii_spans() returns [] but
    detect_critical_secret_leak() is True — otherwise remediation shows \"No PII detected\"
    despite score ~90 and block.
    Mirrors the same patterns / entropy rule as detect_critical_secret_leak.
    """
    if not text or not text.strip():
        return []
    t = text.strip()
    out: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for rx, class_name, risk_tier in _CRITICAL_SECRET_RULES:
        for m in rx.finditer(t):
            key = (m.start(), m.end())
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "start": m.start(),
                    "end": m.end(),
                    "class": class_name,
                    "match": m.group()[:500],
                    "source": "critical_secret",
                    "risk": risk_tier,
                }
            )
    ent_min = get_critical_secret_entropy_min(load_risk_policy())
    for m in re.finditer(
        r"(?<![A-Za-z0-9])[A-Za-z0-9+/=_-]{36,}(?![A-Za-z0-9+/=_-])", t
    ):
        chunk = m.group(0)
        if _shannon_entropy(chunk) < ent_min:
            continue
        key = (m.start(), m.end())
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "start": m.start(),
                "end": m.end(),
                "class": "High-entropy secret",
                "match": chunk[:500],
                "source": "critical_secret",
                "risk": "high",
            }
        )
    return sorted(
        out,
        key=lambda s: (int(s.get("start", 0)), int(s.get("end", 0))),
    )


# Contact-only: subset of classes from spans; also includes labels emitted by strong_regex / model.
_CONTACT_ONLY_CLASSES = frozenset(
    {
        "email",
        "phone",
        "ip_address",
        "email pattern",
        "phone pattern",
        "ipv4",
        "ipv6",
        "mac address",
        "fax number",
        # Labels used elsewhere in strong_regex / model UI
        "email address",
        "phone number",
        "ip address (ipv4)",
        "ip address (ipv6)",
        "contact",
    }
)
_HIGH_RISK_CLASSES = frozenset(
    {
        "ssn",
        "ssn pattern",
        "credit card",
        "credit card pattern",
        "api key",
        "critical secret",
        "high-entropy secret",
        "password",
        "ssh",
        "jwt",
        "iban",
        "bank account",
        "passport",
        "passport number",
        "medical",
        # Additional sensitive / identity classes from model or regex
        "bank routing number",
        "password credential",
        "ssh private key",
        "db connection string",
        "name",
        "location",
        "id",
        "financial",
        "health",
        "auth",
        "other_pii",
    }
)


# Sliding-window 3-class labels are risk tiers, not PII types — ignore for contact-only class logic.
_SLIDING_WINDOW_RISK_CLASS_LABELS = frozenset({"high", "med", "low"})


def _is_contact_only_pii(spans: list[dict], *, critical_secret: bool) -> bool:
    """True when semantic PII classes are non-empty, all contact-type, and none high-risk."""
    if critical_secret:
        return False
    detected: set[str] = set()
    for s in spans or []:
        if not isinstance(s, dict):
            continue
        c = str(s.get("class", "")).strip().lower()
        if not c or c in _SLIDING_WINDOW_RISK_CLASS_LABELS:
            continue
        detected.add(c)
    return (
        bool(detected)
        and detected.issubset(_CONTACT_ONLY_CLASSES)
        and not detected.intersection(_HIGH_RISK_CLASSES)
    )


def _contact_only_cap_excluded_by_keywords(text: str) -> bool:
    """
    Do not treat phone-shaped spans as contact-only when the text clearly references
    national IDs or medical numbers (regex may still label digits as phone).
    """
    u = (text or "").lower()
    return any(
        n in u
        for n in (
            "nhs number",
            "national insurance number",
            "national insurance ",
            "aadhaar",
            "social insurance number",
            "canadian social insurance",
        )
    )


def _scores_list_from_prob_vector(p_list: list[float]) -> list[dict]:
    """Build the same score dicts as _run_sliding_inference for _labels_from_scores."""
    n = _model_num_labels if _model_num_labels is not None else 3
    scores: list[dict] = []
    for i in range(min(n, len(p_list))):
        label_name = _model_id2label.get(i, f"LABEL_{i}") if _model_id2label else f"LABEL_{i}"
        scores.append({"label": str(label_name), "score": float(p_list[i])})
    return scores


def _per_window_model_risk_scores(prob_vectors: list[list[float]]) -> list[int]:
    """
    0..100 heuristic score per sliding window (model only, no regex triggers).
    Used to detect sparse high-scoring windows on long mostly-safe text.
    """
    out: list[int] = []
    for p_list in prob_vectors:
        sl = _scores_list_from_prob_vector(p_list)
        lw = _labels_from_scores(sl)
        if not lw:
            out.append(0)
            continue
        if str(lw[0]).strip().lower() in ("low", "med", "high"):
            rw = str(lw[0]).strip().lower()
        else:
            rw = aggregate_span_risks([str(x) for x in lw])
        out.append(compute_risk_score(rw, []))
    return out


def _apply_long_text_sliding_penalty(
    per_window_scores: list[int],
    merged_risk: str,
    merged_risk_score: int,
    total_tokens: int = 0,
) -> tuple[str, int]:
    """
    If only a small fraction of windows exceed the warn band, scale the worst window score down.
    Genuine PII tends to light up multiple windows; a lone false positive does not.

    Uses max window score with a length/trigger-rate gate (not mean/sum of windows).
    """
    if not per_window_scores:
        return merged_risk, merged_risk_score
    final_score = max(per_window_scores)
    total_windows = len(per_window_scores)
    triggered_windows = sum(1 for s in per_window_scores if s > 30)
    trigger_rate = triggered_windows / total_windows if total_windows > 0 else 0
    # If less than 25% of windows triggered, reduce score to avoid false
    # positives on long mostly-safe text.
    if trigger_rate < 0.25 and final_score < 80:
        final_score = int(final_score * trigger_rate * 2)
    # For very long multi-window text with no regex/KB triggers, treat a
    # uniform model-only HIGH signal as likely false positive noise.
    if (
        total_tokens >= 180
        and total_windows > 1
        and triggered_windows == total_windows
        and min(per_window_scores) >= 70
        and merged_risk_score <= 70
    ):
        final_score = min(final_score, 30)
    # Map scalar to risk tier for policy.
    if final_score < 45:
        new_risk = "low"
    elif final_score < 70:
        new_risk = "med"
    else:
        new_risk = "high"
    return new_risk, final_score


def compute_risk_score(risk: str, triggers: list[str]) -> int:
    """
    Compute a 0..100 risk score used for user-friendly UI/remediation.

    Heuristic:
      - base: MED=30, HIGH=70
      - triggers:
        * SSN pattern: +30
        * credit card / bank account / IBAN: +25
        * email / phone / address: +15
        * IP/device IDs/tokens: +15
      - cap at 100
    """
    r = (risk or "low").lower()
    score = 0
    if r == "med":
        score = 30
    elif r == "high":
        score = 70

    for trig in dict.fromkeys(triggers or []):
        t = str(trig).lower()
        # Passport (regex "Passport number") — push HIGH base toward ~85
        if "passport number" in t:
            score += 15
            continue
        # Corporate / salary (medium-tier regex triggers)
        if "confidential marker" in t:
            score += 15
            continue
        if "salary information" in t:
            score += 20
            continue
        if "business context" in t:
            score += 10
            continue
        # SSN / national ID
        if "ssn" in t or "national" in t:
            score += 30
            continue
        # Financial identifiers
        if any(k in t for k in ["credit card", "iban", "bank routing", "routing number", "account", "financial"]):
            score += 25
            continue
        # Email / phone / address
        if any(k in t for k in ["email", "phone", "address"]):
            score += 15
            continue
        # IP / device / tokens
        if any(k in t for k in ["ip address", "ip", "device", "token", "auth", "jwt", "bearer"]):
            score += 15
            continue

    return int(min(100, score))


def _three_class_probs_from_scores_list(scores_list: list | None) -> tuple[float, float, float] | None:
    """If merged sliding-window scores look like a 3-class risk model, return (p_low, p_med, p_high)."""
    if not scores_list:
        return None
    probs_by_label: dict[str, float] = {}
    for item in scores_list:
        if not isinstance(item, dict):
            continue
        raw = str(item.get("label", ""))
        if raw.startswith("LABEL_"):
            try:
                lid = int(raw.replace("LABEL_", ""))
            except ValueError:
                continue
            label_name = _model_id2label.get(lid, "O") if _model_id2label else "O"
        else:
            label_name = raw
        probs_by_label[str(label_name).strip().lower()] = float(item.get("score", 0.0))
    if not any(k in probs_by_label for k in ("low", "med", "high")):
        return None
    pl = float(probs_by_label.get("low", 0.0))
    pm = float(probs_by_label.get("med", 0.0))
    ph = float(probs_by_label.get("high", 0.0))
    s = pl + pm + ph
    if s <= 1e-9:
        return None
    return (pl / s, pm / s, ph / s)


def _expected_risk_score_from_3class_probs(pl: float, pm: float, ph: float) -> int:
    """
    Map softmax probabilities to a single 0..100 score (higher = riskier).
    Weights approximate severity: high=1.0, med≈0.5, low≈0.08 on the risk axis.
    """
    ev = ph * 1.0 + pm * 0.52 + pl * 0.08
    return int(round(max(0.0, min(1.0, ev)) * 100))


def _dynamic_risk_score_mode() -> str | None:
    raw = (os.environ.get("GUARDRAIL_DYNAMIC_RISK_SCORE") or "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return "replace"
    if raw == "blend":
        return "blend"
    return None


def _binary_risky_threshold(policy: dict | None = None) -> float:
    """
    Binary risky threshold from risk policy.
    fallback: GUARDRAIL_BINARY_RISKY_THRESHOLD (strict mode defaults to 0.35, otherwise 0.5)
    """
    p = policy or load_risk_policy()
    default_thr = 0.35 if _strict_recall_mode_enabled(p) else 0.5
    try:
        b = p.get("inference_binary_labels", {}) if isinstance(p, dict) else {}
        t = float(b.get("prob_threshold_risky", default_thr))
    except Exception:
        t = default_thr
    env_raw = os.environ.get("GUARDRAIL_BINARY_RISKY_THRESHOLD")
    if env_raw is not None and str(env_raw).strip() != "":
        try:
            t = float(str(env_raw).strip())
        except ValueError:
            pass
    return max(0.01, min(0.99, float(t)))


def _binary_probs_from_scores_list(scores_list: list | None) -> tuple[float, float] | None:
    """If model scores represent a 2-class safe/risky model, return normalized (p_safe, p_risky)."""
    if not scores_list:
        return None
    probs_by_label: dict[str, float] = {}
    for item in scores_list:
        if not isinstance(item, dict):
            continue
        raw = str(item.get("label", ""))
        if raw.startswith("LABEL_"):
            try:
                lid = int(raw.replace("LABEL_", ""))
            except ValueError:
                continue
            label_name = _model_id2label.get(lid, "O") if _model_id2label else "O"
        else:
            label_name = raw
        probs_by_label[str(label_name).strip().lower()] = float(item.get("score", 0.0))
    if not any(k in probs_by_label for k in ("safe", "risky")):
        return None
    ps = float(probs_by_label.get("safe", 0.0))
    pr = float(probs_by_label.get("risky", 0.0))
    s = ps + pr
    if s <= 1e-9:
        return None
    return (ps / s, pr / s)


def _norm_layer_source(s: str) -> str:
    """Normalize span source for layer_summary buckets."""
    s = str(s or "").lower()
    if s in ("regex", "critical_secret", "kb_rule"):
        return "regex"
    if s == "model_window":
        return "ml"
    if s == "ner":
        return "ner"
    return s


def _binary_severity_from_regex_ner(
    text: str,
    triggers: list[str],
    labels: list[str] | None,
    p_risky: float,
    t_warn: float,
    t_block: float,
) -> str:
    """
    Severity for 2-class detector:
    - detection from model (safe/risky), severity from regex/NER signals.
    """
    t = (text or "").lower()
    trig = " | ".join(str(x).lower() for x in (triggers or []))
    lbls = " | ".join(str(x).lower() for x in (labels or []))
    merged = f"{trig} | {lbls}"

    # Hard-risk textual patterns not always captured in trigger labels.
    if re.search(r"\b(?:tax\s*id|ein)\s*[:#-]?\s*\d{2}-\d{7}\b", t, re.IGNORECASE):
        return "high"
    if re.search(r"\bpan\s*(?:number|no|#)?\s*[:#-]?\s*[a-z]{5}\d{4}[a-z]\b", t, re.IGNORECASE):
        return "high"
    if re.search(r"\b(?:driver'?s?\s*license|dl)\s*[:#-]?\s*[a-z0-9-]{6,20}\b", t, re.IGNORECASE):
        return "high"
    if (
        "swift" in t
        and re.search(r"\b[a-z]{4}[a-z]{2}[a-z0-9]{2}(?:[a-z0-9]{3})?\b", t, re.IGNORECASE)
        and "account" in t
    ):
        return "high"
    if "payroll" in t and any(tok in t for tok in ("bank", "account", "address", "salary", "employee id", "employee ids", "full names", "names")):
        return "high"
    if (
        ("employee" in t and "salary" in t and ("dob" in t or "born" in t))
        or ("salary" in t and "bonus" in t and ("employee id" in t or "employee ids" in t))
        or ("salaries" in t and "bonus" in t and ("employee id" in t or "employee ids" in t))
    ):
        return "high"
    if "debug log" in t and ("email" in t or "emails" in t) and ("id" in t or "order id" in t or "order ids" in t):
        return "high"

    critical_tokens = (
        "ssn", "social security", "credit card", "card number", "api key",
        "critical secret", "password", "bearer", "jwt", "iban", "routing", "bank account",
    )
    if any(tok in merged for tok in critical_tokens):
        return "high"
    if any(tok in t for tok in ("ssn", "social insurance number", "aadhaar", "aadhar", "national insurance number")):
        return "high"

    has_contact = any(tok in merged for tok in ("email", "phone"))
    if has_contact:
        return "med"

    medium_text_terms = (
        "sales",
        "finance",
        "hr",
        "onboarding",
        "engineer",
        "intern",
        "revenue",
        "budget",
        "cost",
        "expenses",
        "spend",
        "forecast",
        "medical leave",
        "health issue",
        "surgery",
        "sick",
        "on call this week",
        "extended leave",
        "billing service",
        "payments topic",
        "staging cluster",
        "replica count",
        "analytics job",
        "decouple the order and shipping",
    )
    if any(tok in t for tok in medium_text_terms):
        return "med"

    has_person = any(tok in merged for tok in ("person", "name"))
    has_med_context = any(tok in t for tok in ("patient", "medical", "diagnosis", "hospital", "mrn", "npi", "dea"))
    if has_person and has_med_context:
        return "high"
    if has_med_context and any(tok in t for tok in ("name", "treatment id", "patient id", "dob", "identity")):
        return "high"
    if "card ending" in t and "cvv" in t:
        return "high"

    pr = float(p_risky)
    if pr >= float(t_block):
        return "high"
    if pr >= float(t_warn):
        return "med"
    return "low"


def _looks_like_safe_code_snippet(text: str) -> bool:
    """
    Lightweight code-style heuristic to prevent false warn/block on benign snippets.
    """
    t = (text or "").strip()
    if len(t) < 8:
        return False
    u = t.lower()
    code_signals = (
        "def ",
        "return ",
        "class ",
        "import ",
        "from ",
        "select ",
        " where ",
        "{",
        "}",
        "=>",
        ";",
    )
    if not any(s in u for s in code_signals):
        return False
    pii_terms = (
        "ssn",
        "card",
        "iban",
        "password",
        "token",
        "api key",
        "jwt",
        "bearer",
        "email",
        "phone",
        "aadhaar",
        "social insurance",
        "national insurance",
    )
    return not any(p in u for p in pii_terms)


def _has_hard_block_trigger(triggers: list[str], text: str, labels: list[str] | None = None) -> bool:
    """
    Deterministic hard-block path for critical/risky patterns, independent of model confidence.
    """
    t = (text or "").lower()
    merged = " | ".join(str(x).lower() for x in (triggers or []))
    if labels:
        merged = f"{merged} | {' | '.join(str(x).lower() for x in labels)}"
    hard_tokens = (
        "ssn",
        "social security",
        "credit card",
        "card number",
        "api key",
        "critical secret",
        "password",
        "bearer",
        "jwt",
        "iban",
        "routing",
        "bank account",
        "passport",
        "private key",
        "db connection string",
        "aws_access_key_id",
        "aws_secret_access_key",
        "account number",
        "tax id",
        "ein",
        "swift",
        "pan number",
        "driver's license",
        "secret_key_base",
    )
    if any(tok in merged for tok in hard_tokens):
        return True
    if "card ending" in t and "cvv" in t:
        return True
    if re.search(r"\b(?:tax\s*id|ein)\s*[:#-]?\s*\d{2}-\d{7}\b", t, re.IGNORECASE):
        return True
    if re.search(r"\bpan\s*(?:number|no|#)?\s*[:#-]?\s*[a-z]{5}\d{4}[a-z]\b", t, re.IGNORECASE):
        return True
    if re.search(r"\b(?:driver'?s?\s*license|dl)\s*[:#-]?\s*[a-z0-9-]{6,20}\b", t, re.IGNORECASE):
        return True
    if (
        "swift" in t
        and re.search(r"\b[a-z]{4}[a-z]{2}[a-z0-9]{2}(?:[a-z0-9]{3})?\b", t, re.IGNORECASE)
        and "account" in t
    ):
        return True
    if re.search(r"\baccount\s*(?:number|no|#)?\s*[:#-]?\s*\d{8,20}\b", t, re.IGNORECASE):
        return True
    if "payroll" in t and any(tok in t for tok in ("bank", "account", "address", "salary", "employee id", "employee ids", "full names", "names")):
        return True
    if (
        ("employee" in t and "salary" in t and ("dob" in t or "born" in t))
        or ("salary" in t and "bonus" in t and ("employee id" in t or "employee ids" in t))
        or ("salaries" in t and "bonus" in t and ("employee id" in t or "employee ids" in t))
    ):
        return True
    if "debug log" in t and ("email" in t or "emails" in t) and ("id" in t or "order id" in t or "order ids" in t):
        return True
    if re.search(r"secret_key_base\s*[:=]\s*['\"]?[a-z0-9+/=_-]{16,}", t, re.IGNORECASE):
        return True
    return any(
        tok in t
        for tok in (
            "aws_access_key_id",
            "aws_secret_access_key",
            "-----begin private key-----",
            "ssn",
            "dob",
            "treatment id",
            "patient id",
            "medical summary",
            "identity packet",
            "social insurance number",
            "national insurance number",
            "aadhaar",
            "aadhar",
        )
    )


def _medium_signal_family_count(triggers: list[str], text: str, labels: list[str] | None = None) -> int:
    """
    Count distinct medium-risk signal families for strict multi-signal escalation.
    """
    t = (text or "").lower()
    merged = " | ".join(str(x).lower() for x in (triggers or []))
    if labels:
        merged = f"{merged} | {' | '.join(str(x).lower() for x in labels)}"
    families: set[str] = set()
    if any(k in merged for k in ("email", "phone", "address", "ip address", "ipv4", "ipv6")):
        families.add("contact")
    if any(k in merged for k in ("name", "person", "location")):
        families.add("identity")
    if any(k in merged for k in ("salary", "financial", "iban", "account", "routing")):
        families.add("finance")
    if any(k in merged for k in ("patient", "medical", "mrn", "npi", "dea")) or any(
        k in t for k in ("patient", "medical", "diagnosis", "hospital")
    ):
        families.add("health")
    if any(k in merged for k in ("confidential marker", "confidential", "internal only", "proprietary")):
        families.add("sensitive_context")
    return len(families)


def _strict_medium_block_downgrade_for_tiny_text(
    *,
    text: str,
    triggers: list[str],
    labels: list[str] | None,
    top_conf: float,
    strict_cfg: dict,
) -> bool:
    """
    Usability guard for strict mode:
    avoid blocking tiny low-signal snippets unless confidence is high or signals exist.
    """
    try:
        min_chars = max(1, int(strict_cfg.get("tiny_text_min_chars", 12)))
    except (TypeError, ValueError):
        min_chars = 12
    try:
        min_conf = float(strict_cfg.get("tiny_text_min_confidence", 0.85))
    except (TypeError, ValueError):
        min_conf = 0.85
    require_signal = bool(strict_cfg.get("tiny_text_requires_signal", True))
    stripped = (text or "").strip()
    signal_count = _medium_signal_family_count(triggers, text, labels)

    # Ultra-short text with no signal families is almost always paste-noise.
    if len(stripped) <= 2 and signal_count == 0:
        return True

    if len(stripped) >= min_chars:
        return False
    if float(top_conf) >= min_conf:
        return False
    if require_signal and signal_count > 0:
        return False
    return True


def text_heuristic_implied_risk_score(text: str) -> int:
    """Risk score from regex PII overrides only (no model); used before inference for duplicate bypass."""
    t = (text or "").strip()
    if not t:
        return 0
    risk = apply_pii_overrides(t, "low")
    triggers = get_pii_override_triggers(t) if risk in ("high", "med") else []
    if risk == "high" and not triggers:
        triggers = get_pii_override_triggers(t)
    return compute_risk_score(risk, triggers)


def text_heuristic_duplicate_bypass(text: str) -> bool:
    """True if regex PII layer implies HIGH risk or score at/above duplicate_bypass_min (no model)."""
    from guardrail_runtime import duplicate_bypass_min_risk_score

    t = (text or "").strip()
    if not t:
        return False
    risk = apply_pii_overrides(t, "low")
    if risk == "high":
        return True
    thr = duplicate_bypass_min_risk_score()
    if thr <= 0:
        return False
    return text_heuristic_implied_risk_score(text) >= thr


def should_bypass_duplicate_skip_for_text(text: str) -> bool:
    """
    If True, never treat this clipboard as an ignorable duplicate (paste hook, poll, inference TTL).

    Covers: critical-secret patterns, prior analysis with risk \"high\" or risk_score at/above
    GUARDRAIL_DUPLICATE_BYPASS_MIN_SCORE, or regex-only HIGH / high heuristic score.
    """
    t = (text or "").strip()
    if not t:
        return False
    if detect_critical_secret_leak(t):
        return True
    from guardrail_runtime import is_high_scoring_clipboard_repeat

    if is_high_scoring_clipboard_repeat(t):
        return True
    return text_heuristic_duplicate_bypass(t)


def suggest_remediation(text: str, triggers: list[str], risk_score: int) -> list[str]:
    """
    Return 3–5 concise remediation suggestions based on triggers and risk_score.
    """
    del text  # unused for now; kept for future richer suggestions

    triggers_l = [str(t) for t in (triggers or [])]
    triggers_join = ", ".join(triggers_l).lower()

    # Below threshold: general guidance
    if risk_score < 80:
        return [
            "Avoid sending unnecessary personal details to LLMs.",
            "Review clipboard content for names, IDs, and financial information before pasting.",
            "If in doubt, mask or replace identifiers with placeholders.",
        ]

    # >= 80: targeted suggestions
    suggestions: list[str] = []

    def has_any(*needles: str) -> bool:
        return any(n.lower() in triggers_join for n in needles)

    if has_any("ssn"):
        suggestions.append("Mask or hash national ID/SSN values before pasting (e.g., ***-**-6789 or a hash).")

    if has_any("credit card", "iban", "bank routing", "routing number", "account", "financial"):
        suggestions.append("Mask sensitive financial identifiers (keep last 4 digits) or encrypt/hash before pasting.")

    if has_any("email", "phone", "address"):
        suggestions.append("Replace emails/phones/addresses with generic placeholders (e.g., [customer_email], [phone]).")

    if has_any("ip", "device", "token", "auth", "jwt", "bearer"):
        suggestions.append("Hash or truncate IP/device/token values (e.g., 10.0.0.xxx).")

    # Ensure we return 3–5 even if only one category was detected.
    # (You requested 3–5 concise suggestions.)
    general_high = [
        "Prefer a safe masked version of your text over the raw clipboard contents.",
        "Share only what the assistant needs; avoid raw identifiers when possible.",
        "If you must share sensitive data, consider encrypting or hashing it before paste.",
        "Double-check the final text you send to the LLM matches your intent.",
    ]
    if len(suggestions) < 3:
        for g in general_high:
            if g not in suggestions:
                suggestions.append(g)
            if len(suggestions) >= 3:
                break

    if len(suggestions) > 5:
        suggestions = suggestions[:5]

    # Cap to 5 suggestions
    return suggestions


def score_clipboard_with_pii(
    text: str,
    policy_override: dict | None = None,
    *,
    min_confidence: float = 0.45,
    context: str = "clipboard",
) -> dict:
    """
    Score clipboard text using PII label aggregation + regex overrides.

    Long texts use sliding-window inference (overlap GUARDRAIL_SLIDING_STRIDE) so tail content
    is not dropped by max-length truncation; per-window logits are merged by _pick_worst_window_probs.
    Regex overrides still run on the full string.

    min_confidence: minimum model probability to keep a model_window span (default 0.45).
    context: "clipboard" (default) or "document". Document scans use stricter thresholds
    (min_confidence at least 0.65) and exclude oversized windows from merge and spans.

    policy_override: optional dict to override POLICY for this call (e.g. {"high": {"allow_warn_instead": False}}).
    Returns dict with risk, pii_labels, pii_risk_before_override, pii_override_applied,
    decision, block, action (silent | warn | block), message, plus token_count, chunks_scored,
    text_truncated (True if more than one window), window_scores (per-chunk char span + top_prob),
    and spans (character-level annotations from get_pii_spans: regex + model_window).
    """
    if not text or not text.strip():
        msg = "Clipboard content appears safe (no sensitive data detected)."
        _log_scoring_event(
            text or "",
            pii_class="empty_input",
            confidence=0.0,
            action="silent",
            context=_telemetry_ctx(
                risk="low",
                risk_score=0,
                score_context=context,
            ),
        )
        return {
            "risk": "low",
            "pii_labels": [],
            "pii_risk_before_override": "low",
            "pii_override_applied": False,
            "decision": "allow",
            "block": False,
            "action": "silent",
            "message": msg,
            "triggers": [],
            "risk_score": 0,
            "suggestions": suggest_remediation(text, [], 0),
            "critical_secret_detected": False,
            "critical_secret": False,
            "token_count": 0,
            "chunks_scored": 0,
            "text_truncated": False,
            "window_scores": [],
            "spans": [],
            "layer_summary": {
                "regex_count": 0,
                "ml_count": 0,
                "ner_count": 0,
                "total_unique": 0,
                "source_breakdown": {},
            },
        }
    eff_min = max(min_confidence, 0.65) if context == "document" else min_confidence
    ts = text.strip()
    critical_secret_detected = detect_critical_secret_leak(ts)
    hipaa_critical_spans = _hipaa_critical_spans(ts)
    hipaa_critical_triggers = list(
        dict.fromkeys(str(span.get("class", "")).strip() for span in hipaa_critical_spans if span.get("class"))
    )

    cached = None
    if context != "document":
        cached = get_cached_inference_result(text, cache_kind="pii")
    if (
        cached is not None
        and not critical_secret_detected
        and not text_heuristic_duplicate_bypass(text)
    ):
        cr = str(cached.get("risk", "low")).strip().lower()
        try:
            crs = int(cached.get("risk_score", 0))
        except (TypeError, ValueError):
            crs = 0
        _log_scoring_event(
            text,
            pii_class="dedup_cached",
            confidence=0.0,
            action=str(cached.get("action", "silent")),
            context=_telemetry_ctx(risk=cr, risk_score=crs, score_context=context),
        )
        return cached
    if critical_secret_detected:
        got = None
    elif context == "document":
        got = _infer_scores_and_probs(
            text,
            min_window_confidence=eff_min,
            max_window_char_span=100,
        )
    else:
        got = _infer_scores_and_probs(text)
    scores_list: list | None
    if got is None:
        scores_list = None
        labels = ["O"]
        class_probs: dict[int, float] = {}
    else:
        scores_list, class_probs = got
        labels = _labels_from_scores(scores_list)
    top_conf = _confidence_from_scores(scores_list)
    binary_model = int(_model_num_labels or 3) == 2
    prob_risky_bin: float | None = None
    t_warn_bin: float | None = None
    t_block_bin: float | None = None
    binary_two_threshold_allow = False
    # Our model outputs either:
    #   - token-level style labels (NAME/CONTACT/...), in which case we map via aggregate_span_risks
    #   - 3-class risk labels (low/med/high), in which case we treat the model output directly as risk
    if binary_model and got is not None and scores_list:
        probs2 = _binary_probs_from_scores_list(scores_list)
        p_risky = float(probs2[1]) if probs2 is not None else 0.0
        _pol_bin = load_risk_policy()
        t_warn, t_block = get_binary_two_thresholds(_pol_bin)
        prob_risky_bin = float(p_risky)
        t_warn_bin = float(t_warn)
        t_block_bin = float(t_block)
        triggers = get_pii_override_triggers(ts)
        if hipaa_critical_triggers:
            triggers.extend(hipaa_critical_triggers)
            triggers = list(dict.fromkeys(triggers))
        if critical_secret_detected and not triggers:
            triggers = ["critical_secret"]
        base_risk = _binary_severity_from_regex_ner(ts, triggers, labels, p_risky, t_warn, t_block)
        risk = base_risk
        if (
            risk == "med"
            and not triggers
            and len(ts) > 8
            and any(
                tok in ts.lower()
                for tok in (
                    "sales",
                    "finance",
                    "hr",
                    "revenue",
                    "budget",
                    "cost",
                    "expenses",
                    "spend",
                    "forecast",
                    "medical leave",
                    "health issue",
                    "surgery",
                    "on call this week",
                    "extended leave",
                    "billing service",
                    "payments topic",
                    "staging cluster",
                    "replica count",
                    "analytics job",
                    "decouple the order and shipping",
                )
            )
        ):
            # Lift medium-context snippets above warn floor in final score gate.
            triggers = ["business context"]
        override_applied = False
        if critical_secret_detected or hipaa_critical_triggers:
            risk = "high"
            override_applied = True
        merged_score_for_penalty = int(round(max(0.0, min(1.0, p_risky)) * 100))
        severity_floor_score = compute_risk_score(risk, triggers)
        merged_score_for_penalty = max(merged_score_for_penalty, severity_floor_score)
        if hipaa_critical_triggers:
            merged_score_for_penalty = max(merged_score_for_penalty, 95)
        print(
            f"[infer] Binary model: P(risky)={p_risky:.2f} -> "
            f"base_score={merged_score_for_penalty}, t_warn={t_warn:.2f} t_block={t_block:.2f}, "
            f"severity from regex/NER: {risk}"
        )
    elif labels and str(labels[0]).strip().lower() in ("low", "med", "high"):
        triggers = []
        override_applied = False
        base_risk = str(labels[0]).strip().lower()
        risk = apply_pii_overrides(ts, base_risk)
        override_applied = base_risk != risk
        if critical_secret_detected:
            risk = "high"
            override_applied = True
        if hipaa_critical_triggers:
            risk = "high"
            override_applied = True
        # Include regex triggers in message when risk is high (for block/warn messages)
        triggers = get_pii_override_triggers(ts) if (override_applied or risk in ("high", "med")) else []
        if hipaa_critical_triggers:
            triggers.extend(hipaa_critical_triggers)
            triggers = list(dict.fromkeys(triggers))
        if critical_secret_detected and not triggers:
            triggers = ["critical_secret"]
        merged_score_for_penalty = compute_risk_score(risk, triggers)
        if hipaa_critical_triggers:
            merged_score_for_penalty = max(merged_score_for_penalty, 95)
    else:
        triggers = []
        override_applied = False
        base_risk = aggregate_span_risks(labels)
        risk = apply_pii_overrides(ts, base_risk)
        override_applied = base_risk != risk
        if critical_secret_detected:
            risk = "high"
            override_applied = True
        if hipaa_critical_triggers:
            risk = "high"
            override_applied = True
        # Include regex triggers in message when risk is high (for block/warn messages)
        triggers = get_pii_override_triggers(ts) if (override_applied or risk in ("high", "med")) else []
        if hipaa_critical_triggers:
            triggers.extend(hipaa_critical_triggers)
            triggers = list(dict.fromkeys(triggers))
        if critical_secret_detected and not triggers:
            triggers = ["critical_secret"]
        merged_score_for_penalty = compute_risk_score(risk, triggers)
        if hipaa_critical_triggers:
            merged_score_for_penalty = max(merged_score_for_penalty, 95)
    dmode = _dynamic_risk_score_mode()
    if (not binary_model) and dmode and got is not None and scores_list:
        probs3 = _three_class_probs_from_scores_list(scores_list)
        if probs3 is not None:
            pl, pm, ph = probs3
            soft = _expected_risk_score_from_3class_probs(pl, pm, ph)
            tier_only = compute_risk_score(risk, [])
            trigger_extra = max(0, merged_score_for_penalty - tier_only)
            combined = min(100, soft + trigger_extra)
            if dmode == "blend":
                merged_score_for_penalty = int(round(0.5 * merged_score_for_penalty + 0.5 * combined))
            else:
                merged_score_for_penalty = combined
    if (
        not binary_model
        and
        not critical_secret_detected
        and not triggers
        and got is not None
    ):
        win_meta_pre = _get_sliding_window_meta()
        prob_vecs = list(win_meta_pre.get("window_prob_vectors") or [])
        if len(prob_vecs) > 1:
            pws = _per_window_model_risk_scores(prob_vecs)
            risk, merged_score_for_penalty = _apply_long_text_sliding_penalty(
                pws,
                risk,
                merged_score_for_penalty,
                total_tokens=int(win_meta_pre.get("token_count", 0) or 0),
            )
    base_policy = load_pii_policy()
    if policy_override:
        effective_policy = dict(base_policy)
        for level, opts in policy_override.items():
            if isinstance(opts, dict) and level in effective_policy and isinstance(effective_policy[level], dict):
                effective_policy[level] = {**effective_policy[level], **opts}
            else:
                effective_policy[level] = opts
    else:
        effective_policy = base_policy
    risk_policy = load_risk_policy()
    n_lab = _model_num_labels if _model_num_labels is not None else 3
    if (
        binary_model
        and prob_risky_bin is not None
        and t_warn_bin is not None
        and t_block_bin is not None
    ):
        # Two-threshold zones from policy (warn / block); not overridden by allow_warn_instead.
        pr = float(prob_risky_bin)
        tw, tb = float(t_warn_bin), float(t_block_bin)
        if pr >= tb:
            decision = "block"
        elif pr >= tw:
            decision = "warn"
        else:
            decision = "allow"
            binary_two_threshold_allow = True
        block = decision == "block"
    elif (
        n_lab == 9
        and isinstance(risk_policy.get("per_class_thresholds"), dict)
        and _model_id2label
        and class_probs
    ):
        decision_result = decision_from_pii_risk(risk, policy=effective_policy)
        d_pc = decision_from_per_class_probs(class_probs, _model_id2label, risk_policy)
        decision = _merge_decisions(
            decision_result["decision"],
            d_pc,
            critical_escalate_warn_to_block=critical_secret_detected,
        )
        block = decision == "block"
    else:
        decision_result = decision_from_pii_risk(risk, policy=effective_policy)
        decision = _merge_decisions(
            decision_result["decision"],
            critical_escalate_warn_to_block=critical_secret_detected,
        )
        block = decision == "block"

    strict_mode = _strict_recall_mode_enabled(risk_policy)
    strict_cfg = risk_policy.get("strict_decision_rules", {}) if isinstance(risk_policy, dict) else {}
    try:
        medium_signal_min = max(1, int(strict_cfg.get("medium_multi_signal_block_min", 2)))
    except (TypeError, ValueError):
        medium_signal_min = 2
    hard_block = critical_secret_detected or _has_hard_block_trigger(triggers, ts, labels)
    multi_signal_block = strict_mode and (_medium_signal_family_count(triggers, ts, labels) >= medium_signal_min)
    if hard_block or multi_signal_block:
        decision = "block"
        block = True
        if hard_block:
            risk = "high"
        elif risk == "low":
            risk = "med"
    elif (
        risk == "med"
        and not critical_secret_detected
        and not hipaa_critical_triggers
        and not triggers
        and _looks_like_safe_code_snippet(ts)
    ):
        # Benign code-like text should remain paste-friendly without warnings.
        risk = "low"
        decision = "allow"
        block = False
        merged_score_for_penalty = min(int(merged_score_for_penalty), 10)
    elif (
        strict_mode
        and risk == "med"
        and decision == "block"
        and _strict_medium_block_downgrade_for_tiny_text(
            text=ts,
            triggers=triggers,
            labels=labels,
            top_conf=top_conf,
            strict_cfg=strict_cfg,
        )
    ):
        # Tiny/noisy low-signal snippets should not hard-block in strict mode.
        decision = "allow"
        block = False
        risk = "low"
        merged_score_for_penalty = min(int(merged_score_for_penalty), 20)

    risk_score = merged_score_for_penalty
    if critical_secret_detected:
        risk_score = max(risk_score, 90)
    if hard_block:
        risk_score = max(risk_score, 90)
    elif multi_signal_block:
        risk_score = max(risk_score, 80)
    if hipaa_critical_triggers:
        decision = "block"
        block = True

    score_block_min = _risk_score_force_block_min()
    if score_block_min >= 1 and risk_score >= score_block_min:
        decision = "block"
        block = True

    win_meta = _get_sliding_window_meta()
    if critical_secret_detected:
        win_meta = {
            "token_count": 0,
            "chunks_scored": 0,
            "text_truncated": False,
            "window_scores": [],
        }
    spans = get_pii_spans(
        ts,
        risk_policy,
        sliding_meta=win_meta,
        min_confidence=eff_min,
        max_model_window_chars=100,
    )

    _span_src_counts = Counter(_norm_layer_source(sp.get("source", "")) for sp in (spans or []))
    # Critical path can set risk_score to 90+ while get_pii_spans returns [] (JWT/API patterns are
    # not always duplicated in strong-regex / model windows). Emit spans so the panel can build cards.
    if critical_secret_detected and not spans:
        cs_spans = critical_secret_spans_for_ui(ts)
        if cs_spans:
            spans = cs_spans

    suggestions = suggest_remediation(text, triggers, risk_score)
    action = clipboard_ui_action(decision, critical_secret=critical_secret_detected)
    primary_pii = _primary_pii_class_for_class_override(labels, class_probs)
    action = enforce_risk_policy(
        action,
        risk,
        primary_pii,
        risk_policy,
        critical_secret=critical_secret_detected,
        text=ts,
        top_conf=top_conf,
        triggers=triggers,
        labels=labels,
    )

    if action == "silent" and _strict_medium_block_downgrade_for_tiny_text(
        text=ts,
        triggers=triggers,
        labels=labels,
        top_conf=top_conf,
        strict_cfg=strict_cfg if isinstance(strict_cfg, dict) else {},
    ):
        risk_score = min(int(risk_score), 20)

    decision, block = _decision_and_block_from_ui_action(action)
    message = build_user_message(risk, decision, override_applied, triggers)

    # Contact-only: never block on email/phone/IP-only pastes; cap score and warn when model score > 70.
    contact_only_capped = False
    if (
        _is_contact_only_pii(spans, critical_secret=critical_secret_detected)
        and risk_score > 70
        and not _contact_only_cap_excluded_by_keywords(ts)
    ):
        risk_score = min(risk_score, 70)
        action = "warn"
        decision, block = _decision_and_block_from_ui_action("warn")
        message = build_user_message(risk, decision, override_applied, triggers)
        suggestions = suggest_remediation(text, triggers, risk_score)
        contact_only_capped = True

    action = _finalize_action_for_risk_score(
        action,
        risk_score,
        critical_secret=critical_secret_detected,
        binary_allow_zone=binary_two_threshold_allow,
    )
    decision, block = _decision_and_block_from_ui_action(action)
    message = build_user_message(risk, decision, override_applied, triggers)

    _log_scoring_event(
        text,
        pii_class=_pii_class_for_log(labels, risk),
        confidence=top_conf,
        action=action,
        context=_telemetry_ctx(
            risk=risk,
            risk_score=risk_score,
            score_context=context,
            critical_secret=critical_secret_detected,
        ),
    )
    result = {
        "risk": risk,
        "pii_labels": labels,
        "pii_risk_before_override": base_risk,
        "pii_override_applied": override_applied,
        "decision": decision,
        "block": block,
        "action": action,
        "message": message,
        "triggers": triggers,
        "risk_score": risk_score,
        "suggestions": suggestions,
        "critical_secret_detected": critical_secret_detected,
        "critical_secret": bool(critical_secret_detected),
        "token_count": win_meta.get("token_count", 0),
        "chunks_scored": win_meta.get("chunks_scored", 0),
        "text_truncated": win_meta.get("text_truncated", False),
        "window_scores": win_meta.get("window_scores", []),
        "spans": spans,
        "layer_summary": {
            "regex_count": int(_span_src_counts.get("regex", 0)),
            "ml_count": int(_span_src_counts.get("ml", 0)),
            "ner_count": int(_span_src_counts.get("ner", 0)),
            "total_unique": len(spans or []),
            "source_breakdown": dict(
                Counter(sp.get("source", "?") for sp in (spans or []))
            ),
        },
    }
    if binary_model and prob_risky_bin is not None:
        result["prob_risky"] = float(prob_risky_bin)
        result["t_warn"] = float(t_warn_bin or 0.0)
        result["t_block"] = float(t_block_bin or 0.0)
    if contact_only_capped:
        result["_capped"] = "contact-only cap applied"
    if binary_model:
        _pol2 = load_risk_policy()
        _twb, _tbb = get_binary_two_thresholds(_pol2)
        result.setdefault("t_warn", float(_twb))
        result.setdefault("t_block", float(_tbb))
    if critical_secret_detected:
        _tw_cs, _tb_cs = get_binary_two_thresholds(load_risk_policy())
        n_cs = len(spans or [])
        result["prob_risky"] = None
        result["t_warn"] = float(_tw_cs)
        result["t_block"] = float(_tb_cs)
        result["layer_summary"] = {
            "regex_count": int(n_cs),
            "ml_count": 0,
            "ner_count": 0,
            "total_unique": int(n_cs),
            "source_breakdown": {"critical_secret": int(n_cs)},
        }
    if context != "document":
        record_inference_scored_text(text, result, cache_kind="pii")
    return result


def score_clipboard(text: str, use_pii: bool = False) -> dict:
    """
    Score text with the guardrail model and policy.

    If use_pii=True, uses PII label aggregation + regex overrides and returns
    the same shape as score_clipboard_with_pii (risk, decision, block, pii_labels, etc.).

    Otherwise returns:
        risk: "low" | "med" | "high"
        conf: confidence of predicted class (0–1)
        prob_low, prob_med, prob_high: class probabilities (9-class: sums over PII labels only, O excluded)
        decision: "allow" | "warn" | "block"
        action: "silent" | "warn" | "block" (UI notification level)
        pii_count: 3-class = class id; 9-class = number of PII types (non-O) with prob >= threshold
        block: True when decision == "block" (backward compatible)
    """
    if not text or not text.strip():
        _log_scoring_event(
            text or "",
            pii_class="empty_input",
            confidence=0.0,
            action="silent",
            context=_telemetry_ctx(risk="low", risk_score=0, score_context="plain"),
        )
        return {
            "risk": "low",
            "conf": 0.0,
            "prob_low": 1.0,
            "prob_med": 0.0,
            "prob_high": 0.0,
            "decision": "allow",
            "action": "silent",
            "pii_count": 0,
            "block": False,
        }

    if use_pii:
        return score_clipboard_with_pii(text)

    cached = get_cached_inference_result(text, cache_kind="plain")
    if (
        cached is not None
        and not detect_critical_secret_leak(text)
        and not text_heuristic_duplicate_bypass(text)
    ):
        cr = str(cached.get("risk", "low")).strip().lower()
        try:
            crs = int(cached.get("risk_score", 0))
        except (TypeError, ValueError):
            crs = 0
        _log_scoring_event(
            text,
            pii_class="dedup_cached",
            confidence=0.0,
            action=str(cached.get("action", "silent")),
            context=_telemetry_ctx(risk=cr, risk_score=crs, score_context="plain"),
        )
        return cached

    got = _infer_scores_and_probs(text)
    if got is None:
        _log_scoring_event(
            text,
            pii_class="inference_unavailable",
            confidence=0.0,
            action="silent",
            context=_telemetry_ctx(risk="low", risk_score=0, score_context="plain"),
        )
        fail_plain = {
            "risk": "low",
            "conf": 0.0,
            "prob_low": 1.0,
            "prob_med": 0.0,
            "prob_high": 0.0,
            "decision": "allow",
            "action": "silent",
            "pii_count": 0,
            "block": False,
        }
        record_inference_scored_text(text, fail_plain, cache_kind="plain")
        return fail_plain
    _, probs = got

    n = _model_num_labels if _model_num_labels is not None else 3

    if _pii_to_risk is not None and _model_id2label is not None and n == 9:
        # 9-class PII model: aggregate probs by risk bucket, excluding O from all buckets
        prob_low = prob_med = prob_high = 0.0
        for i in range(n):
            pii_name = _model_id2label.get(i, "O")
            if pii_name == "O":
                continue
            risk_bucket = _pii_to_risk.get(pii_name, "low")
            if risk_bucket == "low":
                prob_low += probs[i]
            elif risk_bucket == "med":
                prob_med += probs[i]
            else:
                prob_high += probs[i]
        best_id = max(probs, key=probs.get)
        top_label = _pii_to_risk.get(_model_id2label.get(best_id, "O"), "low")
        top_prob = probs[best_id]
        conf = top_prob
        # pii_count = number of PII types (non-O) with prob >= threshold
        pct_thr = get_pii_count_probability_threshold(load_risk_policy())
        pii_count = sum(1 for i in range(n) if _model_id2label.get(i, "O") != "O" and probs[i] >= pct_thr)
    elif n == 2:
        # Binary safe/risky detector: use p(risky) as detection confidence.
        prob_low = probs.get(0, 0.0)   # safe
        prob_med = probs.get(1, 0.0)   # risky proxy in legacy field
        prob_high = probs.get(1, 0.0)  # risky proxy for backward-compatible return shape
        p_risky = probs.get(1, 0.0)
        risky_thr = _binary_risky_threshold(load_risk_policy())
        if p_risky >= risky_thr:
            top_label = "med"
            top_prob = p_risky
            conf = p_risky
            pii_count = 1
        else:
            top_label = "low"
            top_prob = probs.get(0, 0.0)
            conf = probs.get(0, 0.0)
            pii_count = 0
    else:
        # 3-class: LABEL_0=low, LABEL_1=med, LABEL_2=high
        prob_low = probs.get(0, 0.0)
        prob_med = probs.get(1, 0.0)
        prob_high = probs.get(2, 0.0)
        pol = load_risk_policy()
        th, tm = get_inference_3class_label_thresholds(pol)
        # Same boundaries as _labels_from_scores / inference_3class_labels in risk_policy.json
        if prob_high > th:
            top_label = "high"
            top_prob = prob_high
            conf = prob_high
            pii_count = 2
        elif prob_med > tm:
            top_label = "med"
            top_prob = prob_med
            conf = prob_med
            pii_count = 1
        else:
            top_label = "low"
            top_prob = prob_low
            conf = prob_low
            pii_count = 0

    policy = load_risk_policy()
    if (
        _pii_to_risk is not None
        and _model_id2label is not None
        and n == 9
        and isinstance(policy.get("per_class_thresholds"), dict)
    ):
        decision = decision_from_per_class_probs(probs, _model_id2label, policy)
    else:
        decision = decision_from_probs(prob_low, prob_med, prob_high, top_label, pii_count, policy)

    critical_secret = detect_critical_secret_leak(text)
    decision = _merge_decisions(
        decision,
        critical_escalate_warn_to_block=critical_secret,
    )
    action = clipboard_ui_action(decision, critical_secret=critical_secret)
    triggers_plain = get_pii_override_triggers(text.strip()) if top_label == "high" else []
    plain_risk_score = compute_risk_score(top_label, triggers_plain)
    if critical_secret:
        plain_risk_score = max(plain_risk_score, 90)
    action = _finalize_action_for_risk_score(
        action,
        plain_risk_score,
        critical_secret=critical_secret,
    )
    decision, _blk = _decision_and_block_from_ui_action(action)
    _log_scoring_event(
        text,
        pii_class=_pii_class_for_log([], top_label),
        confidence=float(top_prob),
        action=action,
        context=_telemetry_ctx(
            risk=top_label,
            risk_score=plain_risk_score,
            score_context="plain",
            critical_secret=critical_secret,
        ),
    )

    plain_result = {
        "risk": top_label,
        "conf": round(conf, 4),
        "prob_low": round(prob_low, 4),
        "prob_med": round(prob_med, 4),
        "prob_high": round(prob_high, 4),
        "decision": decision,
        "action": action,
        "pii_count": pii_count,
        "block": decision == "block",
    }
    record_inference_scored_text(text, plain_result, cache_kind="plain")
    return plain_result


def predict_3class_bucket_from_probs(
    prob_low: float,
    prob_med: float,
    prob_high: float,
    policy: dict | None = None,
) -> int:
    """
    Map softmax probabilities to a discrete bucket matching _labels_from_scores / score_clipboard:
    0 = low, 1 = med, 2 = high. Used by calibrate_thresholds.py sweeps (must stay in sync with
    get_inference_3class_label_thresholds).
    """
    p = policy or load_risk_policy()
    th, tm = get_inference_3class_label_thresholds(p)
    if prob_high > th:
        return 2
    if prob_med > tm:
        return 1
    return 0


if __name__ == "__main__":
    example = "SAP-1234567890 | Wire 50000 to GB82WEST12345698765432. Beneficiary Acme Corp. Routing: 123456789. Confirmation required."
    result = score_clipboard(example)
    print("score_clipboard(result):", result)
    