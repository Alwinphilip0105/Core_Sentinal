"""
Benchmark 3 PHI/PII detection configurations with project-native defaults.

Configurations:
  1) Regex + NER only
  2) ML only at two probability cutoffs (see load_thresholds / column legend)
  3) Full pipeline: (Regex-or-NER match) OR ML at the same two cutoffs

The two ML cutoffs are documented in the report footer:
  • val-calib — from model_records.json → calibration.prob_threshold_risky
    (validation-split sweep; see calibrate_thresholds.py --validation-sweep).
  • test-rec — from model_records.json → metrics.recommended_threshold
    (held-out test / PR-curve style recommendation, e.g. binary_threshold_check).

Primary input format (annotated JSONL): one record per line:
{
  "text": "raw text ...",
  "spans": [{"start": 10, "end": 22, "type": "EMAIL"}, ...]
}

Project-native fallback:
- If --dataset is omitted, uses `arrow_datasets/test` from this repo and evaluates one
  candidate per row (decoded text). This gives complete binary metrics and TP/FP/TN/FN.

Notes:
- Annotated mode evaluates candidate spans (positive PHI spans + sampled non-PHI negatives).
- HIPAA categories are used as a label taxonomy for PII span definitions.
- This is a detection benchmark, not a HIPAA compliance certification tool.
- Per-category metrics are computed as one-vs-none within each category subset:
  positives: true_type == category
  negatives: true_type == NONE
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from risk_mapping import strong_regex_pii_spans  # noqa: E402

HIPAA_TYPES = [
    "NAME",
    "DATE",
    "MRN",
    "SSN",
    "PHONE",
    "EMAIL",
    "AGE",
    "LOCATION",
    "HEALTHPLAN",
    "ACCOUNT",
    "LICENSE",
    "VEHICLE",
    "DEVICE",
    "URL",
    "IP",
    "BIOMETRIC",
    "PHOTO",
    "FAX",
]

PII_TYPES = [
    "NAME",
    "CONTACT",
    "LOCATION",
    "ID",
    "FINANCIAL",
    "HEALTH",
    "AUTH",
    "OTHER_PII",
]

NONE_TYPE = "NONE"


@dataclass
class Candidate:
    text_id: int
    start: int
    end: int
    text: str
    true_type: str
    is_phi: int


@dataclass
class TextRecord:
    text: str
    candidates: list[Candidate]


def _safe_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def load_thresholds(model_records_path: Path) -> tuple[float, float]:
    """
    Return (threshold_validation_calib, threshold_test_recommended).

    - validation calib: calibration.prob_threshold_risky (validation-set sweep report).
    - test recommended: metrics.recommended_threshold (hold-out / PR-style pick on test metrics).
    """
    if not model_records_path.is_file():
        raise FileNotFoundError(f"model_records not found: {model_records_path}")
    with open(model_records_path, encoding="utf-8") as f:
        data = json.load(f)

    calibrated = _safe_float(
        ((data.get("calibration") or {}).get("prob_threshold_risky")),
        default=0.5,
    )
    holdout = _safe_float(
        ((data.get("metrics") or {}).get("recommended_threshold")),
        default=0.45,
    )
    return max(0.01, min(0.99, calibrated)), max(0.01, min(0.99, holdout))


def benchmark_column_names(thr_validation_calib: float, thr_test_recommended: float) -> dict[str, str]:
    """Stable config keys + human-readable labels for benchmark tables and plots."""
    tv, tt = float(thr_validation_calib), float(thr_test_recommended)
    return {
        "regex_ner": "Regex+NER",
        "ml_val_calib": f"ML val-calib t={tv:.3f}",
        "ml_test_rec": f"ML test-rec t={tt:.3f}",
        "full_val_calib": f"Full val-calib t={tv:.3f}",
        "full_test_rec": f"Full test-rec t={tt:.3f}",
    }


def benchmark_threshold_legend_text() -> str:
    return (
        "Column legend (t = P(risky) cutoff on the risky class for the binary classifier):\n"
        "  val-calib — validation-split threshold from model_records.calibration.prob_threshold_risky "
        "(reports/threshold_calibration.json; see calibrate_thresholds.py --validation-sweep).\n"
        "  test-rec  — threshold from model_records.metrics.recommended_threshold "
        "(held-out test / PR-curve style evaluation, e.g. reports/binary_threshold_check.json).\n"
        '  Full …    — (Regex match OR NER entity) OR (ML score ≥ t). "ML …" uses only the classifier.'
    )


def _norm_phi_type(raw: Any) -> str:
    s = str(raw or "").strip().upper().replace("-", "").replace("_", "")
    direct = {
        # Generic project PII labels
        "CONTACT": "CONTACT",
        "ID": "ID",
        "FINANCIAL": "FINANCIAL",
        "HEALTH": "HEALTH",
        "AUTH": "AUTH",
        "OTHERPII": "OTHER_PII",
        # HIPAA-style labels
        "NAME": "NAME",
        "PERSON": "NAME",
        "DATE": "DATE",
        "DOB": "DATE",
        "MRN": "MRN",
        "MEDICALRECORDNUMBER": "MRN",
        "SSN": "SSN",
        "SOCIALSECURITY": "SSN",
        "PHONE": "PHONE",
        "PHONENUMBER": "PHONE",
        "TELEPHONE": "PHONE",
        "EMAIL": "EMAIL",
        "EMAILADDRESS": "EMAIL",
        "AGE": "AGE",
        "LOCATION": "LOCATION",
        "ADDRESS": "LOCATION",
        "GPE": "LOCATION",
        "LOC": "LOCATION",
        "HEALTHPLAN": "HEALTHPLAN",
        "INSURANCE": "HEALTHPLAN",
        "ACCOUNT": "ACCOUNT",
        "ACCOUNTNUMBER": "ACCOUNT",
        "LICENSE": "LICENSE",
        "DEA": "LICENSE",
        "NPI": "LICENSE",
        "VEHICLE": "VEHICLE",
        "DEVICE": "DEVICE",
        "SERIALNUMBER": "DEVICE",
        "URL": "URL",
        "URI": "URL",
        "IP": "IP",
        "IPADDRESS": "IP",
        "BIOMETRIC": "BIOMETRIC",
        "FINGERPRINT": "BIOMETRIC",
        "PHOTO": "PHOTO",
        "IMAGE": "PHOTO",
        "FAX": "FAX",
    }
    if s in direct:
        return direct[s]
    return NONE_TYPE


def _extract_spans(record: dict) -> list[dict]:
    for key in ("spans", "entities", "annotations", "phi_spans"):
        val = record.get(key)
        if isinstance(val, list):
            return [x for x in val if isinstance(x, dict)]
    return []


def _extract_text(record: dict) -> str:
    for key in ("text", "content", "input", "prompt", "sentence"):
        if key in record and record[key] is not None:
            t = str(record[key])
            if t.strip():
                return t
    return ""


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for m in re.finditer(r"[^.!?\n]+[.!?\n]?", text):
        s, e = m.start(), m.end()
        if e > s and text[s:e].strip():
            spans.append((s, e))
    if not spans and text.strip():
        spans = [(0, len(text))]
    return spans


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return not (a_end <= b_start or a_start >= b_end)


def load_annotated_records(
    dataset_path: Path,
    *,
    negatives_per_positive: int,
    seed: int,
    max_records: int | None,
) -> list[TextRecord]:
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    rng = random.Random(seed)
    records: list[TextRecord] = []
    with open(dataset_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_records is not None and len(records) >= max_records:
                break
            raw = line.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            text = _extract_text(obj)
            if not text:
                continue
            raw_spans = _extract_spans(obj)
            pos_candidates: list[Candidate] = []
            used_pos: list[tuple[int, int]] = []
            for sp in raw_spans:
                start = int(sp.get("start", -1))
                end = int(sp.get("end", -1))
                if start < 0 or end <= start or end > len(text):
                    continue
                phi = _norm_phi_type(sp.get("type") or sp.get("label") or sp.get("class"))
                if phi == NONE_TYPE:
                    continue
                used_pos.append((start, end))
                pos_candidates.append(
                    Candidate(
                        text_id=len(records),
                        start=start,
                        end=end,
                        text=text[start:end],
                        true_type=phi,
                        is_phi=1,
                    )
                )

            neg_needed = max(1, negatives_per_positive * max(1, len(pos_candidates)))
            sentence_pool = []
            for s, e in _sentence_spans(text):
                if any(_overlaps(s, e, ps, pe) for ps, pe in used_pos):
                    continue
                seg = text[s:e].strip()
                if len(seg) < 3:
                    continue
                sentence_pool.append((s, e))

            rng.shuffle(sentence_pool)
            neg_candidates: list[Candidate] = []
            for s, e in sentence_pool[:neg_needed]:
                neg_candidates.append(
                    Candidate(
                        text_id=len(records),
                        start=s,
                        end=e,
                        text=text[s:e],
                        true_type=NONE_TYPE,
                        is_phi=0,
                    )
                )

            all_candidates = pos_candidates + neg_candidates
            if not all_candidates:
                continue
            records.append(TextRecord(text=text, candidates=all_candidates))
    return records


def load_project_arrow_records(
    arrow_test_dir: Path,
    tokenizer_dir: Path,
    *,
    max_records: int | None,
) -> list[TextRecord]:
    if not arrow_test_dir.exists():
        raise FileNotFoundError(f"Project test split not found: {arrow_test_dir}")
    ds = load_from_disk(str(arrow_test_dir))
    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir.resolve()))
    records: list[TextRecord] = []
    for idx, row in enumerate(ds):
        if max_records is not None and len(records) >= max_records:
            break
        input_ids = row.get("input_ids")
        text = ""
        if isinstance(input_ids, list) and input_ids:
            text = tok.decode(input_ids, skip_special_tokens=True).strip()
        if not text:
            continue
        raw_risk = str(row.get("risk", "")).strip().lower()
        raw_label = row.get("label")
        is_phi = 1 if raw_risk in ("risky", "high", "med") else 0
        if not raw_risk:
            try:
                is_phi = 1 if int(raw_label) > 0 else 0
            except (TypeError, ValueError):
                is_phi = 0
        candidate = Candidate(
            text_id=idx,
            start=0,
            end=len(text),
            text=text,
            true_type="ACCOUNT" if is_phi else NONE_TYPE,
            is_phi=is_phi,
        )
        records.append(TextRecord(text=text, candidates=[candidate]))
    return records


def _regex_type_for_class(name: str) -> str:
    n = (name or "").lower()
    if "ssn" in n or "insurance number" in n or "aadhaar" in n:
        return "SSN"
    if "email" in n:
        return "CONTACT"
    if "phone" in n:
        return "CONTACT"
    if "passport" in n:
        return "ID"
    if "routing" in n or "iban" in n or "card" in n:
        return "FINANCIAL"
    if "ip" in n:
        return "CONTACT"
    return "ID"


def detect_regex_spans(text: str) -> list[dict]:
    out = []
    for sp in strong_regex_pii_spans(text):
        out.append(
            {
                "start": int(sp.get("start", 0)),
                "end": int(sp.get("end", 0)),
                "type": _regex_type_for_class(str(sp.get("class", ""))),
            }
        )
    return out


class NerEngine:
    def __init__(self) -> None:
        self._nlp = None
        self._ready = False

    def _load_once(self) -> None:
        if self._ready:
            return
        self._ready = True
        try:
            import spacy
        except Exception:
            self._nlp = None
            return
        for model_name in ("en_core_web_trf", "en_core_web_sm", "en_core_web_md"):
            try:
                self._nlp = spacy.load(model_name)
                return
            except OSError:
                continue
        self._nlp = None

    def detect(self, text: str) -> list[dict]:
        self._load_once()
        if self._nlp is None:
            return []
        out = []
        doc = self._nlp(text[:10000])
        label_map = {
            "PERSON": "NAME",
            "DATE": "ID",
            "GPE": "LOCATION",
            "LOC": "LOCATION",
            "FAC": "LOCATION",
            "ORG": "OTHER_PII",
            "MONEY": "FINANCIAL",
        }
        for ent in doc.ents:
            typ = _norm_phi_type(label_map.get(ent.label_))
            if typ == NONE_TYPE:
                continue
            out.append(
                {
                    "start": int(ent.start_char),
                    "end": int(ent.end_char),
                    "type": typ,
                }
            )
        return out


class MlScorer:
    def __init__(self, model_dir: Path) -> None:
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model not found: {model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir.resolve()))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir.resolve()))
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.id2label = {int(k): str(v).lower() for k, v in (self.model.config.id2label or {}).items()}
        self.n_labels = int(getattr(self.model.config, "num_labels", 2))

    def score_batch(self, texts: list[str], batch_size: int) -> list[float]:
        if not texts:
            return []
        out: list[float] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            tok = self.tokenizer(
                chunk,
                truncation=True,
                max_length=256,
                padding=True,
                return_tensors="pt",
            )
            tok = {k: v.to(self.device) for k, v in tok.items()}
            with torch.no_grad():
                logits = self.model(**tok).logits
                probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            for p in probs:
                out.append(self._risky_prob(p))
        return out

    def _risky_prob(self, probs: np.ndarray) -> float:
        if self.n_labels == 2:
            risky_idx = 1
            for i, name in self.id2label.items():
                if name == "risky":
                    risky_idx = i
                    break
            return float(probs[risky_idx])
        low_idx = med_idx = high_idx = None
        for i, name in self.id2label.items():
            if name == "low":
                low_idx = i
            elif name == "med":
                med_idx = i
            elif name == "high":
                high_idx = i
        if med_idx is not None and high_idx is not None:
            return float(probs[med_idx] + probs[high_idx])
        if high_idx is not None:
            return float(probs[high_idx])
        return float(probs[-1])


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, fp, tn, fn


def _metrics_from(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    tp, fp, tn, fn = _confusion(y_true, y_pred)
    n = max(1, tp + fp + tn + fn)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    f2 = (5 * precision * recall / (4 * precision + recall)) if (4 * precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    denom = math.sqrt(max(1e-12, (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom else 0.0
    try:
        auc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan")
    except ValueError:
        auc = float("nan")
    try:
        auprc = float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan")
    except ValueError:
        auprc = float("nan")
    return {
        "TP": float(tp),
        "FP": float(fp),
        "TN": float(tn),
        "FN": float(fn),
        "Accuracy": (tp + tn) / n,
        "Precision": precision,
        "Recall (TPR)": recall,
        "F1 Score": f1,
        "F2 Score": f2,
        "FPR": fpr,
        "FNR": fnr,
        "Specificity": spec,
        "MCC": mcc,
        "AUC-ROC": auc,
        "AUPRC": auprc,
    }


def _fmt(v: float) -> str:
    if isinstance(v, float) and math.isnan(v):
        return "nan"
    if abs(v) >= 1000 and float(v).is_integer():
        return str(int(v))
    return f"{v:.4f}"


def evaluate(
    records: list[TextRecord],
    ml: MlScorer,
    ner: NerEngine,
    thr_validation_calib: float,
    thr_test_recommended: float,
    batch_size: int,
    categories: list[str],
) -> dict[str, Any]:
    all_candidates = [c for r in records for c in r.candidates]
    y_true = np.asarray([c.is_phi for c in all_candidates], dtype=np.int64)

    regex_ner_hit: list[int] = []
    by_text_predictions: dict[int, list[dict]] = {}
    for idx, rec in enumerate(records):
        preds = detect_regex_spans(rec.text) + ner.detect(rec.text)
        by_text_predictions[idx] = preds
        for c in rec.candidates:
            hit = any(_overlaps(c.start, c.end, p["start"], p["end"]) for p in preds)
            regex_ner_hit.append(1 if hit else 0)
    regex_ner_score = np.asarray(regex_ner_hit, dtype=np.float64)
    regex_ner_pred = (regex_ner_score >= 0.5).astype(np.int64)

    candidate_texts = [c.text for c in all_candidates]
    ml_scores = np.asarray(ml.score_batch(candidate_texts, batch_size=batch_size), dtype=np.float64)
    ml_pred_val = (ml_scores >= thr_validation_calib).astype(np.int64)
    ml_pred_test = (ml_scores >= thr_test_recommended).astype(np.int64)

    # Parallel fusion: regex/NER and ML each evaluate all candidates.
    # Final "Full" decision remains OR fusion (binary decision surface).
    #
    # IMPORTANT for ROC/AUC: do not use max(ml_score, prefilter_binary) as the score because it
    # collapses ranking when prefilter==1 (many points become exactly 1.0).
    # Use a continuous blend for score-based metrics, while keeping OR fusion for decisions.
    prefilter = regex_ner_pred.astype(np.int64)
    full_pred_val = ((prefilter == 1) | (ml_pred_val == 1)).astype(np.int64)
    full_pred_test = ((prefilter == 1) | (ml_pred_test == 1)).astype(np.int64)
    alpha = float(os.environ.get("GUARDRAIL_FUSION_ALPHA_ML", "0.7"))
    alpha = max(0.0, min(1.0, alpha))
    beta = 1.0 - alpha
    regex_ner_conf = regex_ner_score.astype(np.float64)
    full_score_val = (alpha * ml_scores) + (beta * regex_ner_conf)
    full_score_test = full_score_val.copy()

    col = benchmark_column_names(thr_validation_calib, thr_test_recommended)
    configs = {
        col["regex_ner"]: (regex_ner_pred, regex_ner_score),
        col["ml_val_calib"]: (ml_pred_val, ml_scores),
        col["ml_test_rec"]: (ml_pred_test, ml_scores),
        col["full_val_calib"]: (full_pred_val, full_score_val),
        col["full_test_rec"]: (full_pred_test, full_score_test),
    }

    metrics: dict[str, dict[str, float]] = {}
    per_category: dict[str, dict[str, dict[str, float]]] = {}
    for cfg_name, (pred, score) in configs.items():
        metrics[cfg_name] = _metrics_from(y_true, pred, score)
        cat_metrics: dict[str, dict[str, float]] = {}
        for cat in categories:
            mask = np.asarray([c.true_type in (cat, NONE_TYPE) for c in all_candidates], dtype=bool)
            if mask.sum() == 0:
                cat_metrics[cat] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
                continue
            y_cat = np.asarray([1 if c.true_type == cat else 0 for c in all_candidates], dtype=np.int64)[mask]
            p_cat = pred[mask]
            tp, fp, _tn, fn = _confusion(y_cat, p_cat)
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
            cat_metrics[cat] = {"precision": prec, "recall": rec, "f1": f1}
        per_category[cfg_name] = cat_metrics

    return {"metrics": metrics, "per_category": per_category, "n_candidates": int(len(all_candidates))}


def render_report(results: dict[str, Any], categories: list[str], profile_name: str) -> str:
    metrics = results["metrics"]
    per_category = results["per_category"]
    cfgs = list(metrics.keys())
    c0 = 20
    cw = max(18, max(len(c) for c in cfgs) + 2)

    def trunc(s: str, w: int) -> str:
        s = str(s)
        return s if len(s) <= w else s[: max(4, w - 1)] + "…"

    def row_metric(metric_name: str, values: list[str]) -> str:
        cells = [trunc(metric_name, c0).ljust(c0)] + [trunc(v, cw).ljust(cw) for v in values]
        return "│ " + " │ ".join(cells) + " │"

    hdr_cells = [trunc("Metric", c0).ljust(c0)] + [trunc(c, cw).ljust(cw) for c in cfgs]
    hdr_line = "│ " + " │ ".join(hdr_cells) + " │"
    inner = len(hdr_line) - 2
    bar = "─" * inner

    lines: list[str] = []
    lines.append(f"{profile_name.upper()} DETECTION — BENCHMARK REPORT")
    lines.append("┌" + bar + "┐")
    lines.append(hdr_line)
    lines.append("├" + bar + "┤")

    ordered = [
        "Accuracy",
        "Precision",
        "Recall (TPR)",
        "F1 Score",
        "F2 Score",
        "FPR",
        "FNR",
        "Specificity",
        "MCC",
        "AUC-ROC",
        "AUPRC",
        "TP",
        "FP",
        "TN",
        "FN",
    ]
    for m in ordered:
        row = [metrics[cfg][m] for cfg in cfgs]
        lines.append(row_metric(m, [_fmt(v) for v in row]))

    lines.append("└" + bar + "┘")
    lines.append("")
    lines.extend(benchmark_threshold_legend_text().split("\n"))
    lines.append("")
    lines.append(
        "Framing: Category labels come from HIPAA-style span types for evaluation only; "
        "this benchmark is not a HIPAA compliance certification."
    )
    lines.append("")
    lines.append(f"Candidate spans evaluated: {results['n_candidates']}")
    lines.append("")
    lines.append("Per-category Precision / Recall / F1")
    lines.append("-" * 72)
    full_compare = cfgs[3] if len(cfgs) >= 4 else cfgs[-1]
    for cat in categories:
        rg = per_category[cfgs[0]][cat]
        fax = per_category[full_compare][cat]
        lines.append(
            f"{cat:<12}  Regex+NER(P/R/F1)=({_fmt(rg['precision'])}/{_fmt(rg['recall'])}/{_fmt(rg['f1'])})  "
            f"Full val-calib(P/R/F1)=({_fmt(fax['precision'])}/{_fmt(fax['recall'])}/{_fmt(fax['f1'])})"
        )

    deltas = []
    for cat in categories:
        full_f1 = per_category[full_compare][cat]["f1"]
        rg_f1 = per_category[cfgs[0]][cat]["f1"]
        deltas.append((cat, full_f1 - rg_f1))
    deltas.sort(key=lambda x: x[1], reverse=True)

    lines.append("")
    lines.append(f"Categories with largest ML lift vs Regex+NER (Full {full_compare!r}):")
    for cat, d in deltas[:5]:
        lines.append(f"- {cat}: ΔF1={d:+.4f}")

    no_value = [(cat, d) for cat, d in deltas if d <= 0.0]
    lines.append("")
    lines.append("Categories where Regex+NER already matches/exceeds Full val-calib (ΔF1 ≤ 0):")
    if not no_value:
        lines.append("- None")
    else:
        for cat, d in no_value[:8]:
            lines.append(f"- {cat}: ΔF1={d:+.4f}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PHI detection pipeline configurations.")
    parser.add_argument("--dataset", default=None, help="Path to annotated JSONL test set")
    parser.add_argument(
        "--model-dir",
        default=str((_ROOT / "models" / "tinybert_guardrail").resolve()),
        help="HF model directory for binary/3-class classifier",
    )
    parser.add_argument(
        "--model-records",
        default=str((_ROOT.parent / "docs" / "data" / "model_records.json").resolve()),
        help="Path to model_records.json (val-calib + test-rec thresholds; see load_thresholds docstring)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit output path. If omitted, writes profile-specific filename.",
    )
    parser.add_argument("--negatives-per-positive", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument(
        "--profile",
        choices=("hipaa", "pii"),
        default="hipaa",
        help="Category profile for per-type metrics",
    )
    parser.add_argument(
        "--arrow-test-dir",
        default=str((_ROOT / "arrow_datasets" / "test").resolve()),
        help="Project test split path used when --dataset is omitted",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    model_records_path = Path(args.model_records).resolve()
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        default_name = "benchmark_report_pii.txt" if args.profile == "pii" else "benchmark_report_hipaa.txt"
        output_path = (_ROOT / default_name).resolve()
    arrow_test_dir = Path(args.arrow_test_dir).resolve()

    categories = HIPAA_TYPES if args.profile == "hipaa" else PII_TYPES
    profile_name = "HIPAA PHI" if args.profile == "hipaa" else "PII"

    thr_validation_calib, thr_test_recommended = load_thresholds(model_records_path)
    if args.dataset:
        dataset_path = Path(args.dataset).resolve()
        records = load_annotated_records(
            dataset_path,
            negatives_per_positive=max(1, int(args.negatives_per_positive)),
            seed=int(args.seed),
            max_records=args.max_records,
        )
    else:
        records = load_project_arrow_records(
            arrow_test_dir,
            model_dir,
            max_records=args.max_records,
        )
    if not records:
        raise SystemExit("No valid records found in dataset.")

    ml = MlScorer(model_dir)
    ner = NerEngine()
    results = evaluate(
        records,
        ml,
        ner,
        thr_validation_calib,
        thr_test_recommended,
        batch_size=max(1, int(args.batch_size)),
        categories=categories,
    )
    report = render_report(results, categories, profile_name)
    try:
        print(report)
    except UnicodeEncodeError:
        # Windows cp1252 terminals can fail on box-drawing characters.
        print(report.encode("ascii", errors="replace").decode("ascii"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nSaved benchmark report: {output_path}")


if __name__ == "__main__":
    main()
