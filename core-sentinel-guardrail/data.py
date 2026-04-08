"""
Data loading, synthetic generation, and tokenization for Core Sentinel Guardrail.

Loads nvidia/Nemotron-PII, generates Faker-based finance/SAP synthetic samples,
performs stratified train/val/test split, and tokenizes with TinyBERT.

Professional metrics (targets for downstream model):
  - P@high > 90%  (Precision on high-risk class)
  - FPR < 3%      (False Positive Rate overall)
"""

import csv
import json
import os
import random
import re
from collections import Counter
from pathlib import Path
import ast

import datasets
from datasets import Dataset, DatasetDict
from faker import Faker
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# --- Constants ---
NEMOTRON_PII_ID = "nvidia/Nemotron-PII"
TINYBERT_ID = "huawei-noah/TinyBERT_General_4L_312D"
NUM_SYNTHETICS = 2000
RISK_DIST = {"low": 0.30, "med": 0.40, "high": 0.30}
SPLIT_RATIOS = (0.70, 0.15, 0.15)
ARROW_SAVE_DIR = os.environ.get("GUARDRAIL_ARROW_SAVE_DIR", "arrow_datasets")
# Nemotron: map PII span count to risk (0 spans=low, 1-4=med, 5+=high)
NEMOTRON_SPAN_LOW_MAX = 0
NEMOTRON_SPAN_MED_MAX = 4
# Optional caps for Nemotron (None = use full dataset)
NEMOTRON_MAX_TRAIN = 20000
NEMOTRON_MAX_VAL = 3000
NEMOTRON_MAX_TEST = 5000

# Patronus AI / EnterprisePII: load from local path (not on Hugging Face by default)
# Obtain data from llm-foundry (GitHub) or Patronus platform, then place JSON/JSONL/CSV here
PATRONUS_PII_PATH = Path("data/patronus")
# Map EnterprisePII sensitivity -> our risk: public/internal=low, confidential=med, restricted/sensitive=high
PATRONUS_SENSITIVITY_TO_RISK = {
    "public": "low", "internal": "low", "low": "low",
    "confidential": "med", "medium": "med", "med": "med",
    "restricted": "high", "sensitive": "high", "high": "high",
}

MANUAL_EVAL_PATH = Path("data/manual_eval/manual_eval.jsonl")
TEXT_KEYS = ("text", "content", "excerpt", "prompt", "input")
LABEL_KEYS = ("risk", "label", "category", "classification")
LABEL_NORMALIZE = {"medium": "med", "low": "low", "med": "med", "high": "high"}

# BigCode PII: Hugging Face dataset; map by PII/secrets intensity
BIGCODE_PII_ID = "bigcode/bigcode-pii-dataset"
BIGCODE_MAX_ROWS = 15000  # cap to limit download/size; None = use all
BUSINESS_REAL_PATH = Path("data/business_real")
ENRON_REAL_PATH = Path("data/enron_real")
KAGGLE_SENSITIVE_PATH = Path("data/kaggle_sensitive")

# Unified real candidate pool: text length and output
MIN_TEXT_LEN = 20
MAX_TEXT_LEN = 1000
_REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
REAL_CANDIDATE_POOL_CSV = _REPORTS_DIR / "real_candidate_pool.csv"
# Synthetic cap: when enough real labeled data, synthetic_count <= SYNTHETIC_CAP_MULTIPLIER * real_labeled_count
SYNTHETIC_CAP_MULTIPLIER = 2

# --- ai4privacy PII masking (text PII, sequence classification from token-level labels) ---
AI4PRIVACY_PII_ID = "ai4privacy/pii-masking-43k"
# Collapsed label schema: one label per document (aggregated from token-level)
COLLAPSED_PII_LABELS = ["O", "NAME", "CONTACT", "LOCATION", "ID", "FINANCIAL", "HEALTH", "AUTH", "OTHER_PII"]
PII_LABEL2ID = {name: i for i, name in enumerate(COLLAPSED_PII_LABELS)}
PII_ID2LABEL = {i: name for i, name in enumerate(COLLAPSED_PII_LABELS)}
# Priority order for "dominant" PII type in a document (highest index = highest priority when aggregating)
PII_PRIORITY_ORDER = ["O", "NAME", "LOCATION", "CONTACT", "ID", "OTHER_PII", "FINANCIAL", "HEALTH", "AUTH"]
# Map fine-grained ai4privacy tag names (substring match) -> collapsed label
FINE_TO_COLLAPSED = {
    "name": "NAME", "person": "NAME", "per": "NAME", "author": "NAME",
    "email": "CONTACT", "phone": "CONTACT", "contact": "CONTACT", "url": "CONTACT",
    "address": "LOCATION", "location": "LOCATION", "loc": "LOCATION", "city": "LOCATION", "country": "LOCATION",
    "id": "ID", "ssn": "ID", "passport": "ID", "license": "ID", "number": "ID",
    "financial": "FINANCIAL", "bank": "FINANCIAL", "credit": "FINANCIAL", "account": "FINANCIAL", "salary": "FINANCIAL",
    "health": "HEALTH", "medical": "HEALTH", "patient": "HEALTH", "diagnosis": "HEALTH",
    "auth": "AUTH", "password": "AUTH", "token": "AUTH", "key": "AUTH", "secret": "AUTH",
}
LABEL_CONFIG_FILENAME = "label_config.json"
# Kaggle ai4privacy-PII: path relative to this script so it works from any CWD
AI4PRIVACY_KAGGLE_PATH = Path(__file__).resolve().parent / "data" / "ai4privacy_kaggle"
# Map mask tags from target_text (e.g. [CREDITCARDNUMBER], [FIRSTNAME_1]) -> collapsed label
MASK_TAG_TO_COLLAPSED = {
    "FIRSTNAME": "NAME", "LASTNAME": "NAME", "FULLNAME": "NAME", "PERSON": "NAME", "NAME": "NAME",
    "PHONENUMBER": "CONTACT", "PHONEIMEI": "CONTACT", "EMAIL": "CONTACT", "URL": "CONTACT", "CONTACT": "CONTACT",
    "ADDRESS": "LOCATION", "ZIPCODE": "LOCATION", "CITY": "LOCATION", "STATE": "LOCATION",
    "COUNTRY": "LOCATION", "LOCATION": "LOCATION",
    "SSN": "ID", "ID": "ID", "PASSPORT": "ID", "LICENSE": "ID", "VEHICLEVIN": "ID",
    "CREDITCARDNUMBER": "FINANCIAL", "IBAN": "FINANCIAL", "BANKACCOUNT": "FINANCIAL", "FINANCIAL": "FINANCIAL",
    "HEALTH": "HEALTH", "MEDICAL": "HEALTH", "HEIGHT": "OTHER_PII",
    "PASSWORD": "AUTH", "TOKEN": "AUTH", "CRYPTO": "AUTH", "IP": "AUTH", "AUTH": "AUTH",
    "GENDER": "OTHER_PII", "AGE": "OTHER_PII", "DATE": "OTHER_PII", "JOBAREA": "OTHER_PII", "OTHER": "OTHER_PII",
}
ENGLISH_LANGUAGE_CODES = ("en", "en-US", "en-GB", "en_US", "en_GB")


def _tags_from_masked_text(masked: str) -> set:
    """Extract PII tag names from target_text like [CREDITCARDNUMBER], [PHONENUMBER]. Returns set of collapsed labels."""
    if not masked:
        return set()
    tags = re.findall(r"\[([A-Za-z0-9_]+)\]", str(masked))
    out = set()
    for t in tags:
        # Strip suffix like _1, _2 (e.g. FIRSTNAME_1 -> FIRSTNAME)
        base = t.upper().split("_")[0] if "_" in t else t.upper()
        c = MASK_TAG_TO_COLLAPSED.get(base) or MASK_TAG_TO_COLLAPSED.get(t.upper(), "OTHER_PII")
        if c != "O":
            out.add(c)
    return out


def load_ai4privacy_kaggle_en(
    path: str | Path | None = None,
    max_train: int | None = 35000,
    max_test: int | None = 5000,
) -> list[dict]:
    """
    Load English PII from Kaggle ai4privacy-PII (verracodeguacas/ai4privacy-pii).
    Expects CSV or JSONL under path with source_text (or text/prompt), target_text (or masked), language.
    Filters to English only. Returns list of {text, label, risk} with collapsed 9-class labels (0..8).
    """
    base = Path(path or AI4PRIVACY_KAGGLE_PATH).resolve()
    if not base.exists():
        print(f"[ai4privacy_kaggle_en] path not found: {base}")
        print("  Download from https://www.kaggle.com/datasets/verracodeguacas/ai4privacy-pii/data and extract CSV/JSONL there.")
        return []

    text_keys = ("unmasked_text", "source_text", "text", "prompt", "input", "Filled Template")
    masked_keys = ("masked_text", "target_text", "masked", "target", "Template")
    lang_key = "language"
    rows_read = []
    for p in list(base.rglob("*.csv"))[:10] + list(base.rglob("*.jsonl"))[:10] + list(base.rglob("*.json"))[:5]:
        try:
            with open(p, encoding="utf-8", errors="replace") as fp:
                if p.suffix.lower() == ".csv":
                    rows_read.extend(list(csv.DictReader(fp)))
                elif p.suffix.lower() == ".jsonl":
                    rows_read.extend(json.loads(line) for line in fp if line.strip())
                else:
                    data = json.load(fp)
                    rows_read.extend(data if isinstance(data, list) else [data])
        except Exception as e:
            print(f"[ai4privacy_kaggle_en] skip {p}: {e}")
            continue

    out = []
    for r in rows_read:
        lang = (r.get(lang_key) or r.get("lang") or "").strip()
        # If language is set, keep only English; if missing (e.g. english_*.jsonl), accept
        if lang and lang not in ENGLISH_LANGUAGE_CODES:
            continue
        text = None
        for k in text_keys:
            if k in r and r[k] not in (None, ""):
                text = str(r[k]).strip()
                break
        if not text or len(text) < 10:
            continue
        text = text[:MAX_TEXT_LEN]
        masked = None
        for k in masked_keys:
            if k in r and r[k] not in (None, ""):
                masked = str(r[k]).strip()
                break
        collected = _tags_from_masked_text(masked)
        if not collected:
            collapsed = "O"
        else:
            order_idx = {x: i for i, x in enumerate(PII_PRIORITY_ORDER)}
            collapsed = max(collected, key=lambda x: order_idx.get(x, -1))
        # label = collapsed 9-class PII label id (0..8), not 3-class low/med/high risk
        label_id = PII_LABEL2ID.get(collapsed, 0)
        out.append({"text": text, "label": label_id, "risk": collapsed})

    if max_train is not None and len(out) > (max_train + (max_test or 0)):
        random.seed(42)
        random.shuffle(out)
        out = out[: max_train + (max_test or 0)]
    counts = dict(Counter(r["risk"] for r in out))
    print(f"[ai4privacy_kaggle_en] English rows: {len(out)} (from {len(rows_read)} total)")
    print(f"[ai4privacy_kaggle_en] per-class (collapsed): {counts}")
    return out


def load_manual_eval(path: str | Path | None = None) -> list[dict]:
    """
    Load manual evaluation set from JSON, JSONL, or CSV.
    Auto-detects text and label columns; normalizes labels to low/med/high.
    Returns list of {"text": str, "risk": str, "is_sap": 0}.
    """
    path = Path(path or MANUAL_EVAL_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Manual eval file not found: {path}")
    rows = []
    if path.suffix == ".csv":
        import csv
        with open(path, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                rows.append(dict(r))
    elif path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        rows = data if isinstance(data, list) else [data]
    out = []
    for r in rows:
        text = None
        for k in TEXT_KEYS:
            if k in r and r[k] not in (None, ""):
                text = str(r[k]).strip()
                break
        if not text:
            continue
        raw = None
        for k in LABEL_KEYS:
            if k in r and r[k] is not None:
                raw = str(r[k]).strip().lower()
                break
        risk = LABEL_NORMALIZE.get(raw, "med") if raw else "med"
        if risk not in ("low", "med", "high"):
            risk = "med"
        out.append({"text": text, "risk": risk, "is_sap": 0})
    return out


def load_nemotron_pii() -> DatasetDict:
    """
    Load nvidia/Nemotron-PII from Hugging Face datasets.

    If the dataset is gated, set an auth token via one of:
      - `HF_TOKEN`
      - `HUGGINGFACE_TOKEN`
      - `HUGGINGFACEHUB_API_TOKEN`
      - `HUGGINGFACE_HUB_TOKEN`
    (or run `huggingface-cli login` and rely on cached credentials).
    """
    hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )
    try:
        if hf_token:
            return datasets.load_dataset(NEMOTRON_PII_ID, token=hf_token)
        return datasets.load_dataset(NEMOTRON_PII_ID)
    except Exception as e:
        # Try cached login (token=True uses credentials stored by Hugging Face CLI).
        try:
            return datasets.load_dataset(NEMOTRON_PII_ID, token=True)
        except Exception:
            raise RuntimeError(
                f"Failed to load {NEMOTRON_PII_ID}. "
                "If it's gated, run `huggingface-cli login` or set an HF token via "
                "`HF_TOKEN` (or similar env vars in load_nemotron_pii())."
            ) from e


def _spans_to_risk(spans, low_max: int = NEMOTRON_SPAN_LOW_MAX, med_max: int = NEMOTRON_SPAN_MED_MAX) -> str:
    """
    Derive risk from Nemotron `spans` metadata.

    Updated mapping:
      - "high" if span_count >= 4 OR any span's type is in
        {"ssn", "credit_card", "api_key", "password"}
      - "med" if span_count >= 2
      - "low" if span_count <= 1 and no critical type detected

    Any span whose label contains "critical_secret" forces "high".
    """
    if spans is None:
        return "low"
    if isinstance(spans, str):
        try:
            spans = json.loads(spans)
        except Exception:
            # Nemotron `spans` is often a Python-literal string (single quotes),
            # not strict JSON. Fall back to literal_eval in that case.
            try:
                spans = ast.literal_eval(spans)
            except Exception:
                return "low"

    if not hasattr(spans, "__len__"):
        return "low"

    critical_secret = False
    critical_type_detected = False
    span_count = 0
    critical_types = {"ssn", "credit_card", "api_key", "password"}

    for sp in spans:
        if not isinstance(sp, dict):
            # Best-effort: treat unknown shapes as non-sensitive.
            span_count += 1
            continue
        lbl = str(sp.get("label") or "").lower()
        if "critical_secret" in lbl:
            critical_secret = True
            continue

        span_count += 1
        for ct in critical_types:
            if ct in lbl:
                critical_type_detected = True
                break

    if critical_secret:
        return "high"
    if critical_type_detected or span_count >= 4:
        return "high"
    if span_count >= 2:
        return "med"
    return "low"


def _print_risk_distribution(rows: list[dict], *, name: str) -> None:
    """Print low/med/high label distribution for a Nemotron-derived split."""
    total = len(rows)
    counts = Counter(r.get("risk") for r in rows)
    low = int(counts.get("low", 0))
    med = int(counts.get("med", 0))
    high = int(counts.get("high", 0))
    if total <= 0:
        print(f"[Nemotron] {name}: n=0 (no rows)")
        return
    low_pct = 100.0 * low / total
    med_pct = 100.0 * med / total
    high_pct = 100.0 * high / total
    print(
        f"[Nemotron] {name} risk distribution (n={total}): "
        f"low={low} ({low_pct:.2f}%), med={med} ({med_pct:.2f}%), high={high} ({high_pct:.2f}%)"
    )


def nemotron_to_risk_rows(
    nemotron: DatasetDict,
    max_train: int | None = NEMOTRON_MAX_TRAIN,
    max_val: int | None = NEMOTRON_MAX_VAL,
    max_test: int | None = NEMOTRON_MAX_TEST,
    low_max: int = NEMOTRON_SPAN_LOW_MAX,
    med_max: int = NEMOTRON_SPAN_MED_MAX,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Convert nvidia/Nemotron-PII to (train, val, test) rows with text, risk, sap_id.
    Risk is derived from span count per document. Val is split from train.
    """
    random.seed(seed)
    train_ds = nemotron["train"]
    test_ds = nemotron["test"]

    def to_rows(ds, cap: int | None):
        rows = []
        for i, row in enumerate(ds):
            if cap is not None and i >= cap:
                break
            text = row.get("text") or ""
            if not text or not text.strip():
                continue
            spans = row.get("spans")
            risk = _spans_to_risk(spans, low_max=low_max, med_max=med_max)
            uid = row.get("uid") or f"nemotron-{i}"
            rows.append({"text": text.strip(), "risk": risk, "sap_id": str(uid)})
        return rows

    train_rows_full = to_rows(train_ds, max_train)
    test_rows = to_rows(test_ds, max_test)

    # Stratified split of train into train + val
    texts = [r["text"] for r in train_rows_full]
    risks = [r["risk"] for r in train_rows_full]
    saps = [r["sap_id"] for r in train_rows_full]
    # Stratify only when each class has enough samples (sklearn requires >=2 per class)
    counts = Counter(risks)
    use_stratify = bool(counts) and min(counts.values()) >= 2
    train_texts, val_texts, train_risks, val_risks, train_saps, val_saps = train_test_split(
        texts,
        risks,
        saps,
        test_size=0.15,
        stratify=risks if use_stratify else None,
        random_state=seed,
    )
    train_rows = [{"text": t, "risk": r, "sap_id": s} for t, r, s in zip(train_texts, train_risks, train_saps)]
    val_rows = [{"text": t, "risk": r, "sap_id": s} for t, r, s in zip(val_texts, val_risks, val_saps)]
    if max_val is not None and len(val_rows) > max_val:
        val_rows = random.sample(val_rows, max_val)
    _print_risk_distribution(train_rows, name="Nemotron train")
    _print_risk_distribution(val_rows, name="Nemotron validation")
    _print_risk_distribution(test_rows, name="Nemotron test")
    return train_rows, val_rows, test_rows


def load_patronus_pii(path: str | Path | None = None) -> list[dict]:
    """
    Load ALL Patronus/EnterprisePII rows from a path (file or directory).
    Supports multiple .json, .jsonl, .csv files; concatenates and deduplicates by normalized text.
    Returns list of {text, risk, sap_id}. Risk derived from sensitivity/label column.
    """
    path = Path(path or PATRONUS_PII_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Patronus PII path not found: {path}. "
            "Obtain EnterprisePII (e.g. from llm-foundry) and place JSON/JSONL/CSV there."
        )
    text_keys = ("text", "content", "excerpt", "prompt", "input")
    label_keys = ("sensitivity", "label", "risk", "category", "classification")

    if path.is_file():
        files = [path]
    else:
        files = sorted(path.glob("*.json")) + sorted(path.glob("*.jsonl")) + sorted(path.glob("*.csv"))

    raw_rows = []
    for f in files:
        if f.suffix == ".csv":
            import csv
            with open(f, encoding="utf-8", newline="") as fp:
                for r in csv.DictReader(fp):
                    raw_rows.append(dict(r))
        elif f.suffix == ".jsonl":
            with open(f, encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if line:
                        raw_rows.append(json.loads(line))
        else:
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)
            if isinstance(data, list):
                raw_rows.extend(data)
            else:
                raw_rows.append(data)

    total_raw = len(raw_rows)
    unique_raw_labels = set()
    seen_normalized = set()
    out = []
    for i, r in enumerate(raw_rows):
        text = None
        for k in text_keys:
            if k in r and r[k] not in (None, ""):
                text = str(r[k]).strip()
                break
        if not text:
            continue
        raw_label = None
        for k in label_keys:
            if k in r and r[k] is not None:
                raw_label = str(r[k]).strip().lower()
                break
        if raw_label is not None:
            unique_raw_labels.add(raw_label)
        risk = PATRONUS_SENSITIVITY_TO_RISK.get(raw_label, "med") if raw_label else "med"
        norm = text.lower().strip()
        if norm in seen_normalized:
            continue
        seen_normalized.add(norm)
        out.append({"text": text, "risk": risk, "sap_id": f"patronus-{len(out)}", "source": "patronus"})
    usable_count = len(out)
    mapped_counts = dict(Counter(r["risk"] for r in out))

    print(f"[Patronus] total raw rows loaded: {total_raw}")
    print(f"[Patronus] total usable rows (after valid text + dedup): {usable_count}")
    print(f"[Patronus] unique label values before mapping: {sorted(unique_raw_labels)}")
    print(f"[Patronus] mapped per-class counts (low/med/high): {mapped_counts}")
    return out


def _bigcode_risk(type_val, fragments) -> str:
    """Map BigCode PII type/fragments to risk: secrets/PII-heavy=high, mixed=med, safe=low."""
    frag_count = 0
    if fragments is not None and hasattr(fragments, "__len__"):
        frag_count = len(fragments)
    type_str = (str(type_val) or "").lower()
    if frag_count >= 3 or "secret" in type_str or "key" in type_str or "password" in type_str or "token" in type_str:
        return "high"
    if frag_count == 0 and ("comment" in type_str or "safe" in type_str or not type_str):
        return "low"
    return "med"


def load_bigcode_pii(max_rows: int | None = BIGCODE_MAX_ROWS) -> list[dict]:
    """
    Load BigCode PII dataset from Hugging Face (text/code with PII annotations).
    Maps: secrets/PII-heavy -> high, mixed/uncertain -> med, safe code/comments -> low.
    Uses 'text' (or 'code') and type/fragments for risk. Returns list of {text, risk, sap_id}.
    If the dataset is gated or unavailable, returns [] and prints a warning (run: huggingface-cli login).
    """
    try:
        # use_auth_token/token=True uses cached HF login for gated datasets
        ds = datasets.load_dataset(BIGCODE_PII_ID, split="test", token=True)
    except Exception as e:
        print(f"[BigCode PII] skipped (gated/unavailable): {e}")
        print("[BigCode PII] To use it: accept terms at https://huggingface.co/datasets/bigcode/bigcode-pii-dataset and run: huggingface-cli login")
        return []
    text_key = "text" if "text" in ds.column_names else "code"
    out = []
    for i, row in enumerate(ds):
        if max_rows is not None and i >= max_rows:
            break
        text = row.get(text_key) or row.get("content") or ""
        if not text or not str(text).strip():
            continue
        text = str(text).strip()
        type_val = row.get("type")
        fragments = row.get("fragments")
        risk = _bigcode_risk(type_val, fragments)
        out.append({"text": text, "risk": risk, "sap_id": f"bigcode-{i}", "source": "bigcode"})
    counts = dict(Counter(r["risk"] for r in out))
    print(f"[BigCode PII] total rows loaded: {len(out)}")
    print(f"[BigCode PII] per-class counts (low/med/high): {counts}")
    return out


def load_business_real(path: str | Path | None = None) -> list[dict]:
    """
    Load optional local business/Enron-style data from path (file or directory).
    Supports .json, .jsonl, .csv; same text/label detection as Patronus; normalizes to low/med/high.
    Returns list of {text, risk, sap_id}.
    """
    path = Path(path or BUSINESS_REAL_PATH)
    if not path.exists():
        print(f"[business_real] path not found (skipped): {path}")
        return []
    text_keys = ("text", "content", "excerpt", "prompt", "input", "body")
    label_keys = ("risk", "label", "category", "classification", "sensitivity")

    if path.is_file():
        files = [path]
    else:
        files = sorted(path.glob("*.json")) + sorted(path.glob("*.jsonl")) + sorted(path.glob("*.csv"))

    raw_rows = []
    for f in files:
        if f.suffix == ".csv":
            import csv
            with open(f, encoding="utf-8", newline="") as fp:
                for r in csv.DictReader(fp):
                    raw_rows.append(dict(r))
        elif f.suffix == ".jsonl":
            with open(f, encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if line:
                        raw_rows.append(json.loads(line))
        else:
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)
            if isinstance(data, list):
                raw_rows.extend(data)
            else:
                raw_rows.append(data)

    out = []
    for i, r in enumerate(raw_rows):
        text = None
        for k in text_keys:
            if k in r and r[k] not in (None, ""):
                text = str(r[k]).strip()
                break
        if not text:
            continue
        raw_label = None
        for k in label_keys:
            if k in r and r[k] is not None:
                raw_label = str(r[k]).strip().lower()
                break
        risk = PATRONUS_SENSITIVITY_TO_RISK.get(raw_label, "med") if raw_label else None
        out.append({"text": text, "risk": risk, "sap_id": f"business-{len(out)}", "source": "business_real"})
    labeled = [r for r in out if r.get("risk") is not None]
    counts = dict(Counter(r["risk"] for r in labeled)) if labeled else {}
    print(f"[business_real] total rows loaded: {len(out)} (labeled: {len(labeled)}, unlabeled: {len(out) - len(labeled)})")
    print(f"[business_real] per-class counts (low/med/high): {counts}")
    return out


def load_enron_real(path: str | Path | None = None) -> list[dict]:
    """
    Load Enron/EDRM-style email or body text from local path.
    Supports .csv, .json, .jsonl, .txt. Extracts text snippets; if no label column, rows are unlabeled candidates.
    Returns list of {text, risk, sap_id, source="enron_real"}. risk is None if unlabeled.
    """
    path = Path(path or ENRON_REAL_PATH)
    if not path.exists():
        print(f"[enron_real] path not found (skipped): {path}")
        return []
    text_keys = ("text", "content", "body", "excerpt", "prompt", "input", "message")
    label_keys = ("risk", "label", "category", "classification", "sensitivity")

    if path.is_file():
        files = [path]
    else:
        files = (
            sorted(path.glob("*.csv")) + sorted(path.glob("*.json")) + sorted(path.glob("*.jsonl")) + sorted(path.glob("*.txt"))
        )

    raw_rows = []
    for f in files:
        if f.suffix == ".csv":
            import csv
            with open(f, encoding="utf-8", newline="") as fp:
                for r in csv.DictReader(fp):
                    raw_rows.append(dict(r))
        elif f.suffix == ".jsonl":
            with open(f, encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if line:
                        raw_rows.append(json.loads(line))
        elif f.suffix == ".json":
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)
            if isinstance(data, list):
                raw_rows.extend(data)
            else:
                raw_rows.append(data)
        else:
            # .txt: whole file or split by double newline into snippets
            with open(f, encoding="utf-8", errors="replace") as fp:
                content = fp.read()
            for part in content.split("\n\n"):
                part = part.strip()
                if len(part) >= MIN_TEXT_LEN:
                    raw_rows.append({"text": part})
            if not raw_rows and content.strip() and len(content.strip()) >= MIN_TEXT_LEN:
                raw_rows.append({"text": content.strip()})

    out = []
    for i, r in enumerate(raw_rows):
        text = None
        for k in text_keys:
            if k in r and r[k] not in (None, ""):
                text = str(r[k]).strip()
                break
        if not text and "text" not in (text_keys and r):
            # .txt rows already have "text"
            if "text" in r:
                text = str(r["text"]).strip()
        if not text:
            continue
        raw_label = None
        for k in label_keys:
            if k in r and r[k] is not None:
                raw_label = str(r[k]).strip().lower()
                break
        risk = PATRONUS_SENSITIVITY_TO_RISK.get(raw_label) if raw_label else None
        out.append({"text": text, "risk": risk, "sap_id": f"enron-{len(out)}", "source": "enron_real"})
    labeled = [r for r in out if r.get("risk") is not None]
    counts = dict(Counter(r["risk"] for r in labeled)) if labeled else {}
    print(f"[enron_real] total rows loaded: {len(out)} (labeled: {len(labeled)}, unlabeled: {len(out) - len(labeled)})")
    print(f"[enron_real] per-class counts (low/med/high): {counts}")
    return out


def load_kaggle_sensitive(path: str | Path | None = None) -> list[dict]:
    """
    Load optional Kaggle or local structured-sensitive data from path.
    Supports .csv, .json, .jsonl. Map labels to low/med/high if present; else unlabeled candidates.
    Returns list of {text, risk, sap_id, source="kaggle_sensitive"}.
    """
    path = Path(path or KAGGLE_SENSITIVE_PATH)
    if not path.exists():
        print(f"[kaggle_sensitive] path not found (skipped): {path}")
        return []
    text_keys = ("text", "content", "excerpt", "prompt", "input", "description", "body")
    label_keys = ("risk", "label", "category", "classification", "sensitivity")

    if path.is_file():
        files = [path]
    else:
        files = sorted(path.glob("*.csv")) + sorted(path.glob("*.json")) + sorted(path.glob("*.jsonl"))

    raw_rows = []
    for f in files:
        if f.suffix == ".csv":
            import csv
            with open(f, encoding="utf-8", newline="") as fp:
                for r in csv.DictReader(fp):
                    raw_rows.append(dict(r))
        elif f.suffix == ".jsonl":
            with open(f, encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if line:
                        raw_rows.append(json.loads(line))
        else:
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)
            if isinstance(data, list):
                raw_rows.extend(data)
            else:
                raw_rows.append(data)

    out = []
    for i, r in enumerate(raw_rows):
        text = None
        for k in text_keys:
            if k in r and r[k] not in (None, ""):
                text = str(r[k]).strip()
                break
        if not text:
            continue
        raw_label = None
        for k in label_keys:
            if k in r and r[k] is not None:
                raw_label = str(r[k]).strip().lower()
                break
        risk = PATRONUS_SENSITIVITY_TO_RISK.get(raw_label, "med") if raw_label else None
        out.append({"text": text, "risk": risk, "sap_id": f"kaggle-{len(out)}", "source": "kaggle_sensitive"})
    labeled = [r for r in out if r.get("risk") is not None]
    counts = dict(Counter(r["risk"] for r in labeled)) if labeled else {}
    print(f"[kaggle_sensitive] total rows loaded: {len(out)} (labeled: {len(labeled)}, unlabeled: {len(out) - len(labeled)})")
    print(f"[kaggle_sensitive] per-class counts (low/med/high): {counts}")
    return out


def load_financial_pii_xlsx(path_env: str = "GUARDRAIL_FINANCIAL_PII_PATH") -> tuple[list[str], list[str]]:
    """
    Load Financial PII xlsx (Training_Set.xlsx + Testing_Set.xlsx) from GUARDRAIL_FINANCIAL_PII_PATH.
    Maps fine-grained labels to low/med/high. Returns (texts, risks) for merging into multi_real_synthetic.
    """
    try:
        import pandas as pd
    except ImportError:
        print("[data] pandas not installed; pip install pandas openpyxl to load Financial PII xlsx")
        return [], []

    base = os.environ.get(path_env)
    if not base:
        print("[data] GUARDRAIL_FINANCIAL_PII_PATH not set, skipping")
        return [], []

    train_path = os.path.join(base, "Training_Set.xlsx")
    test_path = os.path.join(base, "Testing_Set.xlsx")

    dfs = []
    for p in (train_path, test_path):
        if os.path.exists(p):
            dfs.append(pd.read_excel(p))
            print(f"[data] loaded {p}: {len(dfs[-1])} rows")

    if not dfs:
        print("[data] no xlsx files found at path")
        return [], []

    df = pd.concat(dfs, ignore_index=True)

    text_col = next(
        (c for c in df.columns if str(c).lower() in ("text", "sentence", "content", "document")),
        df.columns[0],
    )
    if len(df.columns) < 2:
        print("[data] Financial PII xlsx: need at least 2 columns (text + label)")
        return [], []
    label_col = next(
        (c for c in df.columns if str(c).lower() in ("label", "category", "pii_type", "class")),
        df.columns[1],
    )

    print(f"[data] using text_col='{text_col}' label_col='{label_col}'")
    print(f"[data] label distribution:\n{df[label_col].value_counts()}")

    label_map = {
        "ssn": "high",
        "social_security": "high",
        "credit_card": "high",
        "card": "high",
        "passport": "high",
        "account": "high",
        "iban": "high",
        "auth": "high",
        "password": "high",
        "api_key": "high",
        "name": "med",
        "person": "med",
        "phone": "med",
        "email": "med",
        "address": "med",
        "location": "med",
        "dob": "med",
        "date_of_birth": "med",
        "o": "low",
        "none": "low",
        "non-pii": "low",
        "safe": "low",
    }

    texts: list[str] = []
    labels: list[str] = []
    skipped = 0
    for _, row in df.iterrows():
        text = str(row[text_col]).strip()
        raw_l = str(row[label_col]).strip().lower()

        risk = label_map.get(raw_l)
        if risk is None:
            for k, v in label_map.items():
                if raw_l.startswith(k) or (len(k) >= 3 and k in raw_l):
                    risk = v
                    break

        if risk is None:
            skipped += 1
            continue

        if len(text) > 10:
            texts.append(text)
            labels.append(risk)

    print(f"[data] mapped {len(texts)} rows, skipped {skipped} unmapped")
    return texts, labels


def _fine_tag_to_collapsed(tag: str) -> str:
    """Map a single fine-grained ai4privacy tag (e.g. B-PER, I-EMAIL) to collapsed label."""
    if not tag:
        return "O"
    t = str(tag).upper()
    if t in ("O", "0"):
        return "O"
    for key, collapsed in FINE_TO_COLLAPSED.items():
        if key.upper() in t:
            return collapsed
    return "OTHER_PII"


def load_ai4privacy_pii(max_train: int | None = 35000, max_test: int | None = 5000) -> list[dict]:
    """
    Load ai4privacy/pii-masking-43k from Hugging Face. Dataset is token-level NER; we aggregate
    to one collapsed PII label per document (by dominant type). Returns list of {text, label, risk}
    where label is 0-8 (COLLAPSED_PII_LABELS) and risk is label name for display.
    """
    try:
        ds = datasets.load_dataset(AI4PRIVACY_PII_ID, token=True)
    except Exception as e:
        print(f"[ai4privacy] skipped (load failed): {e}")
        return []
    # Handle DatasetDict: usually "train" and "test"
    if hasattr(ds, "keys"):
        keys = list(ds.keys())
        train_ds = ds["train"] if "train" in keys else ds[keys[0]]
        test_ds = ds["test"] if "test" in keys else None
    else:
        train_ds = ds
        test_ds = None
    if train_ds is None:
        print("[ai4privacy] no train split found")
        return []
    # Detect column names: tokens + ner_tags or labels
    cols = train_ds.column_names
    token_col = "tokens" if "tokens" in cols else ("input_ids" if "input_ids" in cols else None)
    tag_col = "ner_tags" if "ner_tags" in cols else ("labels" if "labels" in cols else ("ner_tags" if "ner_tags" in cols else None))
    if not token_col or not tag_col:
        # Fallback: try "text" or first column for text
        if "text" in cols:
            token_col = "text"
            tag_col = None
    out = []
    # Get label names from features if available (ClassLabel or Sequence(ClassLabel))
    id2fine = {}
    if tag_col and tag_col in train_ds.features:
        f = train_ds.features[tag_col]
        if hasattr(f, "feature"):
            f = f.feature
        if hasattr(f, "names"):
            id2fine = {i: n for i, n in enumerate(f.names)}
        elif hasattr(f, "_int2str"):
            id2fine = getattr(f, "_int2str", {}) or {}
    n = 0
    for row in train_ds:
        if max_train is not None and n >= max_train:
            break
        if token_col == "tokens":
            tokens = row.get(token_col) or []
            text = " ".join(tokens) if isinstance(tokens, list) else str(tokens)
        elif token_col == "text":
            text = row.get(token_col) or ""
        else:
            text = str(row.get(cols[0], ""))
        if not text or len(text.strip()) < 10:
            continue
        text = text.strip()[:MAX_TEXT_LEN]
        collected = set()
        if tag_col and tag_col in row:
            tags = row[tag_col]
            if isinstance(tags, list):
                for t in tags:
                    if id2fine and isinstance(t, int):
                        fine = id2fine.get(t, "O")
                    else:
                        fine = str(t)
                    c = _fine_tag_to_collapsed(fine)
                    if c != "O":
                        collected.add(c)
        if not collected:
            collapsed = "O"
        else:
            # Dominant = highest priority in PII_PRIORITY_ORDER
            order_idx = {x: i for i, x in enumerate(PII_PRIORITY_ORDER)}
            collapsed = max(collected, key=lambda x: order_idx.get(x, -1))
        # label = collapsed 9-class PII label id (0..8), not 3-class low/med/high risk
        label_id = PII_LABEL2ID.get(collapsed, 0)
        out.append({"text": text, "label": label_id, "risk": collapsed})
        n += 1
    n_after_train = len(out)
    if test_ds and max_test:
        for row in test_ds:
            if len(out) >= n_after_train + max_test:
                break
            if token_col == "tokens":
                tokens = row.get(token_col) or []
                text = " ".join(tokens) if isinstance(tokens, list) else str(tokens)
            elif token_col == "text":
                text = row.get(token_col) or ""
            else:
                text = str(row.get(cols[0], ""))
            if not text or len(text.strip()) < 10:
                continue
            text = text.strip()[:MAX_TEXT_LEN]
            collected = set()
            if tag_col and tag_col in row:
                for t in (row[tag_col] or []):
                    if id2fine and isinstance(t, int):
                        fine = id2fine.get(t, "O")
                    else:
                        fine = str(t)
                    c = _fine_tag_to_collapsed(fine)
                    if c != "O":
                        collected.add(c)
            collapsed = "O" if not collected else max(collected, key=lambda x: PII_PRIORITY_ORDER.index(x) if x in PII_PRIORITY_ORDER else 0)
            # label = collapsed 9-class PII label id (0..8), not 3-class low/med/high risk
            out.append({"text": text, "label": PII_LABEL2ID.get(collapsed, 0), "risk": collapsed})
    counts = dict(Counter(r["risk"] for r in out))
    print(f"[ai4privacy] total rows loaded: {len(out)}")
    print(f"[ai4privacy] per-class (collapsed) counts: {counts}")
    return out


def _rebalance_pii_rows(
    rows: list[dict],
    label_key: str = "label",
    risk_key: str = "risk",
    max_per_class: int | None = None,
    min_per_class: int | None = None,
) -> list[dict]:
    """
    Rebalance 9-class PII rows: cap majority (max_per_class) and/or oversample minority (min_per_class).
    Helps when OTHER_PII dominates and ID/HEALTH/O are rare. Returns new list (shuffled).
    """
    if max_per_class is None and min_per_class is None:
        return list(rows)
    random.seed(42)
    by_risk = {}
    for r in rows:
        k = r.get(risk_key, r.get(label_key, 0))
        by_risk.setdefault(k, []).append(r)
    out = []
    for risk, group in by_risk.items():
        if max_per_class is not None and len(group) > max_per_class:
            group = random.sample(group, max_per_class)
        if min_per_class is not None and len(group) < min_per_class:
            group = group + random.choices(group, k=min_per_class - len(group))
        out.extend(group)
    random.shuffle(out)
    counts = dict(Counter(r.get(risk_key) for r in out))
    print(f"[rebalance] after: n={len(out)}, per_class={counts}")
    return out


def _stratified_split_by_label(rows: list[dict], label_key: str = "label", ratios: tuple = (0.85, 0.15, 0.0)) -> tuple:
    """Stratified split by label_key (int); returns (train, val, test) lists. Same logic as stratified_split."""
    train_ratio, val_ratio, test_ratio = ratios
    random.seed(42)
    rows = list(rows)
    random.shuffle(rows)
    labels = [r[label_key] for r in rows]
    sap_ids = [r.get("sap_id", f"pii-{i}") for i, r in enumerate(rows)]
    min_count = min(Counter(labels).values()) if labels else 0
    use_stratify = min_count >= 2
    n = len(rows)
    if use_stratify and len(set(labels)) > 1:
        train_rows, rest = train_test_split(rows, test_size=(1 - train_ratio), stratify=labels, random_state=42)
        rest_labels = [r[label_key] for r in rest]
        val_frac = val_ratio / (val_ratio + test_ratio) if (val_ratio + test_ratio) > 0 else 0
        if val_frac > 0 and val_frac < 1.0 and len(rest) > 0 and min(Counter(rest_labels).values()) >= 2:
            val_rows, test_rows = train_test_split(rest, test_size=(1 - val_frac), stratify=rest_labels, random_state=42)
        else:
            val_rows = rest[: int(len(rest) * val_frac)] if val_frac > 0 else rest
            test_rows = rest[len(val_rows):]
    else:
        n_train = max(1, int(n * train_ratio))
        n_val = max(0, int(n * val_ratio))
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test, n_val = 0, n - n_train
        train_rows = rows[:n_train]
        val_rows = rows[n_train : n_train + n_val]
        test_rows = rows[n_train + n_val :]
    return train_rows, val_rows, test_rows


def _build_real_pool_from_sources(
    source_lists: list[tuple[str, list[dict]]],
    out_csv: Path | str | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Merge real sources: normalize text, dedupe, filter 20–1000 chars. Save to CSV.
    Returns (all_merged_for_csv, labeled_only) where labeled_only have risk set for training.
    """
    import csv as csv_module
    out_csv = Path(out_csv or REAL_CANDIDATE_POOL_CSV)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    before_counts = {name: len(rows) for name, rows in source_lists}
    print("[real_candidate_pool] Per-source counts (before filter/dedup):", before_counts)

    seen = set()
    merged = []
    for name, rows in source_lists:
        for r in rows:
            text = (r.get("text") or "").strip()
            if not text:
                continue
            norm = text.strip().lower()
            if norm in seen:
                continue
            if len(text) < MIN_TEXT_LEN or len(text) > MAX_TEXT_LEN:
                continue
            seen.add(norm)
            risk = r.get("risk")
            merged.append({
                "text": text,
                "source": r.get("source") or name,
                "risk": risk if risk is not None else "",
                "sap_id": r.get("sap_id", f"pool-{len(merged)}"),
            })

    after_counts = dict(Counter(r["source"] for r in merged))
    print("[real_candidate_pool] Per-source counts (after filter/dedup):", after_counts)
    labeled = [r for r in merged if r.get("risk")]
    print(f"[real_candidate_pool] Total merged: {len(merged)} (labeled: {len(labeled)}, unlabeled: {len(merged) - len(labeled)})")
    print(f"[real_candidate_pool] Saving to {out_csv}")

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv_module.DictWriter(f, fieldnames=["text", "source", "risk", "sap_id"])
        w.writeheader()
        w.writerows(merged)

    return merged, labeled


def build_real_candidate_pool(
    patronus_path: str | Path | None = None,
    business_path: str | Path | None = None,
    enron_path: str | Path | None = None,
    kaggle_path: str | Path | None = None,
    out_csv: Path | str | None = None,
) -> list[dict]:
    """
    Load all real sources (Patronus, Enron, BigCode, Kaggle, business_real), merge, dedupe, filter length,
    save to reports/real_candidate_pool.csv. Returns merged list of {text, source, risk, sap_id}.
    """
    def _load_patronus():
        try:
            return load_patronus_pii(patronus_path)
        except FileNotFoundError:
            return []

    source_lists = [
        ("patronus", _load_patronus()),
        ("enron_real", load_enron_real(enron_path)),
        ("bigcode", load_bigcode_pii()),
        ("kaggle_sensitive", load_kaggle_sensitive(kaggle_path)),
        ("business_real", load_business_real(business_path)),
    ]
    merged, _ = _build_real_pool_from_sources(source_lists, out_csv=out_csv)
    return merged


def _merge_and_dedupe_sources(
    source_lists: list[tuple[str, list[dict]]],
) -> list[dict]:
    """
    Merge rows from multiple (name, rows) lists and deduplicate by normalized text.
    Returns list of {text, risk, sap_id}; first occurrence wins. Only includes rows with risk set.
    """
    seen = set()
    out = []
    for name, rows in source_lists:
        for r in rows:
            if r.get("risk") is None:
                continue
            text = r.get("text") or ""
            if not text.strip():
                continue
            norm = text.strip().lower()
            if norm in seen:
                continue
            seen.add(norm)
            out.append({"text": text.strip(), "risk": r["risk"], "sap_id": r.get("sap_id", f"merged-{len(out)}"), "source": r.get("source", name)})
    return out


def patronus_to_risk_rows(
    rows: list[dict],
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Build Patronus rows with text/risk/is_sap and split into train/val/test when enough data.
    When too few rows (< 20 or min class count < 2), return all as train augmentation (val/test empty).
    """
    raw_count = len(rows)
    patronus_rows = [
        {"text": r["text"], "risk": r["risk"], "is_sap": 0, "sap_id": r.get("sap_id", f"patronus-{i}")}
        for i, r in enumerate(rows)
    ]
    usable_count = len(patronus_rows)
    risk_counts = Counter(r["risk"] for r in patronus_rows)
    min_class = min(risk_counts.values()) if risk_counts else 0

    print(f"[Patronus] raw row count: {raw_count}")
    print(f"[Patronus] usable mapped row count: {usable_count}")
    print(f"[Patronus] risk counts: {dict(risk_counts)}")
    print("[Patronus] first 5 mapped rows (risk + first 120 chars of text):")
    for r in patronus_rows[:5]:
        text_preview = (r["text"] or "")[:120]
        print(f"  risk={r['risk']!r}  text={text_preview!r}")

    if not patronus_rows:
        return [], [], []

    if usable_count < 20 or min_class < 2:
        print("Too few Patronus rows for reliable 3-way stratified split; using all Patronus rows as train augmentation.")
        return patronus_rows, [], []

    return stratified_split(patronus_rows, SPLIT_RATIOS)


def _sap_id(fake: Faker) -> str:
    """Generate a synthetic SAP-style ID (e.g. numeric or alphanumeric)."""
    return f"SAP-{fake.random_number(digits=10, fix_len=True)}"


def _finance_text(fake: Faker, risk: str) -> str:
    """Generate short finance-like text; risk influences content style."""
    company = fake.company()
    amount = fake.random_int(1000, 999999)
    iban_like = fake.iban() if hasattr(fake, "iban") else fake.bban()
    if risk == "low":
        parts = [
            f"Payment of ${amount:,} to {company}.",
            f"Reference: invoice {fake.random_number(digits=8)}.",
        ]
    elif risk == "med":
        parts = [
            f"Transfer {amount} USD to account ending {fake.random_number(digits=4)}.",
            f"Beneficiary: {company}. Ref: {fake.uuid4()[:8]}.",
        ]
    else:
        parts = [
            f"Wire {amount} to {iban_like}. Beneficiary {company}.",
            f"Routing: {fake.random_number(digits=9)}. Confirmation required.",
        ]
    return " ".join(parts)


def generate_faker_synthetics(n: int = NUM_SYNTHETICS) -> list[dict]:
    """
    Generate n synthetic samples with SAP IDs and finance text.
    Risk distribution: 30% low, 40% med, 30% high.
    """
    fake = Faker()
    Faker.seed(42)
    random.seed(42)

    n_low = int(n * RISK_DIST["low"])
    n_med = int(n * RISK_DIST["med"])
    n_high = n - n_low - n_med

    rows = []
    for risk, count in [("low", n_low), ("med", n_med), ("high", n_high)]:
        for _ in range(count):
            sap = _sap_id(fake)
            text = _finance_text(fake, risk)
            rows.append({
                "sap_id": sap,
                "text": f"{sap} | {text}",
                "risk": risk,
            })

    random.shuffle(rows)
    return rows


def generate_low_risk_samples(n: int = 3000, seed: int = 42) -> list[dict]:
    """Synthetic clearly non-sensitive text for the low-risk class."""
    import random as _random
    from faker import Faker as _Faker

    _random.seed(seed)
    fake = _Faker()
    _Faker.seed(seed)

    samples = []
    templates = [
        lambda: fake.sentence(nb_words=_random.randint(6, 18)),
        lambda: f"The meeting is on {fake.day_of_week()} at {fake.time()}.",
        lambda: f"Please review the {fake.file_name()} document.",
        lambda: f"{fake.company()} reported strong results this quarter.",
        lambda: fake.paragraph(nb_sentences=2),
        lambda: f"The project deadline is {fake.month_name()} {fake.day_of_month()}.",
        lambda: f"Reminder: {fake.bs()}.",
        lambda: f"Today's agenda includes {fake.catch_phrase().lower()}.",
        lambda: f"The report covers {fake.word()} and {fake.word()} metrics.",
        lambda: f"Team update: {fake.sentence(nb_words=10)}",
    ]

    for _ in range(n):
        text = _random.choice(templates)()
        samples.append({"text": str(text), "risk": "low"})

    return samples


def balance_dataset(
    texts: list,
    labels: list,
    caps: dict | None = None,
    seed: int = 42,
) -> tuple[list, list]:
    """Cap per-class counts, shuffle, return parallel lists. Labels are risk strings (low/med/high)."""
    import random as _random
    from collections import defaultdict

    if caps is None:
        caps = {"high": 6000, "med": 3000, "low": 3000}
    _random.seed(seed)
    grouped: dict = defaultdict(list)
    for t, lab in zip(texts, labels):
        grouped[lab].append(t)

    out_texts: list = []
    out_labels: list = []
    for label, cap in caps.items():
        items = grouped.get(label, [])
        if len(items) > cap:
            items = _random.sample(items, cap)
        out_texts.extend(items)
        out_labels.extend([label] * len(items))

    combined = list(zip(out_texts, out_labels))
    _random.shuffle(combined)
    if not combined:
        return [], []
    out_texts, out_labels = zip(*combined)
    return list(out_texts), list(out_labels)


def balance_train_risk_classes(train_rows: list[dict]) -> list[dict]:
    """
    Append synthetic low-risk samples, cap classes, shuffle (training split only).
    """
    texts = [r["text"] for r in train_rows]
    labels = [r["risk"] for r in train_rows]
    for s in generate_low_risk_samples(3000):
        texts.append(s["text"])
        labels.append(s["risk"])
    texts, labels = balance_dataset(
        texts,
        labels,
        caps={"high": 6000, "med": 3000, "low": 3000},
        seed=42,
    )
    print("[data] balanced distribution:", Counter(labels))
    return [{"text": t, "risk": lab, "sap_id": ""} for t, lab in zip(texts, labels)]


def stratified_split(rows: list[dict], ratios: tuple[float, float, float] = SPLIT_RATIOS):
    """Split rows into train/val/test with stratification on risk when possible."""
    train_ratio, val_ratio, test_ratio = ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    random.seed(42)
    rows = list(rows)
    random.shuffle(rows)

    texts = [r["text"] for r in rows]
    risks = [r["risk"] for r in rows]
    sap_ids = [r.get("sap_id", "") for r in rows]

    # Stratify only if every class has at least 2 samples (sklearn requirement)
    min_count = min(Counter(risks).values()) if risks else 0
    use_stratify = min_count >= 2

    if use_stratify:
        train_texts, rest_texts, train_risks, rest_risks, train_saps, rest_saps = train_test_split(
            texts, risks, sap_ids, test_size=(1 - train_ratio), stratify=risks, random_state=42
        )
        val_frac = val_ratio / (val_ratio + test_ratio)
        val_texts, test_texts, val_risks, test_risks, val_saps, test_saps = train_test_split(
            rest_texts, rest_risks, rest_saps, test_size=(1 - val_frac), stratify=rest_risks, random_state=42
        )
    else:
        n = len(rows)
        n_train = max(1, int(n * train_ratio))
        n_val = max(0, int(n * val_ratio))
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_val = n - n_train
        train_texts, train_risks, train_saps = texts[:n_train], risks[:n_train], sap_ids[:n_train]
        val_texts, val_risks, val_saps = texts[n_train : n_train + n_val], risks[n_train : n_train + n_val], sap_ids[n_train : n_train + n_val]
        test_texts, test_risks, test_saps = texts[n_train + n_val :], risks[n_train + n_val :], sap_ids[n_train + n_val :]

    return (
        [{"text": t, "risk": r, "sap_id": s} for t, r, s in zip(train_texts, train_risks, train_saps)],
        [{"text": t, "risk": r, "sap_id": s} for t, r, s in zip(val_texts, val_risks, val_saps)],
        [{"text": t, "risk": r, "sap_id": s} for t, r, s in zip(test_texts, test_risks, test_saps)],
    )


RISK_TO_ID = {"low": 0, "med": 1, "high": 2}


def tokenize_dataset(ds: Dataset, tokenizer, text_column: str = "text", max_length: int = 128) -> Dataset:
    """Tokenize text column with TinyBERT tokenizer; keep risk and label columns."""

    def tokenize(examples):
        out = tokenizer(
            examples[text_column],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors=None,
        )
        return out

    cols_to_remove = [c for c in ds.column_names if c != "risk" and c != "label"]
    return ds.map(tokenize, batched=True, remove_columns=cols_to_remove, desc="Tokenizing")


def _print_split_summary(train_rows: list, val_rows: list, test_rows: list, prefix: str = ""):
    """Print split sizes and per-class counts for train/val/test."""
    for name, rlist in [("train", train_rows), ("validation", val_rows), ("test", test_rows)]:
        n = len(rlist)
        counts = dict(Counter(r["risk"] for r in rlist)) if rlist else {}
        print(f"{prefix}{name}: n={n}  per_class={counts}")


def build_and_save(
    arrow_dir: str = ARROW_SAVE_DIR,
    tinybert_id: str = TINYBERT_ID,
    max_length: int = 128,
    data_source: str = "nemotron",
    patronus_path: str | Path | None = None,
    business_path: str | Path | None = None,
    balance_pii_max: int | None = None,
    balance_pii_min: int | None = None,
) -> DatasetDict:
    """
    Build train/val/test and save tokenized Arrow datasets.
    data_source: "nemotron", "faker", "patronus", "patronus_synthetic", "both", "multi_real_synthetic", "ai4privacy_*".
    For multi_real_synthetic, optional GUARDRAIL_FINANCIAL_PII_PATH loads Training_Set.xlsx / Testing_Set.xlsx (see load_financial_pii_xlsx).
    balance_pii_max/min: for ai4privacy modes, cap/oversample per class (e.g. max 5000, min 200).
    """
    nemotron = None
    if data_source in ("nemotron", "both", "patronus"):
        nemotron = load_nemotron_pii()

    if data_source == "nemotron":
        train_rows, val_rows, test_rows = nemotron_to_risk_rows(nemotron)
    elif data_source == "faker":
        rows = generate_faker_synthetics(NUM_SYNTHETICS)
        train_rows, val_rows, test_rows = stratified_split(rows)
    elif data_source == "patronus":
        base_train, base_val, base_test = nemotron_to_risk_rows(nemotron)
        rows = load_patronus_pii(patronus_path)
        if not rows:
            path = Path(patronus_path or PATRONUS_PII_PATH)
            raise ValueError(
                f"No Patronus/EnterprisePII rows loaded from {path}. "
                "Add JSON/JSONL/CSV files with 'text' (or content/excerpt) and 'sensitivity' (or label/risk). "
                "See data/patronus/README.md and https://github.com/mosaicml/llm-foundry for EnterprisePII."
            )
        p_train, p_val, p_test = patronus_to_risk_rows(rows)
        if not p_val and not p_test:
            train_rows = base_train + p_train
            val_rows = base_val
            test_rows = base_test
        else:
            train_rows = base_train + p_train
            val_rows = base_val + p_val
            test_rows = base_test + p_test
        random.shuffle(train_rows)
        if val_rows:
            random.shuffle(val_rows)
        if test_rows:
            random.shuffle(test_rows)
    elif data_source == "both":
        n_train, n_val, n_test = nemotron_to_risk_rows(nemotron)
        f_rows = generate_faker_synthetics(NUM_SYNTHETICS)
        f_train, f_val, f_test = stratified_split(f_rows)
        train_rows = n_train + f_train
        val_rows = n_val + f_val
        test_rows = n_test + f_test
        random.shuffle(train_rows)
        random.shuffle(val_rows)
        random.shuffle(test_rows)
    elif data_source == "patronus_synthetic":
        # Training = Patronus + synthetic; val = split from pool if enough data; test = [] (use manual_eval for final eval)
        patronus_raw = load_patronus_pii(patronus_path)
        patronus_rows = [
            {"text": r["text"], "risk": r["risk"], "is_sap": 0, "sap_id": r.get("sap_id", f"p-{i}")}
            for i, r in enumerate(patronus_raw)
        ]
        synthetic_rows = generate_faker_synthetics(NUM_SYNTHETICS)
        pool = patronus_rows + synthetic_rows
        random.seed(42)
        random.shuffle(pool)
        n_patronus = len(patronus_rows)
        n_synthetic = len(synthetic_rows)
        pool_risks = [r["risk"] for r in pool]
        risk_counts_pool = Counter(pool_risks)
        min_class = min(risk_counts_pool.values()) if risk_counts_pool else 0
        if len(pool) < 20 or min_class < 2:
            train_rows = pool
            val_rows = []
            test_rows = []
            print("WARNING: Too few rows or a class with < 2 examples; validation set is empty.")
        else:
            texts = [r["text"] for r in pool]
            saps = [r.get("sap_id", "") for r in pool]
            train_texts, val_texts, train_risks, val_risks, train_saps, val_saps = train_test_split(
                texts, pool_risks, saps, test_size=0.15, stratify=pool_risks, random_state=42
            )
            train_rows = [{"text": t, "risk": r, "sap_id": s} for t, r, s in zip(train_texts, train_risks, train_saps)]
            val_rows = [{"text": t, "risk": r, "sap_id": s} for t, r, s in zip(val_texts, val_risks, val_saps)]
            test_rows = []
        print(f"Patronus rows: {n_patronus}")
        print(f"Synthetic rows: {n_synthetic}")
        print(f"Merged training pool (before val split): {len(pool)}")
        print(f"Validation size: {len(val_rows)}")
        print("Per-class counts — train:", dict(Counter(r["risk"] for r in train_rows)))
        print("Per-class counts — validation:", dict(Counter(r["risk"] for r in val_rows)))
        if val_rows and (len(val_rows) < 10 or min(Counter(r["risk"] for r in val_rows).values()) < 2):
            print("WARNING: Validation set is very small or has a class with < 2 examples.")
    elif data_source == "multi_real_synthetic":
        # Label-aware: only labeled real data for training; unlabeled go to candidate pool. Synthetic capped.
        def _load_patronus():
            try:
                return load_patronus_pii(patronus_path)
            except FileNotFoundError:
                return []
        source_lists = [
            ("patronus", _load_patronus()),
            ("enron_real", load_enron_real()),
            ("bigcode", load_bigcode_pii()),
            ("kaggle_sensitive", load_kaggle_sensitive()),
            ("business_real", load_business_real(business_path)),
        ]
        _, real_labeled = _build_real_pool_from_sources(source_lists)

        real_labeled_rows = [
            {"text": r["text"], "risk": r["risk"], "is_sap": 0, "sap_id": r["sap_id"], "source": r.get("source")}
            for r in real_labeled
        ]
        fin_texts, fin_labels = load_financial_pii_xlsx()
        for i, (t, r) in enumerate(zip(fin_texts, fin_labels)):
            real_labeled_rows.append(
                {
                    "text": t,
                    "risk": r,
                    "is_sap": 0,
                    "sap_id": f"finxlsx-{i}",
                    "source": "financial_pii_xlsx",
                }
            )
        uf_list: list = []
        try:
            from feedback_store import load_user_feedback_export_rows

            uf_list = load_user_feedback_export_rows()
        except Exception as ex:
            print(f"[user_feedback] could not load export.jsonl: {ex}")
        for i, r in enumerate(uf_list):
            real_labeled_rows.append(
                {
                    "text": r["text"],
                    "risk": r["risk"],
                    "is_sap": 0,
                    "sap_id": f"ufb-{i}",
                    "source": "user_feedback",
                }
            )
        if uf_list:
            print(f"[user_feedback] merged {len(uf_list)} labeled rows from data/user_feedback/export.jsonl")
        n_real_labeled = len(real_labeled_rows)
        n_real_unlabeled = sum(len(rows) for _, rows in source_lists) - n_real_labeled  # approx; pool has dedup/filter
        cap = max(0, SYNTHETIC_CAP_MULTIPLIER * n_real_labeled) if n_real_labeled else NUM_SYNTHETICS
        n_synthetic_target = min(NUM_SYNTHETICS, cap) if n_real_labeled else NUM_SYNTHETICS
        _syn_raw = generate_faker_synthetics(n_synthetic_target)
        synthetic_rows = [
            {"text": r["text"], "risk": r["risk"], "is_sap": 0, "sap_id": r["sap_id"], "source": "synthetic"}
            for r in _syn_raw
        ]
        pool = real_labeled_rows + synthetic_rows
        random.seed(42)
        random.shuffle(pool)
        pool_risks = [r["risk"] for r in pool]
        risk_counts_pool = Counter(pool_risks)
        min_class = min(risk_counts_pool.values()) if risk_counts_pool else 0
        if len(pool) < 20 or min_class < 2:
            train_rows = pool
            val_rows = []
            test_rows = []
            print("WARNING: Too few rows or a class with < 2 examples; validation set is empty.")
        else:
            pool_sources = [r.get("source", "?") for r in pool]
            try:
                train_texts, val_texts, train_risks, val_risks, train_saps, val_saps, train_srcs, val_srcs = train_test_split(
                    [r["text"] for r in pool], pool_risks, [r.get("sap_id", "") for r in pool], pool_sources,
                    test_size=0.15, stratify=pool_risks, random_state=42
                )
            except ValueError:
                train_texts, val_texts, train_risks, val_risks, train_saps, val_saps, train_srcs, val_srcs = train_test_split(
                    [r["text"] for r in pool], pool_risks, [r.get("sap_id", "") for r in pool], pool_sources,
                    test_size=0.15, random_state=42
                )
            train_rows = [{"text": t, "risk": r, "sap_id": s, "source": src} for t, r, s, src in zip(train_texts, train_risks, train_saps, train_srcs)]
            val_rows = [{"text": t, "risk": r, "sap_id": s, "source": src} for t, r, s, src in zip(val_texts, val_risks, val_saps, val_srcs)]
            test_rows = []

        # Manual eval overlap warning
        try:
            manual_rows = load_manual_eval()
            manual_norms = {r["text"].strip().lower() for r in manual_rows}
            train_norms = {r["text"].strip().lower() for r in train_rows}
            val_norms = {r["text"].strip().lower() for r in val_rows}
            overlap_train = manual_norms & train_norms
            overlap_val = manual_norms & val_norms
            if overlap_train or overlap_val:
                print("WARNING: manual_eval text appears in training or validation. Do not use manual_eval as training data.")
                if overlap_train:
                    print(f"  Overlap with train: {len(overlap_train)} normalized text(s)")
                if overlap_val:
                    print(f"  Overlap with validation: {len(overlap_val)} normalized text(s)")
        except Exception:
            pass

        print(f"[multi_real_synthetic] Real labeled count: {n_real_labeled}")
        print(f"[multi_real_synthetic] Real unlabeled (in pool): (see pool CSV)")
        print(f"[multi_real_synthetic] Synthetic count: {len(synthetic_rows)} (cap: {n_synthetic_target})")
        print(f"[multi_real_synthetic] Final train count: {len(train_rows)}")
        print(f"[multi_real_synthetic] Final validation count: {len(val_rows)}")
        print("Per-class counts — train:", dict(Counter(r["risk"] for r in train_rows)))
        print("Per-class counts — validation:", dict(Counter(r["risk"] for r in val_rows)))
        print("Per-source counts — train:", dict(Counter(r.get("source", "?") for r in train_rows)))
        # Save manifest for dataset_sources_report
        try:
            _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            manifest = {
                "real_labeled_count": n_real_labeled,
                "synthetic_count": len(synthetic_rows),
                "train_count": len(train_rows),
                "val_count": len(val_rows),
                "train_per_source": dict(Counter(r.get("source", "?") for r in train_rows)),
                "val_per_source": dict(Counter(r.get("source", "?") for r in val_rows)),
            }
            with open(_REPORTS_DIR / "train_val_manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        except Exception:
            pass
    elif data_source == "ai4privacy_text_only":
        rows = load_ai4privacy_pii(max_train=35000, max_test=5000)
        if not rows:
            raise ValueError("ai4privacy_text_only: no data loaded from ai4privacy/pii-masking-43k.")
        if balance_pii_max is not None or balance_pii_min is not None:
            rows = _rebalance_pii_rows(rows, max_per_class=balance_pii_max, min_per_class=balance_pii_min)
        train_rows, val_rows, test_rows = _stratified_split_by_label(rows, label_key="label", ratios=(0.85, 0.15, 0.0))
        print(f"[ai4privacy_text_only] train: {len(train_rows)}, val: {len(val_rows)}, test: {len(test_rows)}")
        print("Per-class (collapsed) — train:", dict(Counter(r["risk"] for r in train_rows)))
        print("Per-class (collapsed) — val:", dict(Counter(r["risk"] for r in val_rows)))
    elif data_source == "ai4privacy_kaggle_en":
        rows = load_ai4privacy_kaggle_en(path=None, max_train=35000, max_test=5000)
        if not rows:
            raise ValueError(
                "ai4privacy_kaggle_en: no English data. Download from "
                "https://www.kaggle.com/datasets/verracodeguacas/ai4privacy-pii/data and extract to data/ai4privacy_kaggle"
            )
        if balance_pii_max is not None or balance_pii_min is not None:
            rows = _rebalance_pii_rows(rows, max_per_class=balance_pii_max, min_per_class=balance_pii_min)
        train_rows, val_rows, test_rows = _stratified_split_by_label(rows, label_key="label", ratios=(0.85, 0.15, 0.0))
        print(f"[ai4privacy_kaggle_en] train: {len(train_rows)}, val: {len(val_rows)}, test: {len(test_rows)}")
        print("Per-class (collapsed) — train:", dict(Counter(r["risk"] for r in train_rows)))
        print("Per-class (collapsed) — val:", dict(Counter(r["risk"] for r in val_rows)))
    else:
        raise ValueError(
            f"data_source must be 'nemotron', 'faker', 'patronus', 'patronus_synthetic', 'both', "
            f"'multi_real_synthetic', 'ai4privacy_text_only', 'ai4privacy_text_plus_real', or 'ai4privacy_kaggle_en'; "
            f"got {data_source!r}"
        )

    # Synthetic low + caps on med/high for training; inject low-risk rows into val/test for evaluation
    if data_source not in ("ai4privacy_text_only", "ai4privacy_text_plus_real", "ai4privacy_kaggle_en"):
        train_rows = balance_train_risk_classes(train_rows)
        low_test = generate_low_risk_samples(n=500, seed=99)
        test_rows.extend(low_test)
        val_rows.extend(generate_low_risk_samples(n=200, seed=77))
        for split_name, rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
            dist = Counter(r["risk"] for r in rows)
            print(f"[data] {split_name} distribution: {dict(dist)}")

    print("Final merged split sizes and per-class counts:")
    _print_split_summary(train_rows, val_rows, test_rows, prefix="  ")

    use_risk_labels = data_source not in ("ai4privacy_text_only", "ai4privacy_text_plus_real", "ai4privacy_kaggle_en")
    if use_risk_labels:
        for row in train_rows + val_rows + test_rows:
            row["label"] = RISK_TO_ID[row["risk"]]

    train_ds = Dataset.from_list(train_rows)
    val_ds = Dataset.from_list(val_rows)
    test_ds = Dataset.from_list(test_rows)

    tokenizer = AutoTokenizer.from_pretrained(tinybert_id)

    train_ds = tokenize_dataset(train_ds, tokenizer, max_length=max_length)
    val_ds = tokenize_dataset(val_ds, tokenizer, max_length=max_length)
    test_ds = tokenize_dataset(test_ds, tokenizer, max_length=max_length)

    combined = DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds,
    })

    out_path = Path(arrow_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    combined.save_to_disk(str(out_path))

    if data_source in ("ai4privacy_text_only", "ai4privacy_text_plus_real", "ai4privacy_kaggle_en"):
        label_config = {
            "num_labels": len(COLLAPSED_PII_LABELS),
            "id2label": PII_ID2LABEL,
            "label2id": PII_LABEL2ID,
            "source": data_source,
        }
        with open(out_path / LABEL_CONFIG_FILENAME, "w", encoding="utf-8") as f:
            json.dump(label_config, f, indent=2)
        print(f"Wrote {out_path / LABEL_CONFIG_FILENAME} (num_labels={label_config['num_labels']})")

    return combined, nemotron


if __name__ == "__main__":
    import os
    import sys
    data_source = sys.argv[1] if len(sys.argv) > 1 else "nemotron"
    patronus_path = sys.argv[2] if len(sys.argv) > 2 else None
    business_path = sys.argv[3] if len(sys.argv) > 3 else None
    balance_max = os.environ.get("BALANCE_PII_MAX")
    balance_min = os.environ.get("BALANCE_PII_MIN")
    combined, nemotron = build_and_save(
        data_source=data_source,
        patronus_path=patronus_path or None,
        business_path=business_path or None,
        balance_pii_max=int(balance_max) if balance_max is not None and balance_max.isdigit() else None,
        balance_pii_min=int(balance_min) if balance_min is not None and balance_min.isdigit() else None,
    )
    print("Data source:", data_source)
    if nemotron:
        print("Nemotron-PII splits (raw):", {k: v.num_rows for k, v in nemotron.items()})
    print("Saved tokenized Arrow datasets to:", ARROW_SAVE_DIR)
    print("Train/val/test sizes:", combined["train"].num_rows, combined["validation"].num_rows, combined["test"].num_rows)
