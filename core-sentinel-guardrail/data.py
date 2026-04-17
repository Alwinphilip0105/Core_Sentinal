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
import math
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
DEFAULT_TRAIN_CAPS = {"high": 5000, "med": 5000, "low": 3000}
DEFAULT_HIGH_TO_MED_MAX_RATIO = 6.0
DEFAULT_EVAL_MAX_CLASS_RATIO = 3.0
DEFAULT_VAL_MIN_COUNTS = {"low": 30, "med": 60, "high": 30}
DEFAULT_TEST_MIN_COUNTS = {"low": 40, "med": 120, "high": 40}
DEFAULT_HOLDOUT_MIN_COUNTS = {"low": 20, "med": 20, "high": 20}
# Train floors: real high is often tiny after pool splits; top up with hard synthetic high (capped).
DEFAULT_TRAIN_MIN_COUNTS = {"low": 600, "med": 1400, "high": 400}
TRAIN_SYNTHETIC_HIGH_TOPUP_CAP = 600
# Optional JSONL pools under data/extra_pools/ (see README.txt there). Not required for training.
EXTRA_POOL_MAX_PER_FILE = 8000
# Max rows read from data/enron_real/enron_pii_prompts.jsonl (memory / build time).
DEFAULT_ENRON_PII_PROMPTS_MAX_ROWS = 20000
def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(str(raw).strip())
    except ValueError:
        return default


def _get_num_synthetics() -> int:
    """Target size for Faker synthetic pool (env GUARDRAIL_NUM_SYNTHETICS, default NUM_SYNTHETICS)."""
    return max(0, _env_int("GUARDRAIL_NUM_SYNTHETICS", NUM_SYNTHETICS))


def _get_synthetic_cap_multiplier() -> int:
    """Synthetic rows <= multiplier * real_labeled in multi_real_synthetic (GUARDRAIL_SYNTHETIC_CAP_MULTIPLIER)."""
    return max(0, _env_int("GUARDRAIL_SYNTHETIC_CAP_MULTIPLIER", SYNTHETIC_CAP_MULTIPLIER))


def _get_binary_safe_pool_target() -> int:
    """
    Target safe rows in binary trainable pool before split.
    Helps avoid severe safe-class undercoverage from real-source skew.
    """
    return max(0, _env_int("GUARDRAIL_BINARY_SAFE_POOL_TARGET", 5000))


def _get_binary_synthetic_floor() -> int:
    """
    Optional floor for synthetic pool in binary mode.
    Set GUARDRAIL_BINARY_SYNTHETIC_FLOOR to raise data volume in one experiment.
    """
    return max(0, _env_int("GUARDRAIL_BINARY_SYNTHETIC_FLOOR", 0))


def _get_binary_adv_risky_samples() -> int:
    """
    Number of adversarial risky synthetic samples injected into binary pool pre-split.
    """
    return max(0, _env_int("GUARDRAIL_BINARY_ADV_RISKY_SAMPLES", 800))


def _get_split_ratios() -> tuple[float, float, float]:
    """
    Train/val/test fractions for stratified_split when no source holdout (default SPLIT_RATIOS).
    Override: GUARDRAIL_SPLIT_RATIOS=\"0.7,0.15,0.15\" (must sum to ~1).
    """
    raw = os.environ.get("GUARDRAIL_SPLIT_RATIOS", "").strip()
    if not raw:
        return SPLIT_RATIOS
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 3:
        print(f"[data] GUARDRAIL_SPLIT_RATIOS must have 3 comma-separated floats; using default {SPLIT_RATIOS}")
        return SPLIT_RATIOS
    try:
        a, b, c = float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        return SPLIT_RATIOS
    s = a + b + c
    if s <= 0 or abs(s - 1.0) > 0.02:
        print(f"[data] GUARDRAIL_SPLIT_RATIOS must sum to ~1.0; got {s}, using default {SPLIT_RATIOS}")
        return SPLIT_RATIOS
    if abs(s - 1.0) > 1e-9:
        a, b, c = a / s, b / s, c / s
    return (a, b, c)


def _get_pool_val_fraction() -> float:
    """Fraction of pool for validation when splitting train+val from merged pool (GUARDRAIL_POOL_VAL_FRACTION)."""
    v = _env_float("GUARDRAIL_POOL_VAL_FRACTION", 0.15)
    if v <= 0.0 or v >= 1.0:
        print("[data] GUARDRAIL_POOL_VAL_FRACTION must be in (0,1); using 0.15")
        return 0.15
    return v


def _get_train_caps() -> dict[str, int]:
    """Per-class caps after balance_train_risk_classes (GUARDRAIL_TRAIN_CAP_LOW/MED/HIGH)."""
    return {
        "low": max(0, _env_int("GUARDRAIL_TRAIN_CAP_LOW", DEFAULT_TRAIN_CAPS["low"])),
        "med": max(0, _env_int("GUARDRAIL_TRAIN_CAP_MED", DEFAULT_TRAIN_CAPS["med"])),
        "high": max(0, _env_int("GUARDRAIL_TRAIN_CAP_HIGH", DEFAULT_TRAIN_CAPS["high"])),
    }


def _get_bigcode_max_rows() -> int | None:
    """
    Row cap for BigCode PII (GUARDRAIL_BIGCODE_MAX_ROWS). Use 'none', 'all', or '-1' for no cap.
    """
    raw = os.environ.get("GUARDRAIL_BIGCODE_MAX_ROWS")
    if raw is None or str(raw).strip() == "":
        return BIGCODE_MAX_ROWS
    s = str(raw).strip().lower()
    if s in ("none", "all", "-1"):
        return None
    try:
        v = int(s)
        return None if v <= 0 else v
    except ValueError:
        return BIGCODE_MAX_ROWS


def _use_multi_real_equal_thirds() -> bool:
    """
    Legacy opt-in for forcing multi_real_synthetic train data to ~33/33/33.
    Disabled by default because it can distort realistic med/high boundaries.
    """
    raw = os.environ.get("GUARDRAIL_MULTI_REAL_EQUAL_THIRDS")
    if raw is None:
        return False
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _binary_mode_enabled() -> bool:
    """Binary guardrail mode (safe/risky) for multi_real_synthetic; default off."""
    raw = os.environ.get("GUARDRAIL_BINARY_MODE")
    if raw is None:
        return False
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _to_binary_risk(risk: str) -> str:
    """Map low/med/high -> safe/risky."""
    r = str(risk or "").strip().lower()
    return "safe" if r == "low" else "risky"


def _rebalance_binary_train_equal_halves(
    train_rows: list[dict],
    *,
    seed: int = 91,
    max_per_class: int = 5000,
) -> list[dict]:
    """
    Rebalance binary train split to 50/50 safe vs risky.
    Target is min(max(safe_count, risky_count), max_per_class) so we upsample minority
    and avoid collapsing the train set to the smaller class.
    """
    rng = random.Random(seed)
    safe_rows = [r for r in train_rows if str(r.get("risk", "")).lower() == "safe"]
    risky_rows = [r for r in train_rows if str(r.get("risk", "")).lower() == "risky"]
    safe_count = len(safe_rows)
    risky_count = len(risky_rows)
    if not safe_rows or not risky_rows:
        print(
            "[data] Binary mode rebalance skipped (missing class): "
            f"safe={safe_count} risky={risky_count}"
        )
        return train_rows
    target = min(max(safe_count, risky_count), int(max_per_class))
    print(
        "[data] Binary mode pre-rebalance train pool: "
        f"safe={safe_count} risky={risky_count} target_per_class={target} cap={int(max_per_class)}"
    )
    safe_pick = rng.sample(safe_rows, target) if safe_count >= target else rng.choices(safe_rows, k=target)
    risky_pick = rng.sample(risky_rows, target) if risky_count >= target else rng.choices(risky_rows, k=target)
    out = [dict(r) for r in safe_pick] + [dict(r) for r in risky_pick]
    rng.shuffle(out)
    return out


def _rebalance_multi_real_train_equal_thirds(train_rows: list[dict], *, seed: int = 88) -> list[dict]:
    """
    Rebalance training data to ~33% low / med / high.

    Target count T is the medium-class count (before this step). High is capped at T
    (random subsample if there are more). Low and med are brought up to T by oversampling
    with replacement when needed.
    """
    rng = random.Random(seed)
    by_risk: dict[str, list[dict]] = {"low": [], "med": [], "high": []}
    for r in train_rows:
        rk = str(r.get("risk", "")).lower()
        if rk in by_risk:
            by_risk[rk].append(r)

    n_low, n_med, n_high = len(by_risk["low"]), len(by_risk["med"]), len(by_risk["high"])
    if n_low + n_med + n_high == 0:
        return train_rows

    T = n_med
    if T == 0:
        if n_low > 0 and n_high > 0:
            T = min(n_low, n_high)
        else:
            T = max(n_low, n_high, 0)
        if T <= 0:
            print("[multi_real_synthetic] equal-thirds: no rows to balance; skipping")
            return train_rows
        print(f"[multi_real_synthetic] equal-thirds: med was 0; using T={T} from low/high")

    def _pick(pool: list[dict], k: int, risk: str, sub_seed: int) -> list[dict]:
        if k <= 0:
            return []
        if not pool:
            return _risk_topup_samples(risk, k, seed=sub_seed)
        if len(pool) >= k:
            chosen = rng.sample(pool, k)
        else:
            chosen = [dict(r) for r in rng.choices(pool, k=k)]
        out: list[dict] = []
        for i, row in enumerate(chosen):
            nr = dict(row)
            nr["risk"] = risk
            sid = str(nr.get("sap_id", "") or "")[:24]
            nr["sap_id"] = f"eq3-{risk}-{sub_seed}-{i}-{sid}"
            out.append(nr)
        return out

    low_rows = _pick(by_risk["low"], T, "low", seed + 11)
    med_rows = _pick(by_risk["med"], T, "med", seed + 22)
    high_rows = _pick(by_risk["high"], T, "high", seed + 33)

    merged = low_rows + med_rows + high_rows
    rng.shuffle(merged)
    print(
        f"[multi_real_synthetic] equal-thirds rebalance: T={T} per class "
        f"(was low/med/high={n_low}/{n_med}/{n_high}) -> train n={len(merged)} "
        f"(~33% each)"
    )
    return merged


def _normalize_for_leakage(text: str) -> str:
    """
    Canonicalized fingerprint used for dedupe/leakage reduction across near-identical templates.
    Variable identifiers (numbers, IDs, emails, phones) are collapsed to placeholder tokens.
    """
    t = str(text or "").strip().lower()
    if not t:
        return ""
    t = re.sub(r"\b[0-9a-f]{8,}\b", "<hex>", t, flags=re.IGNORECASE)
    t = re.sub(r"\b[A-Z]{2,5}[-_ ]?\d{3,}\b", "<id>", t, flags=re.IGNORECASE)
    t = re.sub(r"\b\d{2,}\b", "<num>", t)
    t = re.sub(r"\b[0-9a-z._%+-]+@[0-9a-z.-]+\.[a-z]{2,}\b", "<email>", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(?:\+?\d[\d\-\s().]{7,}\d)\b", "<phone>", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def load_extra_pool_rows_for_training(existing_norms: set[str]) -> tuple[list[dict], int]:
    """
    Load optional curated JSONL files from data/extra_pools/ (or GUARDRAIL_EXTRA_POOL_DIR):
      - high_extra.jsonl          -> risk high (extra real-like high-risk lines)
      - hard_negative_low.jsonl   -> risk low (benign text that should not be scored as high)
      - hard_negative_med.jsonl   -> risk med (boundary medium vs high)

    Each line: {"text": "..."} (risk is implied by filename). Skips duplicates vs existing_norms
    and within files (same leakage-normalized fingerprint).

    Returns (rows ready to append to real_labeled_rows, duplicate_skip_count).
    """
    raw_dir = os.environ.get("GUARDRAIL_EXTRA_POOL_DIR", "").strip()
    base = Path(raw_dir) if raw_dir else (Path(__file__).resolve().parent / "data" / "extra_pools")
    if not base.is_dir():
        return [], 0

    max_per = _env_int("GUARDRAIL_EXTRA_POOL_MAX_PER_FILE", EXTRA_POOL_MAX_PER_FILE)
    specs = [
        ("high_extra.jsonl", "high", "extra_pool_high"),
        ("hard_negative_low.jsonl", "low", "extra_pool_hard_low"),
        ("hard_negative_med.jsonl", "med", "extra_pool_hard_med"),
    ]
    out: list[dict] = []
    skipped = 0
    local_seen: set[str] = set(existing_norms)

    for fname, risk, src in specs:
        path = base / fname
        if not path.is_file():
            continue
        n_from_file = 0
        with open(path, encoding="utf-8-sig") as f:
            for line_no, line in enumerate(f, start=1):
                if n_from_file >= max_per:
                    print(f"[extra_pools] {fname}: reached cap ({max_per} lines per file)")
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[extra_pools] {fname}:{line_no}: invalid JSON, skipped")
                    continue
                text = str(obj.get("text") or obj.get("content") or "").strip()
                if not text:
                    continue
                if len(text) < MIN_TEXT_LEN or len(text) > MAX_TEXT_LEN:
                    continue
                norm = _normalize_for_leakage(text)
                if not norm:
                    continue
                if norm in local_seen:
                    skipped += 1
                    continue
                local_seen.add(norm)
                out.append({
                    "text": text,
                    "risk": risk,
                    "is_sap": 0,
                    "sap_id": f"{src}-{n_from_file}",
                    "source": src,
                })
                n_from_file += 1
        if n_from_file:
            print(f"[extra_pools] {fname}: loaded {n_from_file} rows (risk={risk})")

    return out, skipped


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
        test_size=_get_pool_val_fraction(),
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
            # utf-8-sig: tolerate UTF-8 BOM (e.g. PowerShell Set-Content -Encoding utf8)
            with open(f, encoding="utf-8-sig") as fp:
                for line in fp:
                    line = line.strip()
                    if line:
                        raw_rows.append(json.loads(line))
        else:
            with open(f, encoding="utf-8-sig") as fp:
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
        norm = _normalize_for_leakage(text)
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
            with open(f, encoding="utf-8-sig") as fp:
                for line in fp:
                    line = line.strip()
                    if line:
                        raw_rows.append(json.loads(line))
        else:
            with open(f, encoding="utf-8-sig") as fp:
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


# Enron: quick content heuristics so corpora are not treated as all high-risk.
_ENRON_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b|\b\d{3}\s+\d{2}\s+\d{4}\b")
_ENRON_CC_RE = re.compile(
    r"\b(?:4[0-9]{12}(?:[0-9]{3})?|"  # Visa
    r"5[1-5][0-9]{14}|"
    r"3[47][0-9]{13}|"
    r"6(?:011|5[0-9]{2})[0-9]{12})\b"
)
_ENRON_PHONE_RE = re.compile(
    r"(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s/]?[0-9]{3}[-.\s/]?[0-9]{4}\b"
)
_ENRON_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_ENRON_TITLE_PAIR_RE = re.compile(r"\b([A-Z][a-z]{2,14})\s+([A-Z][a-z]{2,14})\b")
_ENRON_PAIR_FIRST_STOP = frozenset({
    "the", "and", "but", "for", "not", "are", "was", "has", "his", "her", "its", "our", "all",
    "this", "that", "with", "from", "have", "been", "will", "your", "any", "can", "may", "did",
    "get", "got", "she", "him", "they", "who", "how", "out", "one", "two", "per", "via",
})


def _enron_has_capitalized_name_pair(text: str) -> bool:
    """Heuristic: two consecutive Title Case words (possible person / entity names), excluding common sentence starters."""
    for m in _ENRON_TITLE_PAIR_RE.finditer(text):
        a = m.group(1)
        if a.lower() in _ENRON_PAIR_FIRST_STOP:
            continue
        return True
    return False


def _enron_heuristic_risk(text: str) -> str:
    """
    Rough label from body text: strong PII-like patterns -> high; title-case name pairs -> med; else low.
    Does not replace curated dataset labels elsewhere; used only in load_enron_real.
    """
    t = text or ""
    if not t.strip():
        return "low"
    if (
        _ENRON_SSN_RE.search(t)
        or _ENRON_CC_RE.search(t)
        or _ENRON_PHONE_RE.search(t)
        or _ENRON_EMAIL_RE.search(t)
    ):
        return "high"
    if _enron_has_capitalized_name_pair(t):
        return "med"
    return "low"


def load_enron_real(path: str | Path | None = None) -> list[dict]:
    """
    Load Enron/EDRM-style email or body text from local path.
    Supports .csv, .json, .jsonl, .txt. Extracts text snippets.

    Risk is assigned per row using content heuristics (not file labels): SSN / card / phone / email
    patterns -> high; capitalized word pairs (likely names) -> med; otherwise low.

    ``enron_pii_prompts.jsonl`` is capped at ``GUARDRAIL_ENRON_MAX_ROWS`` (default 20000) to avoid huge loads.
    """
    path = Path(path or ENRON_REAL_PATH)
    if not path.exists():
        print(f"[enron_real] path not found (skipped): {path}")
        return []
    text_keys = ("text", "content", "body", "excerpt", "prompt", "input", "message")

    if path.is_file():
        files = [path]
    else:
        # Skip backup dumps like enron_pii_prompts_full.jsonl (still *.jsonl) to avoid loading 2x.
        jsonl_files = [p for p in sorted(path.glob("*.jsonl")) if "_full" not in p.name.lower()]
        files = (
            sorted(path.glob("*.csv"))
            + sorted(path.glob("*.json"))
            + jsonl_files
            + sorted(path.glob("*.txt"))
        )

    raw_rows = []
    for f in files:
        if f.suffix == ".csv":
            import csv
            with open(f, encoding="utf-8", newline="") as fp:
                for r in csv.DictReader(fp):
                    raw_rows.append(dict(r))
        elif f.suffix == ".jsonl":
            max_pii = _env_int("GUARDRAIL_ENRON_MAX_ROWS", DEFAULT_ENRON_PII_PROMPTS_MAX_ROWS)
            is_pii_prompts = f.name == "enron_pii_prompts.jsonl"
            n_pii = 0
            with open(f, encoding="utf-8-sig") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    if is_pii_prompts and n_pii >= max_pii:
                        print(f"[enron_real] capped at {max_pii} rows (GUARDRAIL_ENRON_MAX_ROWS)")
                        break
                    raw_rows.append(json.loads(line))
                    if is_pii_prompts:
                        n_pii += 1
        elif f.suffix == ".json":
            with open(f, encoding="utf-8-sig") as fp:
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
        risk = _enron_heuristic_risk(text)
        out.append({"text": text, "risk": risk, "sap_id": f"enron-{len(out)}", "source": "enron_real"})

    counts = dict(Counter(r["risk"] for r in out)) if out else {}
    print(
        "[enron_real] risk from content heuristics (SSN/card/phone/email -> high; Title Case name pairs -> med; else low)"
    )
    print(f"[enron_real] total rows loaded: {len(out)} (all labeled by heuristic)")
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
            with open(f, encoding="utf-8-sig") as fp:
                for line in fp:
                    line = line.strip()
                    if line:
                        raw_rows.append(json.loads(line))
        else:
            with open(f, encoding="utf-8-sig") as fp:
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
            norm = _normalize_for_leakage(text)
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
        ("bigcode", load_bigcode_pii(_get_bigcode_max_rows())),
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
            norm = _normalize_for_leakage(text)
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

    return stratified_split(patronus_rows)


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


def generate_faker_synthetics(n: int | None = None) -> list[dict]:
    """
    Generate n synthetic samples with SAP IDs and finance text.
    Risk distribution: 30% low, 40% med, 30% high.
    If n is None, uses GUARDRAIL_NUM_SYNTHETICS (default NUM_SYNTHETICS).
    """
    if n is None:
        n = _get_num_synthetics()
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


def generate_business_safe_negatives(n: int = 3000, seed: int = 99) -> list[dict]:
    """
    Safer business-like negatives (no direct PII fields) used to diversify binary safe class.
    """
    import random as _random
    from faker import Faker as _Faker

    _random.seed(seed)
    fake = _Faker()
    _Faker.seed(seed)

    verbs = ("review", "approve", "triage", "archive", "summarize", "route", "validate")
    docs = ("design brief", "runbook", "incident note", "sprint recap", "quarterly summary", "ops checklist")
    systems = ("billing-service", "analytics-worker", "api-gateway", "etl-orchestrator", "crm-sync")
    samples: list[dict] = []
    for i in range(max(0, int(n))):
        mode = i % 7
        if mode == 0:
            text = (
                f"Project update: {fake.company()} requested { _random.choice(verbs) } of the "
                f"{ _random.choice(docs) } by {fake.day_of_week()} {fake.time(pattern='%H:%M')}."
            )
        elif mode == 1:
            text = (
                f"Ops log: deployment for { _random.choice(systems) } completed in "
                f"{_random.randint(3, 24)} minutes, no customer data attached."
            )
        elif mode == 2:
            text = (
                f"Finance planning note: budget line {fake.lexify('BUD-????')}-{_random.randint(100,999)} "
                f"moved to next quarter for vendor consolidation."
            )
        elif mode == 3:
            text = (
                f"Meeting recap: prioritize regression fixes and release notes; "
                f"exclude personal identifiers in all shared artifacts."
            )
        elif mode == 4:
            text = (
                f"Security checklist item: rotate non-production secrets and verify "
                f"redaction in telemetry dashboards."
            )
        elif mode == 5:
            text = (
                f"QA handoff: reproduce issue in staging with synthetic fixture set "
                f"{fake.lexify('SAFE-????')}-{_random.randint(1000,9999)}."
            )
        else:
            text = (
                f"Documentation task: {fake.bs().capitalize()} for service "
                f"{_random.choice(systems)} and publish sanitized examples only."
            )
        samples.append(
            {
                "text": text,
                "risk": "low",
                "sap_id": f"safebiz-{seed}-{i}",
                "source": "synthetic_safe_business",
            }
        )
    return samples


def generate_adversarial_risky_samples(n: int = 800, seed: int = 177) -> list[dict]:
    """
    Adversarial risky snippets (obfuscated formats) to improve recall on critical/risky edge cases.
    """
    import random as _random
    from faker import Faker as _Faker

    _random.seed(seed)
    fake = _Faker()
    _Faker.seed(seed)

    rows: list[dict] = []
    for i in range(max(0, int(n))):
        mode = i % 8
        if mode == 0:
            text = f"Payroll import: SSN {fake.numerify('### ## ####')} must be verified before sync."
        elif mode == 1:
            text = f"Card check: {' '.join(list(fake.numerify('################')))} failed preauth."
        elif mode == 2:
            text = f"Security note: Authorization Bearer {fake.pystr(min_chars=36, max_chars=48)}"
        elif mode == 3:
            text = (
                f"Ops paste: AWS_ACCESS_KEY_ID=AKIA{fake.pystr(min_chars=16, max_chars=16).upper()} "
                f"AWS_SECRET_ACCESS_KEY={fake.pystr(min_chars=40, max_chars=44)}"
            )
        elif mode == 4:
            text = f"Payment fallback route: IBAN GB29 NWBK 6016 1331 9268 19 for vendor payout."
        elif mode == 5:
            text = (
                f"Incident triage: token split sk - live - {fake.pystr(min_chars=24, max_chars=30)} "
                f"must not leave secure notes."
            )
        elif mode == 6:
            text = (
                f"Clinical export: Patient {fake.name()} MRN {fake.numerify('#######')} "
                f"diagnosis summary should be blocked in LLM chat."
            )
        else:
            text = (
                f"DB handoff: mongodb://admin:{fake.pystr(min_chars=10, max_chars=14)}@"
                f"{fake.word()}.mongodb.net/prod"
            )
        rows.append(
            {
                "text": text,
                "risk": "high",
                "sap_id": f"adv-risk-{seed}-{i}",
                "source": "synthetic_adversarial_risky",
            }
        )
    return rows


def generate_hard_medium_samples(n: int = 2200, seed: int = 143) -> list[dict]:
    """
    Medium-risk near-boundary samples.
    Designed to be harder than generic synthetic text while avoiding clear block-level triggers.
    """
    import random as _random
    from faker import Faker as _Faker

    _random.seed(seed)
    fake = _Faker()
    _Faker.seed(seed)

    def _mask(val: str, keep: int = 4) -> str:
        s = str(val or "")
        if len(s) <= keep:
            return "*" * max(1, len(s))
        return "*" * (len(s) - keep) + s[-keep:]

    samples: list[dict] = []
    for i in range(max(0, int(n))):
        mode = i % 6
        if mode == 0:
            text = (
                f"Customer onboarding ticket: contact {fake.email()} and phone {fake.phone_number()}. "
                f"Account ref ending {_mask(fake.random_number(digits=10, fix_len=True), keep=3)} for verification."
            )
        elif mode == 1:
            text = (
                f"Support handoff for {fake.name()} in {fake.city()}. "
                f"Address partially redacted: {fake.street_name()} [redacted], case {fake.uuid4()[:8]}."
            )
        elif mode == 2:
            text = (
                f"Reimbursement note: transfer approved to beneficiary {fake.company()}. "
                f"Only last digits shared: account ending {_mask(fake.random_number(digits=12, fix_len=True), keep=4)}."
            )
        elif mode == 3:
            text = (
                f"Draft email cleanup: remove personal references before LLM paste. "
                f"Example placeholders: [user_email], [phone], [city], [ticket:{fake.uuid4()[:6]}]."
            )
        elif mode == 4:
            text = (
                f"QA transcript excerpt: user mentioned DOB month and city for identity check, "
                f"but full identifiers are masked in summary #{fake.random_number(digits=6, fix_len=True)}."
            )
        else:
            text = (
                f"Analyst note for incident follow-up: partial token {_mask(fake.uuid4().replace('-', ''), keep=6)} "
                f"and contact {fake.email()} retained for audit."
            )
        samples.append({"text": text, "risk": "med", "sap_id": f"hard-med-{i}", "source": "synthetic_hard_med"})
    return samples


def generate_high_risk_samples(n: int = 1200, seed: int = 211) -> list[dict]:
    """Synthetic high-risk samples for class-coverage top-up in evaluation splits."""
    import random as _random
    from faker import Faker as _Faker

    _random.seed(seed)
    fake = _Faker()
    _Faker.seed(seed)

    samples: list[dict] = []
    for i in range(max(0, int(n))):
        mode = i % 6
        if mode == 0:
            tok_a = str(fake.uuid4()).replace("-", "")
            tok_b = str(fake.uuid4()).replace("-", "")
            text = (
                f"Escalation: credentials leak detected for {fake.user_name()}. "
                f"API key sk_live_{tok_a[:24]} and password reset token {tok_b}."
            )
        elif mode == 1:
            text = (
                f"Finance transfer approval contains full account {fake.iban() if hasattr(fake, 'iban') else fake.bban()} "
                f"with routing {fake.random_number(digits=9, fix_len=True)} and beneficiary {fake.name()}."
            )
        elif mode == 2:
            tok_c = str(fake.uuid4()).replace("-", "")
            tok_d = str(fake.uuid4()).replace("-", "")
            text = (
                f"Security incident report includes bearer token {tok_c}{tok_d[:8]} "
                f"and private endpoint key AKIA{tok_d[:16].upper()}."
            )
        elif mode == 3:
            text = (
                f"HR breach sample: employee {fake.name()}, SSN {fake.random_number(digits=3, fix_len=True)}-"
                f"{fake.random_number(digits=2, fix_len=True)}-{fake.random_number(digits=4, fix_len=True)}, "
                f"salary {fake.random_int(70000, 190000)}."
            )
        elif mode == 4:
            ins_id = str(fake.uuid4()).replace("-", "")[:12]
            text = (
                f"Medical record excerpt for patient {fake.name()} at {fake.address().replace(chr(10), ' ')} "
                f"with insurance ID {ins_id} and diagnosis notes."
            )
        else:
            jwt_a = str(fake.uuid4()).replace("-", "")
            jwt_b = str(fake.uuid4()).replace("-", "")
            jwt_c = str(fake.uuid4()).replace("-", "")
            text = (
                f"Production config leak: DB_URL=postgres://admin:{fake.password(length=18)}@{fake.domain_name()}:5432/prod "
                f"JWT={jwt_a}.{jwt_b[:20]}.{jwt_c[:24]}"
            )
        samples.append(
            {
                "text": text,
                "risk": "high",
                "sap_id": f"hard-high-{i}",
                "source": "synthetic_hard_high",
            }
        )
    return samples


def _risk_topup_samples(risk: str, n: int, *, seed: int) -> list[dict]:
    rk = str(risk or "").lower()
    n = max(0, int(n))
    if n <= 0:
        return []
    if rk == "low":
        rows = generate_low_risk_samples(n=n, seed=seed)
        return [
            {"text": r["text"], "risk": "low", "sap_id": f"topup-low-{i}", "source": "synthetic_topup_low"}
            for i, r in enumerate(rows)
        ]
    if rk == "med":
        rows = generate_hard_medium_samples(n=n, seed=seed)
        return [
            {
                "text": r["text"],
                "risk": "med",
                "sap_id": r.get("sap_id", f"topup-med-{i}"),
                "source": r.get("source", "synthetic_topup_med"),
            }
            for i, r in enumerate(rows)
        ]
    rows = generate_high_risk_samples(n=n, seed=seed)
    return [
        {
            "text": r["text"],
            "risk": "high",
            "sap_id": r.get("sap_id", f"topup-high-{i}"),
            "source": r.get("source", "synthetic_topup_high"),
        }
        for i, r in enumerate(rows)
    ]


def _rebalance_split_for_class_coverage(
    rows: list[dict],
    *,
    split_name: str,
    min_counts: dict[str, int],
    max_class_ratio: float = DEFAULT_EVAL_MAX_CLASS_RATIO,
    seed: int = 42,
) -> list[dict]:
    """
    Ensure low/med/high minimum counts and limit skew in a split.
    Used for validation/test (and optional holdout) so macro metrics are reliable.
    """
    rng = random.Random(seed)
    out = list(rows or [])
    if not out:
        out = []

    by_risk: dict[str, list[dict]] = {"low": [], "med": [], "high": []}
    for r in out:
        rk = str(r.get("risk", "")).lower()
        if rk in by_risk:
            by_risk[rk].append(r)

    for rk in ("low", "med", "high"):
        target = max(0, int((min_counts or {}).get(rk, 0)))
        cur = len(by_risk[rk])
        if cur < target:
            need = target - cur
            by_risk[rk].extend(_risk_topup_samples(rk, need, seed=seed + (13 * (1 + len(rk)))))
            print(f"[data] {split_name}: topped up {rk} by {need}")

    counts_now = {k: len(v) for k, v in by_risk.items()}
    positive_counts = [v for v in counts_now.values() if v > 0]
    if positive_counts:
        min_n = min(positive_counts)
        cap = int(max(1, min_n) * max(1.0, float(max_class_ratio)))
        for rk in ("low", "med", "high"):
            cur = len(by_risk[rk])
            if cur > cap:
                by_risk[rk] = rng.sample(by_risk[rk], cap)
                print(f"[data] {split_name}: downsampled {rk} from {cur} -> {cap} (ratio cap {max_class_ratio:.1f}:1)")

    merged = by_risk["low"] + by_risk["med"] + by_risk["high"]
    rng.shuffle(merged)
    return merged


def _enforce_holdout_coverage(
    source_holdout_rows: list[dict],
    trainable_real_rows: list[dict],
    *,
    min_counts: dict[str, int],
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Keep holdout source-separated when possible, and guarantee meaningful low/med/high support.
    Pulls missing classes from trainable real rows (removed from train pool), then synthetic top-up.
    """
    rng = random.Random(seed)
    holdout = list(source_holdout_rows or [])
    trainable = list(trainable_real_rows or [])

    holdout_by = {"low": [], "med": [], "high": []}
    for r in holdout:
        rk = str(r.get("risk", "")).lower()
        if rk in holdout_by:
            holdout_by[rk].append(r)

    # Borrow real rows first so holdout remains realistic.
    for rk in ("low", "med", "high"):
        target = max(0, int((min_counts or {}).get(rk, 0)))
        cur = len(holdout_by[rk])
        need = max(0, target - cur)
        if need <= 0:
            continue
        candidates = [r for r in trainable if str(r.get("risk", "")).lower() == rk]
        if candidates:
            take_n = min(need, len(candidates))
            chosen = rng.sample(candidates, take_n)
            chosen_ids = {id(x) for x in chosen}
            trainable = [r for r in trainable if id(r) not in chosen_ids]
            holdout_by[rk].extend(chosen)
            need -= take_n
            print(f"[data] holdout: moved {take_n} real '{rk}' rows from trainable pool")
        if need > 0:
            holdout_by[rk].extend(_risk_topup_samples(rk, need, seed=seed + (17 * (1 + len(rk)))))
            print(f"[data] holdout: synthetic top-up for '{rk}' by {need}")

    merged_holdout = holdout_by["low"] + holdout_by["med"] + holdout_by["high"]
    rng.shuffle(merged_holdout)
    return merged_holdout, trainable


def _rebalance_split_for_med_coverage(
    rows: list[dict],
    *,
    split_name: str,
    min_med: int,
    high_to_med_max_ratio: float = DEFAULT_HIGH_TO_MED_MAX_RATIO,
    seed: int = 42,
) -> list[dict]:
    """
    Keep medium class from becoming negligible and cap high/med skew in a split.
    """
    rng = random.Random(seed)
    out = list(rows or [])
    if not out:
        return out
    by_risk: dict[str, list[dict]] = {"low": [], "med": [], "high": []}
    for r in out:
        rk = str(r.get("risk", "")).lower()
        if rk in by_risk:
            by_risk[rk].append(r)

    med_n = len(by_risk["med"])
    high_n = len(by_risk["high"])
    if med_n < max(0, int(min_med)):
        need = int(min_med) - med_n
        extra = generate_hard_medium_samples(need, seed=seed + 11)
        by_risk["med"].extend(extra)
        med_n = len(by_risk["med"])
        print(f"[data] {split_name}: topped up med by {need} hard samples")

    # Cap high class relative to med to avoid one-class-dominant validation/test.
    cap_high = int(max(1, med_n) * max(1.0, float(high_to_med_max_ratio)))
    if high_n > cap_high:
        by_risk["high"] = rng.sample(by_risk["high"], cap_high)
        print(f"[data] {split_name}: downsampled high from {high_n} -> {cap_high}")

    merged = by_risk["low"] + by_risk["med"] + by_risk["high"]
    rng.shuffle(merged)
    return merged


def _rebalance_train_for_class_floors(
    rows: list[dict],
    *,
    min_counts: dict[str, int] | None = None,
    synthetic_high_cap: int | None = None,
    seed: int = 42,
) -> list[dict]:
    """
    Ensure minimum low/med/high counts in the training split without downsampling.
    Synthetic hard-high top-up is capped per run (GUARDRAIL_TRAIN_SYNTHETIC_HIGH_CAP) so the
    majority of training data stays real + existing synthetic mix for low/med.
    """
    rng = random.Random(seed)
    out = list(rows or [])
    if not out:
        return out

    mc = {
        "low": _env_int("GUARDRAIL_TRAIN_MIN_LOW", DEFAULT_TRAIN_MIN_COUNTS["low"]),
        "med": _env_int("GUARDRAIL_TRAIN_MIN_MED", DEFAULT_TRAIN_MIN_COUNTS["med"]),
        "high": _env_int("GUARDRAIL_TRAIN_MIN_HIGH", DEFAULT_TRAIN_MIN_COUNTS["high"]),
    }
    if min_counts:
        for k, v in min_counts.items():
            if k in mc and v is not None:
                mc[k] = max(0, int(v))

    cap_high = synthetic_high_cap if synthetic_high_cap is not None else _env_int(
        "GUARDRAIL_TRAIN_SYNTHETIC_HIGH_CAP", TRAIN_SYNTHETIC_HIGH_TOPUP_CAP
    )
    cap_high = max(0, int(cap_high))

    by_risk: dict[str, list[dict]] = {"low": [], "med": [], "high": []}
    for r in out:
        rk = str(r.get("risk", "")).lower()
        if rk in by_risk:
            by_risk[rk].append(r)

    for rk in ("low", "med", "high"):
        target = max(0, int(mc.get(rk, 0)))
        cur = len(by_risk[rk])
        if cur >= target:
            continue
        need = target - cur
        if rk == "high":
            allow = min(need, cap_high)
            if allow < need:
                print(
                    f"[data] train: high floor wants {need} more rows but synthetic high cap allows {allow} "
                    f"(GUARDRAIL_TRAIN_SYNTHETIC_HIGH_CAP={cap_high})"
                )
            if allow <= 0:
                continue
            by_risk[rk].extend(_risk_topup_samples("high", allow, seed=seed + 401))
            print(f"[data] train: topped up high by {allow} toward floor {target} (was {cur})")
        else:
            by_risk[rk].extend(_risk_topup_samples(rk, need, seed=seed + 223 + sum(ord(c) for c in rk)))
            print(f"[data] train: topped up {rk} by {need} toward floor {target} (was {cur})")

    merged = by_risk["low"] + by_risk["med"] + by_risk["high"]
    rng.shuffle(merged)
    return merged


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
        caps = dict(DEFAULT_TRAIN_CAPS)
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
    Rebalance training split with medium hard examples, low-risk baselines, and class caps.
    """
    texts = [r["text"] for r in train_rows]
    labels = [r["risk"] for r in train_rows]
    # Keep low coverage, but prioritize medium hard examples to improve boundary learning.
    for s in generate_low_risk_samples(1200):
        texts.append(s["text"])
        labels.append(s["risk"])
    for s in generate_hard_medium_samples(2800):
        texts.append(s["text"])
        labels.append(s["risk"])
    texts, labels = balance_dataset(
        texts,
        labels,
        caps=_get_train_caps(),
        seed=42,
    )
    print("[data] balanced distribution:", Counter(labels))
    return [{"text": t, "risk": lab, "sap_id": ""} for t, lab in zip(texts, labels)]


def stratified_split(rows: list[dict], ratios: tuple[float, float, float] | None = None):
    """Split rows into train/val/test with stratification on risk when possible."""
    if ratios is None:
        ratios = _get_split_ratios()
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
BINARY_RISK_TO_ID = {"safe": 0, "risky": 1}


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

    Scale / split (optional env): GUARDRAIL_NUM_SYNTHETICS, GUARDRAIL_SYNTHETIC_CAP_MULTIPLIER,
    GUARDRAIL_SPLIT_RATIOS (train,val,test), GUARDRAIL_POOL_VAL_FRACTION (val share of trainable pool),
    GUARDRAIL_TRAIN_CAP_LOW/MED/HIGH, GUARDRAIL_BIGCODE_MAX_ROWS (or "none" for full BigCode split).
    """
    nemotron = None
    binary_mode = _binary_mode_enabled() and data_source == "multi_real_synthetic"
    if data_source in ("nemotron", "both", "patronus"):
        nemotron = load_nemotron_pii()

    if data_source == "nemotron":
        train_rows, val_rows, test_rows = nemotron_to_risk_rows(nemotron)
    elif data_source == "faker":
        rows = generate_faker_synthetics()
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
        f_rows = generate_faker_synthetics()
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
        synthetic_rows = generate_faker_synthetics()
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
                texts, pool_risks, saps, test_size=_get_pool_val_fraction(), stratify=pool_risks, random_state=42
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
            ("bigcode", load_bigcode_pii(_get_bigcode_max_rows())),
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

        _pool_norms = {_normalize_for_leakage(r["text"]) for r in real_labeled_rows}
        try:
            extra_pool_rows, extra_pool_dups = load_extra_pool_rows_for_training(_pool_norms)
        except Exception as ex:
            print(f"[extra_pools] skipped: {ex}")
            extra_pool_rows, extra_pool_dups = [], 0
        for er in extra_pool_rows:
            real_labeled_rows.append(er)
        if extra_pool_rows:
            print(
                f"[extra_pools] merged {len(extra_pool_rows)} curated rows "
                f"(duplicate fingerprints skipped vs existing pool: {extra_pool_dups})"
            )

        n_real_labeled = len(real_labeled_rows)
        n_real_unlabeled = sum(len(rows) for _, rows in source_lists) - n_real_labeled  # approx; pool has dedup/filter
        cap = max(0, _get_synthetic_cap_multiplier() * n_real_labeled) if n_real_labeled else _get_num_synthetics()
        n_synthetic_target = min(_get_num_synthetics(), cap) if n_real_labeled else _get_num_synthetics()
        if binary_mode:
            n_synthetic_target = max(n_synthetic_target, _get_binary_synthetic_floor())
        _syn_raw = generate_faker_synthetics(n_synthetic_target)
        synthetic_rows = [
            {"text": r["text"], "risk": r["risk"], "is_sap": 0, "sap_id": r["sap_id"], "source": "synthetic"}
            for r in _syn_raw
        ]
        holdout_raw = str(os.environ.get("GUARDRAIL_SOURCE_HOLDOUT", "enron_real,kaggle_sensitive") or "").strip()
        holdout_sources = {s.strip().lower() for s in holdout_raw.split(",") if s.strip()}
        source_holdout_rows: list[dict] = []
        trainable_real_rows = list(real_labeled_rows)
        if holdout_sources:
            source_holdout_rows = [
                r for r in real_labeled_rows if str(r.get("source", "")).strip().lower() in holdout_sources
            ]
            trainable_real_rows = [
                r for r in real_labeled_rows if str(r.get("source", "")).strip().lower() not in holdout_sources
            ]
            if source_holdout_rows:
                print(
                    "[multi_real_synthetic] source-separated holdout enabled:",
                    sorted(holdout_sources),
                    f"(rows={len(source_holdout_rows)})",
                )
            else:
                print("[multi_real_synthetic] source holdout requested but no matching rows found; continuing normally.")

        if holdout_sources:
            source_holdout_rows, trainable_real_rows = _enforce_holdout_coverage(
                source_holdout_rows,
                trainable_real_rows,
                min_counts=DEFAULT_HOLDOUT_MIN_COUNTS,
                seed=42,
            )

        pool = trainable_real_rows + synthetic_rows
        if binary_mode:
            for row in pool:
                row["risk"] = _to_binary_risk(row.get("risk", "low"))
            for row in source_holdout_rows:
                row["risk"] = _to_binary_risk(row.get("risk", "low"))
            adv_n = _get_binary_adv_risky_samples()
            if adv_n > 0:
                adv_rows = generate_adversarial_risky_samples(adv_n, seed=719)
                for i, r in enumerate(adv_rows):
                    pool.append(
                        {
                            "text": r["text"],
                            "risk": "risky",
                            "is_sap": 0,
                            "sap_id": f"bin-adv-risk-{i}",
                            "source": r.get("source", "synthetic_adversarial_risky"),
                        }
                    )
                print(
                    "[data] Binary mode adversarial risky injection: "
                    f"added_risky={adv_n}"
                )
            pool_counts_before = Counter(r["risk"] for r in pool)
            safe_before = int(pool_counts_before.get("safe", 0))
            risky_before = int(pool_counts_before.get("risky", 0))
            safe_target_pool = max(0, min(_get_binary_safe_pool_target(), max(safe_before, risky_before)))
            safe_gap = max(0, safe_target_pool - safe_before)
            if safe_gap > 0:
                extra_safe = generate_business_safe_negatives(safe_gap, seed=313)
                for i, r in enumerate(extra_safe):
                    pool.append(
                        {
                            "text": r["text"],
                            "risk": "safe",
                            "is_sap": 0,
                            "sap_id": f"bin-safe-{i}",
                            "source": r.get("source", "synthetic_safe_business"),
                        }
                    )
                print(
                    "[data] Binary mode safe-pool expansion: "
                    f"pre-split safe={safe_before} risky={risky_before}, "
                    f"added_safe={safe_gap}, target_safe_pool={safe_target_pool}"
                )
            else:
                print(
                    "[data] Binary mode safe-pool expansion: "
                    f"pre-split safe={safe_before} risky={risky_before}, no extra safe rows needed"
                )
        random.seed(42)
        random.shuffle(pool)
        pool_risks = [r["risk"] for r in pool]
        risk_counts_pool = Counter(pool_risks)
        min_class = min(risk_counts_pool.values()) if risk_counts_pool else 0
        if len(pool) < 20 or min_class < 2:
            train_rows = pool
            val_rows = []
            test_rows = list(source_holdout_rows)
            print("WARNING: Too few rows or a class with < 2 examples; validation set is empty.")
        else:
            if source_holdout_rows:
                pool_sources = [r.get("source", "?") for r in pool]
                try:
                    train_texts, val_texts, train_risks, val_risks, train_saps, val_saps, train_srcs, val_srcs = train_test_split(
                        [r["text"] for r in pool], pool_risks, [r.get("sap_id", "") for r in pool], pool_sources,
                        test_size=_get_pool_val_fraction(), stratify=pool_risks, random_state=42
                    )
                except ValueError:
                    train_texts, val_texts, train_risks, val_risks, train_saps, val_saps, train_srcs, val_srcs = train_test_split(
                        [r["text"] for r in pool], pool_risks, [r.get("sap_id", "") for r in pool], pool_sources,
                        test_size=_get_pool_val_fraction(), random_state=42
                    )
                train_rows = [{"text": t, "risk": r, "sap_id": s, "source": src} for t, r, s, src in zip(train_texts, train_risks, train_saps, train_srcs)]
                val_rows = [{"text": t, "risk": r, "sap_id": s, "source": src} for t, r, s, src in zip(val_texts, val_risks, val_saps, val_srcs)]
                test_rows = list(source_holdout_rows)
            else:
                # No source-separated holdout available: keep a stratified in-domain test split.
                train_rows, val_rows, test_rows = stratified_split(pool)

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
        print(f"[multi_real_synthetic] Final holdout test count: {len(test_rows)}")
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
                "test_count": len(test_rows),
                "train_per_source": dict(Counter(r.get("source", "?") for r in train_rows)),
                "val_per_source": dict(Counter(r.get("source", "?") for r in val_rows)),
                "test_per_source": dict(Counter(r.get("source", "?") for r in test_rows)),
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

    # Rebalance splits for better medium-risk coverage and lower template leakage.
    if data_source not in ("ai4privacy_text_only", "ai4privacy_text_plus_real", "ai4privacy_kaggle_en"):
        if binary_mode:
            train_rows = _rebalance_binary_train_equal_halves(train_rows, seed=91, max_per_class=5000)
            bs = Counter(r["risk"] for r in train_rows)
            print(
                "[data] Binary mode: merged med+high -> risky. "
                f"Train: {bs.get('safe', 0)} safe / {bs.get('risky', 0)} risky"
            )
        else:
            train_rows = balance_train_risk_classes(train_rows)
            train_rows = _rebalance_split_for_med_coverage(train_rows, split_name="train", min_med=1400, seed=42)
            train_rows = _rebalance_train_for_class_floors(train_rows, seed=55)
            if data_source == "multi_real_synthetic":
                if _use_multi_real_equal_thirds():
                    train_rows = _rebalance_multi_real_train_equal_thirds(train_rows, seed=88)
                else:
                    print(
                        "[multi_real_synthetic] keeping post-balance train distribution "
                        "(GUARDRAIL_MULTI_REAL_EQUAL_THIRDS not enabled)"
                    )
            val_rows = _rebalance_split_for_class_coverage(
                val_rows,
                split_name="val",
                min_counts=DEFAULT_VAL_MIN_COUNTS,
                max_class_ratio=DEFAULT_EVAL_MAX_CLASS_RATIO,
                seed=77,
            )
            test_rows = _rebalance_split_for_class_coverage(
                test_rows,
                split_name="test",
                min_counts=DEFAULT_TEST_MIN_COUNTS,
                max_class_ratio=DEFAULT_EVAL_MAX_CLASS_RATIO,
                seed=99,
            )

        # Keep legacy synthetic eval injection opt-in only (defaults OFF to reduce leakage bias).
        inject_eval_synth = str(os.environ.get("GUARDRAIL_INJECT_SYNTH_EVAL", "0")).strip().lower() in {"1", "true", "yes"}
        if inject_eval_synth:
            low_test = generate_low_risk_samples(n=160, seed=99)
            val_rows.extend(generate_low_risk_samples(n=120, seed=77))
            test_rows.extend(low_test)
            print("[data] GUARDRAIL_INJECT_SYNTH_EVAL=1 -> added low-risk synthetic rows to val/test")
        for split_name, rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
            dist = Counter(r["risk"] for r in rows)
            print(f"[data] {split_name} distribution: {dict(dist)}")

    print("Final merged split sizes and per-class counts:")
    _print_split_summary(train_rows, val_rows, test_rows, prefix="  ")
    if data_source == "multi_real_synthetic":
        _td = Counter(r["risk"] for r in train_rows)
        _tn = len(train_rows)
        _nl = _td.get("low", 0)
        _vd = Counter(r["risk"] for r in val_rows)
        _ted = Counter(r["risk"] for r in test_rows)
        print(
            "[multi_real_synthetic] Final class distribution before Arrow save - "
            f"train: {dict(_td)} (n={_tn}, low_share={_nl / max(1, _tn):.4f}); "
            f"val: {dict(_vd)} (n={len(val_rows)}); test: {dict(_ted)} (n={len(test_rows)})"
        )

    use_risk_labels = data_source not in ("ai4privacy_text_only", "ai4privacy_text_plus_real", "ai4privacy_kaggle_en")
    if use_risk_labels:
        for row in train_rows + val_rows + test_rows:
            if binary_mode:
                row["label"] = BINARY_RISK_TO_ID[row["risk"]]
            else:
                row["label"] = RISK_TO_ID[row["risk"]]

    train_ds = Dataset.from_list(train_rows)
    val_ds = Dataset.from_list(val_rows)
    test_ds = Dataset.from_list(test_rows)

    try:
        tokenizer = AutoTokenizer.from_pretrained(tinybert_id)
    except RuntimeError as e:
        err = str(e).lower()
        if "client has been closed" in err or "winerror 10054" in err:
            print("[data] warning: Hugging Face network hiccup; retrying tokenizer load from local cache.")
            tokenizer = AutoTokenizer.from_pretrained(tinybert_id, local_files_only=True)
        else:
            raise

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
    elif binary_mode:
        label_config = {
            "num_labels": 2,
            "id2label": {"0": "safe", "1": "risky"},
            "label2id": {"safe": 0, "risky": 1},
            "source": data_source,
            "binary_mode": True,
        }
        with open(out_path / LABEL_CONFIG_FILENAME, "w", encoding="utf-8") as f:
            json.dump(label_config, f, indent=2)
        print(f"Wrote {out_path / LABEL_CONFIG_FILENAME} (num_labels=2, binary safe/risky)")

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
