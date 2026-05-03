"""
Fine-tune TinyBERT for 3-class sequence classification (guardrail risk).

Labels: 0 = no PII / low, 1 = some / med, 2 = many / high.
Uses arrow_datasets from data.py. Saves model to ./models/tinybert_guardrail.
Target: P@high > 90%.

If you get CVE-2025-32434 / torch.load requiring torch>=2.6: either upgrade with
  pip install --upgrade "torch>=2.6"
or (accepting risk) set env ALLOW_TORCH_LOAD_PRE26=1 before running.
"""

import json
import os
import io
import shutil
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    BertForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    logging as hf_logging,
)

from data import ARROW_SAVE_DIR, RISK_TO_ID

# --- Config ---
_MODEL_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = _MODEL_ROOT / "config"
TRAIN_CONFIG_PATH = CONFIG_DIR / "train_config.json"
MODEL_ID = os.environ.get("GUARDRAIL_MODEL_ID", "huawei-noah/TinyBERT_General_4L_312D")
NUM_LABELS_DEFAULT = 3
SAVE_DIR = str(_MODEL_ROOT / "models" / "tinybert_guardrail")
EPOCHS = 6
LR = 2e-5
BATCH_SIZE = 16
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
CLASSIFIER_DROPOUT = 0.1
LABEL_SMOOTHING = 0.05
EARLY_STOPPING_PATIENCE = 2
BEST_MODEL_METRIC = "eval_f1"
ID_TO_RISK_3 = {v: k for k, v in RISK_TO_ID.items()}
ID_TO_RISK_2 = {0: "safe", 1: "risky"}
LABEL_CONFIG_FILENAME = "label_config.json"


def load_train_config() -> dict:
    """Load optional train_config.json; env vars override. Used for dataset/tweaks."""
    defaults = {
        "epochs": EPOCHS,
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "warmup_ratio": 0.08,
        "weight_decay": WEIGHT_DECAY,
        "max_grad_norm": MAX_GRAD_NORM,
        "classifier_dropout": CLASSIFIER_DROPOUT,
        "label_smoothing_factor": LABEL_SMOOTHING,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "metric_for_best_model": BEST_MODEL_METRIC,
        "use_class_weights": True,
        "class_weights_3class": None,
        "class_weight_mode_3class": "balanced",
        "rebalance_train_3class": True,
        "rebalance_mode_3class": "min",
        "rebalance_seed": 42,
    }
    if TRAIN_CONFIG_PATH.exists():
        try:
            with open(TRAIN_CONFIG_PATH, encoding="utf-8") as f:
                defaults.update({k: v for k, v in json.load(f).items() if k in defaults})
        except Exception:
            pass
    for key, env_key in [
        ("epochs", "GUARDRAIL_EPOCHS"),
        ("learning_rate", "GUARDRAIL_LR"),
        ("batch_size", "GUARDRAIL_BATCH_SIZE"),
        ("warmup_ratio", "GUARDRAIL_WARMUP_RATIO"),
        ("label_smoothing_factor", "GUARDRAIL_LABEL_SMOOTHING"),
        ("early_stopping_patience", "GUARDRAIL_EARLY_STOPPING_PATIENCE"),
        ("rebalance_seed", "GUARDRAIL_REBALANCE_SEED"),
    ]:
        v = os.environ.get(env_key)
        if v is not None:
            try:
                defaults[key] = int(v) if key in ("epochs", "batch_size", "early_stopping_patience", "rebalance_seed") else float(v)
            except ValueError:
                pass
    metric_env = os.environ.get("GUARDRAIL_BEST_METRIC")
    if metric_env:
        defaults["metric_for_best_model"] = str(metric_env).strip()
    cw_mode_env = os.environ.get("GUARDRAIL_CLASS_WEIGHT_MODE_3CLASS")
    if cw_mode_env:
        defaults["class_weight_mode_3class"] = str(cw_mode_env).strip().lower()
    rb_mode_env = os.environ.get("GUARDRAIL_REBALANCE_MODE_3CLASS")
    if rb_mode_env:
        defaults["rebalance_mode_3class"] = str(rb_mode_env).strip().lower()
    rb_on_env = os.environ.get("GUARDRAIL_REBALANCE_3CLASS")
    if rb_on_env is not None:
        defaults["rebalance_train_3class"] = str(rb_on_env).strip().lower() in ("1", "true", "yes", "on")
    wl, wm, wh = (
        os.environ.get("GUARDRAIL_WEIGHT_LOW"),
        os.environ.get("GUARDRAIL_WEIGHT_MED"),
        os.environ.get("GUARDRAIL_WEIGHT_HIGH"),
    )
    if all(x is not None and str(x).strip() != "" for x in (wl, wm, wh)):
        try:
            defaults["class_weights_3class"] = [
                float(str(wl).strip()),
                float(str(wm).strip()),
                float(str(wh).strip()),
            ]
        except ValueError:
            pass
    return defaults


def compute_class_weights(train_ds: Dataset, num_labels: int) -> torch.Tensor | None:
    """Inverse frequency weights for imbalanced 9-class; None if not applicable."""
    if num_labels != 9 or "labels" not in train_ds.column_names:
        return None
    from collections import Counter
    counts = Counter(int(x) for x in train_ds["labels"])
    total = sum(counts.values())
    if total == 0:
        return None
    weights = []
    for i in range(num_labels):
        c = max(counts.get(i, 0), 1)
        weights.append(total / (num_labels * c))
    w = torch.tensor(weights, dtype=torch.float32)
    return w / w.sum() * num_labels


def compute_three_class_weights(
    train_ds: Dataset,
    cfg_weights: list | None = None,
    mode: str = "balanced",
) -> torch.Tensor | None:
    """
    3-class weighting:
    - manual list from config/env wins when provided
    - otherwise derive balanced weights from observed class counts
    """
    if "labels" not in train_ds.column_names:
        return None
    if cfg_weights and isinstance(cfg_weights, list) and len(cfg_weights) == 3:
        try:
            w = [float(cfg_weights[0]), float(cfg_weights[1]), float(cfg_weights[2])]
            return torch.tensor(w, dtype=torch.float32)
        except Exception:
            pass
    from collections import Counter
    counts = Counter(int(x) for x in train_ds["labels"])
    total = float(sum(counts.get(i, 0) for i in (0, 1, 2)))
    if total <= 0:
        return torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    m = str(mode or "balanced").strip().lower()
    if m in ("uniform", "none", "off"):
        return torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    vals = []
    for i in (0, 1, 2):
        c = max(float(counts.get(i, 0)), 1.0)
        base = total / (3.0 * c)
        vals.append(np.sqrt(base) if m == "sqrt_balanced" else base)
    w = torch.tensor(vals, dtype=torch.float32)
    w = w / torch.mean(w)
    return w


def rebalance_three_class_train_dataset(
    train_ds: Dataset,
    mode: str = "min",
    seed: int = 42,
) -> Dataset:
    """
    Rebalance 3-class train split by sampling per class.
    mode:
      - min: downsample each class to minority count (no oversampling)
      - max: oversample each class to majority count
      - median: target median class count (downsample/oversample as needed)
    """
    if "labels" not in train_ds.column_names:
        return train_ds
    labels = [int(x) for x in train_ds["labels"]]
    by_cls = {c: [] for c in (0, 1, 2)}
    for idx, lb in enumerate(labels):
        if lb in by_cls:
            by_cls[lb].append(idx)
    counts = {k: len(v) for k, v in by_cls.items()}
    if any(v == 0 for v in counts.values()):
        print("[train] rebalance skipped: one or more classes missing in train split.")
        return train_ds
    vals = sorted(counts.values())
    m = str(mode or "min").strip().lower()
    if m == "max":
        target = vals[-1]
    elif m == "median":
        target = vals[1]
    else:
        target = vals[0]
    rng = np.random.default_rng(int(seed))
    selected = []
    for c in (0, 1, 2):
        idxs = np.asarray(by_cls[c], dtype=np.int64)
        n = len(idxs)
        if n > target:
            picked = rng.choice(idxs, size=target, replace=False)
        elif n < target:
            picked = rng.choice(idxs, size=target, replace=True)
        else:
            picked = idxs
        selected.extend(int(i) for i in picked.tolist())
    rng.shuffle(selected)
    out = train_ds.select(selected)
    from collections import Counter
    after = Counter(int(x) for x in out["labels"])
    print(f"[train] 3-class rebalance mode={m} target={target} before={counts} after={dict(after)}")
    return out


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy for imbalanced PII classes."""

    def __init__(self, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights.to(kwargs["model"].device) if class_weights is not None else None
        # Keep parity with TrainingArguments label_smoothing_factor even with custom weighted loss.
        self.label_smoothing = float(getattr(self.args, "label_smoothing_factor", 0.0) or 0.0)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels", None)
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None and labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=self.class_weights,
                label_smoothing=self.label_smoothing,
            )
            loss = loss_fct(logits, labels)
        else:
            loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def safe_nonempty_splits(dataset_dict):
    """
    Return a DatasetDict containing only splits that exist and have num_rows > 0.
    Does not modify the original.
    """
    out = {}
    for name, ds in dataset_dict.items():
        if ds is not None and getattr(ds, "num_rows", 0) > 0:
            out[name] = ds
    return DatasetDict(out) if out else DatasetDict({})


def load_arrow_splits(arrow_dir: str = ARROW_SAVE_DIR):
    """
    Load arrow_datasets by loading each split separately. Empty splits (e.g. test with 0 rows)
    cause load_from_disk to raise IndexError when loading the whole dir; we avoid that by
    loading only train/validation/test subdirs that exist and have rows > 0.
    - train: required; raise if missing or empty.
    - validation: optional; if missing/empty, training runs without eval.
    - test: optional; ignored if missing or empty.
    """
    path = Path(arrow_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Arrow datasets not found at {path}. Run: python data.py multi_real_synthetic"
        )
    loaded = {}
    for split_name in ("train", "validation", "test"):
        split_path = path / split_name
        if not split_path.is_dir():
            continue
        try:
            ds = load_from_disk(str(split_path))
            if ds is not None and getattr(ds, "num_rows", 0) > 0:
                loaded[split_name] = ds
        except (IndexError, Exception):
            # Empty or corrupted split (e.g. 0 rows) -> skip
            continue
    datasets = DatasetDict(loaded) if loaded else DatasetDict({})
    if "train" not in datasets or datasets["train"].num_rows == 0:
        raise ValueError(
            f"Train split is missing or empty in {path}. "
            "Ensure data.py produced a non-empty train set."
        )
    print("Loaded Arrow splits (non-empty only):")
    for name, ds in datasets.items():
        print(f"  {name}: {ds.num_rows} rows")
    return datasets


def make_compute_metrics(num_labels: int, id2label: dict):
    """Build compute_metrics that uses the given num_labels and id2label (id -> str)."""
    # Ensure int keys for id2label
    id2label = {int(k): v for k, v in id2label.items()}
    label_list = [id2label.get(i, str(i)) for i in range(num_labels)]

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        labels = np.asarray(labels)

        # Overall accuracy
        acc = accuracy_score(labels, preds)

        # Macro precision/recall/F1 + per-class metrics
        p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )
        p_per, r_per, f_per, _ = precision_recall_fscore_support(
            labels, preds, average=None, labels=list(range(num_labels)), zero_division=0
        )
        out = {
            "accuracy": float(acc),
            "precision": float(p_macro),
            "recall": float(r_macro),
            "f1": float(f_macro),
        }
        for i in range(num_labels):
            name = label_list[i]
            out[f"precision_{name}"] = float(p_per[i])
            out[f"recall_{name}"] = float(r_per[i])
            out[f"f1_{name}"] = float(f_per[i])
        if num_labels == 3 and "high" in label_list:
            out["P_high"] = out["precision_high"]
        return out

    return compute_metrics


def get_best_and_last_eval(log_history, metric_for_best_model="eval_P_high", greater_is_better=True):
    """
    From trainer.state.log_history, find all evaluation entries (those containing eval_loss).
    Returns (best_entry, last_entry) where:
    - last_entry: the final evaluation entry in the log (chronologically last).
    - best_entry: the entry with the best metric_for_best_model (e.g. eval_P_high), or
      if that metric is missing, the entry with lowest eval_loss.
    Do NOT use the first eval entry; last is always eval_entries[-1].
    """
    eval_entries = [e for e in log_history if "eval_loss" in e]
    if not eval_entries:
        return None, None
    last_entry = eval_entries[-1]
    # Best: by metric_for_best_model if present, else by lowest eval_loss
    if metric_for_best_model and any(metric_for_best_model in e for e in eval_entries):
        best_entry = max(
            (e for e in eval_entries if metric_for_best_model in e),
            key=lambda e: e[metric_for_best_model] if greater_is_better else (-e[metric_for_best_model]),
        )
    else:
        best_entry = min(eval_entries, key=lambda e: e["eval_loss"])
    return best_entry, last_entry


def _format_eval_section(entry, section_name):
    """Print one section (Best or Last) with all eval_* metrics from entry."""
    def _v(k, default=0.0):
        return entry.get(k, default)

    epoch_val = entry.get("epoch")
    epoch_str = epoch_val if epoch_val is not None else "N/A"

    print("\n" + "=" * 60)
    print(section_name)
    print("=" * 60)
    rows = [("epoch", epoch_str), ("eval_loss", f"{_v('eval_loss'):.4f}")]
    for key in sorted(entry.keys()):
        if key.startswith("eval_") and key != "eval_loss":
            # Hide HEALTH metrics from console output when that class is unused
            if "_HEALTH" in key:
                continue
            val = entry[key]
            rows.append((key, f"{val:.4f}" if isinstance(val, (int, float)) else str(val)))
    col_w = max(len(str(r[0])) for r in rows) + 2
    for name, val in rows:
        print(f"  {name:<{col_w}} {val}")
    print("=" * 60)


def main():
    datasets = load_arrow_splits()
    train_ds = datasets["train"].rename_column("label", "labels")
    val_ds = None
    if "validation" in datasets and datasets["validation"].num_rows > 0:
        val_ds = datasets["validation"].rename_column("label", "labels")
    else:
        print("WARNING: Validation split missing or empty. Training without evaluation.")

    # Label config: from arrow_dir/label_config.json (e.g. ai4privacy 9-class / binary) or default 3-class
    arrow_path = Path(ARROW_SAVE_DIR)
    label_config_path = arrow_path / LABEL_CONFIG_FILENAME
    binary_env = str(os.environ.get("GUARDRAIL_BINARY_MODE", "")).strip().lower() in ("1", "true", "yes", "on")
    if label_config_path.exists():
        with open(label_config_path, encoding="utf-8") as f:
            label_config = json.load(f)
        num_labels = int(label_config["num_labels"])
        id2label_raw = label_config.get("id2label", {})
        id2label = {int(k): str(v) for k, v in id2label_raw.items()}
        print(f"Using label_config: num_labels={num_labels}, id2label keys: {list(id2label.keys())[:5]}...")
    else:
        num_labels = NUM_LABELS_DEFAULT
        id2label = ID_TO_RISK_3
        print("Using default 3-class labels (no label_config.json).")

    unique_labels: set[int] = set()
    try:
        unique_labels.update(int(x) for x in train_ds["labels"])
        if val_ds is not None:
            unique_labels.update(int(x) for x in val_ds["labels"])
    except Exception:
        pass
    binary_dataset = len(unique_labels) == 2 and unique_labels.issubset({0, 1})
    binary_active = bool(binary_env or num_labels == 2 or binary_dataset)
    if binary_active:
        num_labels = 2
        id2label = ID_TO_RISK_2
        print(
            "[train] Binary classifier mode active: "
            f"env={binary_env}, dataset_unique_labels={sorted(unique_labels) if unique_labels else 'unknown'}"
        )

    compute_metrics = make_compute_metrics(num_labels, id2label)

    # Optional: allow loading .bin weights with torch<2.6 (CVE-2025-32434 workaround; use at your own risk)
    if os.environ.get("ALLOW_TORCH_LOAD_PRE26") == "1":
        import transformers.utils.import_utils as _iu
        _iu.check_torch_load_is_safe = lambda: None

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    except RuntimeError as e:
        err = str(e).lower()
        if "client has been closed" in err or "winerror 10054" in err:
            print("[train] warning: Hugging Face network hiccup; retrying tokenizer load from local cache.")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
        else:
            raise
    try:
        warnings.filterwarnings(
            "ignore",
            message=".*beta.*gamma.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=".*Some weights of the model checkpoint.*",
            category=UserWarning,
        )
        # Suppress noisy TinyBERT checkpoint logs during model load.
        prev_level = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        with redirect_stdout(io.StringIO()):
            try:
                model = BertForSequenceClassification.from_pretrained(
                    MODEL_ID,
                    num_labels=num_labels,
                    id2label=id2label,
                    label2id={str(v): k for k, v in id2label.items()},
                    ignore_mismatched_sizes=True,
                )
            except RuntimeError as e:
                err = str(e).lower()
                if "client has been closed" in err or "winerror 10054" in err:
                    print("[train] warning: Hugging Face network hiccup; retrying model load from local cache.")
                    model = BertForSequenceClassification.from_pretrained(
                        MODEL_ID,
                        num_labels=num_labels,
                        id2label=id2label,
                        label2id={str(v): k for k, v in id2label.items()},
                        ignore_mismatched_sizes=True,
                        local_files_only=True,
                    )
                else:
                    raise
        hf_logging.set_verbosity(prev_level)
        print(
            "[TinyBERT] Loaded pretrained encoder. "
            "Classifier head randomly initialized — this is expected for fine-tuning."
        )
    except ValueError as e:
        hf_logging.set_verbosity_warning()
        if "torch.load" in str(e) and "2.6" in str(e):
            raise SystemExit(
                "CVE-2025-32434: transformers requires torch>=2.6 to load .bin weights. "
                "TinyBERT on the Hub has no safetensors yet.\n"
                "Fix one of:\n"
                "  1. pip install --upgrade 'torch>=2.6'\n"
                "  2. Or set env and accept risk: $env:ALLOW_TORCH_LOAD_PRE26='1'; python train.py"
            ) from e
        raise

    model.config.id2label = id2label
    if hasattr(model.config, "label2id"):
        model.config.label2id = {v: k for k, v in id2label.items()}
        print(f"Label mapping: {model.config.label2id}")

    train_cfg = load_train_config()
    if binary_active:
        # Binary-safe defaults requested for med/high merged training.
        train_cfg["epochs"] = 4
        train_cfg["learning_rate"] = 1e-5
        train_cfg["weight_decay"] = 0.01
        train_cfg["classifier_dropout"] = 0.3
        train_cfg["label_smoothing_factor"] = 0.1
        train_cfg["early_stopping_patience"] = 3
        train_cfg["rebalance_train_3class"] = False
        print(
            "[train] Binary hyperparams: epochs=4 lr=1e-5 dropout=0.3 "
            "label_smoothing=0.1 weight_decay=0.01 early_stop=3"
        )
    if num_labels == 3 and train_cfg.get("rebalance_train_3class", True):
        train_ds = rebalance_three_class_train_dataset(
            train_ds,
            mode=str(train_cfg.get("rebalance_mode_3class", "min")),
            seed=int(train_cfg.get("rebalance_seed", 42)),
        )
    if getattr(model.config, "classifier_dropout", None) is not None:
        model.config.classifier_dropout = float(train_cfg.get("classifier_dropout", CLASSIFIER_DROPOUT))

    class_weights = None
    # 3-class: keep weights mild to improve high precision and reduce false alarms.
    if num_labels == 3 and train_cfg.get("use_class_weights", True):
        class_weights = compute_three_class_weights(
            train_ds,
            train_cfg.get("class_weights_3class"),
            mode=str(train_cfg.get("class_weight_mode_3class", "balanced")),
        )
        if class_weights is not None:
            print(
                "Using 3-class weights:",
                [round(float(x), 4) for x in class_weights.tolist()],
                f"(mode={str(train_cfg.get('class_weight_mode_3class', 'balanced'))})",
            )
    elif train_cfg.get("use_class_weights") and num_labels == 9:
        class_weights = compute_class_weights(train_ds, num_labels)
        if class_weights is not None:
            print("Using class weights for imbalanced 9-class:", class_weights.tolist())
    elif num_labels == 2:
        from sklearn.utils.class_weight import compute_class_weight

        y = np.asarray([int(x) for x in train_ds["labels"]])
        cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
        class_weights = torch.tensor(cw, dtype=torch.float32)
        print(
            f"Binary class weights (balanced): safe={cw[0]:.4f}, risky={cw[1]:.4f}"
        )

    best_metric = str(train_cfg.get("metric_for_best_model", BEST_MODEL_METRIC) or BEST_MODEL_METRIC).strip()
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=SAVE_DIR,
        num_train_epochs=float(train_cfg.get("epochs", EPOCHS)),
        learning_rate=float(train_cfg.get("learning_rate", LR)),
        per_device_train_batch_size=train_cfg.get("batch_size", BATCH_SIZE),
        per_device_eval_batch_size=train_cfg.get("batch_size", BATCH_SIZE),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.08)),
        weight_decay=float(train_cfg.get("weight_decay", WEIGHT_DECAY)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", MAX_GRAD_NORM)),
        label_smoothing_factor=float(train_cfg.get("label_smoothing_factor", LABEL_SMOOTHING)),
        eval_strategy="epoch" if val_ds is not None else "no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True if val_ds is not None else False,
        metric_for_best_model=best_metric if val_ds is not None else None,
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
    )

    trainer_cls = WeightedTrainer if class_weights is not None else Trainer
    trainer_kw = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    if class_weights is not None:
        trainer_kw["class_weights"] = class_weights
    trainer = trainer_cls(**trainer_kw)
    if val_ds is not None:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=int(train_cfg.get("early_stopping_patience", EARLY_STOPPING_PATIENCE))))

    trainer.train()
    saved_ok = False
    try:
        trainer.save_model(SAVE_DIR)
        saved_ok = True
    except Exception as e:
        err = str(e).lower()
        if "os error 1224" in err or "user-mapped section open" in err:
            print("[train] warning: safetensors save lock on Windows; falling back to .bin serialization.")
        else:
            raise

    if not saved_ok:
        # Fallback path for Windows mmap lock edge cases.
        try:
            trainer.model.save_pretrained(SAVE_DIR, safe_serialization=False)
            saved_ok = True
        except Exception as e:
            err = str(e).lower()
            if "os error 1224" in err or "user-mapped section open" in err:
                print("[train] warning: .bin fallback save also locked; keeping existing epoch checkpoints.")
            else:
                raise

    if saved_ok:
        # Best effort: also write safetensors for downstream compatibility.
        try:
            trainer.model.save_pretrained(SAVE_DIR, safe_serialization=True)
        except Exception as e:
            err = str(e).lower()
            if "os error 1224" in err or "user-mapped section open" in err:
                print("[train] warning: could not write safetensors due to active file mapping; .bin checkpoint kept.")
            else:
                raise
    tokenizer.save_pretrained(SAVE_DIR)

    # If root model.safetensors could not be updated (Windows mmap / file lock), sync from best checkpoint.
    def _sync_root_safetensors_from_checkpoint() -> None:
        try:
            import safetensors.torch as st
        except ImportError:
            return
        save_p = Path(SAVE_DIR)
        root_st = save_p / "model.safetensors"
        best = getattr(trainer.state, "best_model_checkpoint", None) or ""
        if not best or not Path(best).is_dir():
            cks = sorted(
                save_p.glob("checkpoint-*"),
                key=lambda p: int(p.name.split("-", 1)[1])
                if p.name.split("-", 1)[1].isdigit()
                else -1,
            )
            best = str(cks[-1]) if cks else ""
        if not best:
            return
        ck_st = Path(best) / "model.safetensors"
        if not ck_st.is_file():
            return
        try:
            cn = int(st.load_file(str(ck_st))["classifier.weight"].shape[0])
        except Exception:
            return
        if cn != num_labels:
            return
        try:
            rn = int(st.load_file(str(root_st))["classifier.weight"].shape[0]) if root_st.is_file() else -1
        except Exception:
            rn = -1
        if rn == num_labels:
            return
        side = save_p / "model_2class.safetensors"
        try:
            shutil.copyfile(ck_st, side)
            print(
                f"[train] Copied trained weights to {side.name} (root model.safetensors is "
                f"stale/locked; classifier on disk was {rn} classes, expected {num_labels}). "
                "Close the app and replace model.safetensors, or use this file / checkpoint-*."
            )
        except OSError as e:
            print(f"[train] warning: could not copy sidecar weights: {e}")

    _sync_root_safetensors_from_checkpoint()

    # Reports dir (package dir, same as model)
    reports_dir = _MODEL_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Save full log history
    log_history = getattr(trainer.state, "log_history", [])
    with open(reports_dir / "train_log_history.json", "w", encoding="utf-8") as f:
        json.dump(log_history, f, indent=2)

    if val_ds is not None:
        metric_for_best = getattr(args, "metric_for_best_model", None) or BEST_MODEL_METRIC
        best_entry, last_entry = get_best_and_last_eval(
            log_history,
            metric_for_best_model=metric_for_best,
            greater_is_better=True,
        )
        if best_entry is None:
            fresh = trainer.evaluate(eval_dataset=val_ds)
            best_entry = last_entry = fresh

        def _entry_to_dict(entry):
            if entry is None:
                return {}
            return {k: v for k, v in entry.items() if k.startswith("eval_") or k == "epoch"}

        summary = {
            "best": _entry_to_dict(best_entry),
            "last": _entry_to_dict(last_entry),
        }
        with open(reports_dir / "train_eval_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 60)
        print("Validation metrics (primary: macro-F1 + per-class recall)")
        print("=" * 60)
        _format_eval_section(best_entry, "Best validation metrics")
        _format_eval_section(last_entry, "Last validation metrics")
    else:
        summary = {"best": None, "last": None, "note": "No validation set used."}
        with open(reports_dir / "train_eval_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    print(f"Model saved to {SAVE_DIR}")
    print(f"Reports saved to {reports_dir}")


if __name__ == "__main__":
    main()
