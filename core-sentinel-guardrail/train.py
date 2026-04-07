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
EPOCHS = 5
LR = 3e-5
BATCH_SIZE = 16
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
CLASSIFIER_DROPOUT = 0.1
ID_TO_RISK_3 = {v: k for k, v in RISK_TO_ID.items()}
LABEL_CONFIG_FILENAME = "label_config.json"


def load_train_config() -> dict:
    """Load optional train_config.json; env vars override. Used for dataset/tweaks."""
    defaults = {
        "epochs": EPOCHS,
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "weight_decay": WEIGHT_DECAY,
        "max_grad_norm": MAX_GRAD_NORM,
        "classifier_dropout": CLASSIFIER_DROPOUT,
        "use_class_weights": False,
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
    ]:
        v = os.environ.get(env_key)
        if v is not None:
            try:
                defaults[key] = int(v) if key == "epochs" or key == "batch_size" else float(v)
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


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy for imbalanced PII classes."""

    def __init__(self, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights.to(kwargs["model"].device) if class_weights is not None else None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels", None)
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None and labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
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

    # Label config: from arrow_dir/label_config.json (e.g. ai4privacy 9-class) or default 3-class
    arrow_path = Path(ARROW_SAVE_DIR)
    label_config_path = arrow_path / LABEL_CONFIG_FILENAME
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

    compute_metrics = make_compute_metrics(num_labels, id2label)

    # Optional: allow loading .bin weights with torch<2.6 (CVE-2025-32434 workaround; use at your own risk)
    if os.environ.get("ALLOW_TORCH_LOAD_PRE26") == "1":
        import transformers.utils.import_utils as _iu
        _iu.check_torch_load_is_safe = lambda: None

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
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
            model = BertForSequenceClassification.from_pretrained(
                MODEL_ID,
                num_labels=num_labels,
                ignore_mismatched_sizes=True,
            )
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
    if getattr(model.config, "classifier_dropout", None) is not None:
        model.config.classifier_dropout = float(train_cfg.get("classifier_dropout", CLASSIFIER_DROPOUT))

    class_weights = None
    # 3-class: label2id is {'low': 0, 'med': 1, 'high': 2} — weights [low, med, high].
    if num_labels == 3:
        class_weights = torch.tensor(
            [
                1.0,  # low  (index 0)
                40.0,  # med  (index 1)
                100.0,  # high (index 2)
            ],
            dtype=torch.float,
        )
        print("Using hardcoded class weights for imbalanced 3-class: [1.0, 40.0, 100.0]")
        print("Class weights: low=1.0, med=40.0, high=100.0")
    elif train_cfg.get("use_class_weights") and num_labels == 9:
        class_weights = compute_class_weights(train_ds, num_labels)
        if class_weights is not None:
            print("Using class weights for imbalanced 9-class:", class_weights.tolist())

    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=SAVE_DIR,
        num_train_epochs=3,
        learning_rate=float(train_cfg.get("learning_rate", LR)),
        per_device_train_batch_size=train_cfg.get("batch_size", BATCH_SIZE),
        per_device_eval_batch_size=train_cfg.get("batch_size", BATCH_SIZE),
        warmup_steps=500,
        weight_decay=float(train_cfg.get("weight_decay", WEIGHT_DECAY)),
        max_grad_norm=1.0,
        eval_strategy="epoch" if val_ds is not None else "no",
        save_strategy="epoch",
        load_best_model_at_end=True if val_ds is not None else False,
        metric_for_best_model="eval_f1" if val_ds is not None else None,
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

    trainer.train()
    trainer.save_model(SAVE_DIR)
    # Save as safetensors so inference works without torch>=2.6 (CVE-2025-32434)
    trainer.model.save_pretrained(SAVE_DIR, safe_serialization=True)
    tokenizer.save_pretrained(SAVE_DIR)

    # Reports dir (package dir, same as model)
    reports_dir = _MODEL_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Save full log history
    log_history = getattr(trainer.state, "log_history", [])
    with open(reports_dir / "train_log_history.json", "w", encoding="utf-8") as f:
        json.dump(log_history, f, indent=2)

    if val_ds is not None:
        metric_for_best = getattr(args, "metric_for_best_model", None) or "eval_f1"
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
        print("Validation metrics (" + ("eval_f1 (9-class)" if num_labels != 3 else "P@high target > 90%") + ")")
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
