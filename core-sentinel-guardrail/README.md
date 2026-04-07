# Core Sentinel Guardrail

Fine-tuned sequence classification for PII/risk guardrail. Supports two label modes.

---

## Label modes: 3-class vs 9-class

**3-class (default)**  
Labels: `low` / `med` / `high` risk. Original guardrail behavior (e.g. Nemotron, Faker, Patronus, multi_real_synthetic).

**9-class (ai4privacy)**  
Fine-grained PII labels: `O`, `NAME`, `CONTACT`, `LOCATION`, `ID`, `FINANCIAL`, `HEALTH`, `AUTH`, `OTHER_PII`. Used when data is built with `ai4privacy_text_only` (Hugging Face) or **`ai4privacy_kaggle_en`** (Kaggle, English only).

- **data.py** writes `arrow_datasets/label_config.json` with `num_labels`, `id2label`, `label2id`, and `source` (e.g. `ai4privacy_text_only`). Only written for ai4privacy modes.
- **train.py** reads `arrow_datasets/label_config.json` to set the model head and metrics. If missing, uses 3-class.
- **infer.py** always exposes low/med/high to callers. For 9-class models it loads `config/pii_to_risk.json` and aggregates 9-class predictions into risk buckets (prob_low, prob_med, prob_high); decision logic and thresholds are unchanged.

---

## Improving the training run

**Dataset (9-class):** Rebalance classes so rare ones (ID, HEALTH, O) get more weight:
```bash
# Cap majority at 5000, oversample minority to 200 (env vars)
set BALANCE_PII_MAX=5000
set BALANCE_PII_MIN=200
python data.py ai4privacy_kaggle_en
```

**Model / training:** Edit `config/train_config.json` (or use env):
- `epochs`, `learning_rate`, `batch_size` – training schedule
- `warmup_ratio`, `weight_decay`, `max_grad_norm` – optimizer
- `classifier_dropout` – TinyBERT head dropout (e.g. 0.1)
- `use_class_weights` – `true` for weighted cross-entropy on 9-class (recommended when imbalanced)

Env overrides: `GUARDRAIL_EPOCHS`, `GUARDRAIL_LR`, `GUARDRAIL_BATCH_SIZE`.

---

## Usage

**Build data (9-class):**
```bash
# Hugging Face (ai4privacy/pii-masking-43k)
python data.py ai4privacy_text_only

# Kaggle English PII (download first: https://www.kaggle.com/datasets/verracodeguacas/ai4privacy-pii/data → extract to data/ai4privacy_kaggle)
python data.py ai4privacy_kaggle_en
```

**Train:**
```bash
python train.py
```

**Infer:**
```python
from infer import score_clipboard
result = score_clipboard("example text")
# result: risk, prob_low, prob_med, prob_high, decision, pii_count, block
```
