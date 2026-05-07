# Core Sentinel ML system — reference (values in git + formulas)

This document summarizes **what is stored in the repository** (JSON configs, reports, website records), **which base model is used**, **training defaults**, **metric definitions**, and **runtime risk-score formulas**. Fine-tuned **model weight files** under `core-sentinel-guardrail/models/` are **not** committed (see `.gitignore`); they are produced locally after `train.py` and loaded at inference from that path or equivalent.

**Primary snapshot file for the website:** `docs/data/model_records.json`  
**Generated:** `2026-04-17T16:00:58.327584+00:00` (UTC, per file)

---

## 1. Model architecture and identifiers

| Item | Value |
|------|--------|
| **Base encoder (Hugging Face)** | `huawei-noah/TinyBERT_General_4L_312D` |
| **Override** | Environment variable `GUARDRAIL_MODEL_ID` |
| **Head** | `transformers.BertForSequenceClassification` with `num_labels` = **2** (binary: safe / risky) or **3** (low / med / high) depending on dataset / `GUARDRAIL_BINARY_MODE` |
| **Binary label mapping** | `0 → safe`, `1 → risky` (`train.py`: `ID_TO_RISK_2`) |
| **3-class label mapping** | `0 → low`, `1 → med`, `2 → high` (`RISK_TO_ID` / `ID_TO_RISK_3` in `train.py`) |
| **Fine-tuned save directory (local)** | `core-sentinel-guardrail/models/tinybert_guardrail` (`SAVE_DIR` in `train.py`) — **not in git** |

Tokenizer: `AutoTokenizer.from_pretrained(MODEL_ID)` (same Hub id as the encoder).

---

## 2. Training configuration (as committed + binary overrides in code)

### 2.1 `core-sentinel-guardrail/config/train_config.json` (3-class-oriented defaults)

| Field | Value |
|-------|--------|
| `epochs` | 6 |
| `learning_rate` | 3e-5 |
| `batch_size` | 16 |
| `warmup_ratio` | 0.1 |
| `weight_decay` | 0.01 |
| `max_grad_norm` | 1.0 |
| `classifier_dropout` | 0.1 |
| `use_class_weights` | true |
| `class_weight_mode_3class` | balanced |
| `rebalance_train_3class` | true |
| `rebalance_mode_3class` | min |
| `rebalance_seed` | 42 |

`train.py` also defines module-level defaults (e.g. `LR = 2e-5`, `EPOCHS = 6`) merged with this file; environment variables can override (see `load_train_config()` in `train.py`).

### 2.2 Binary mode (when `GUARDRAIL_BINARY_MODE` is on, or dataset has two classes)

`train.py` **overrides** training settings to:

| Setting | Value |
|---------|--------|
| `epochs` | 4 |
| `learning_rate` | 1e-5 |
| `weight_decay` | 0.01 |
| `classifier_dropout` | 0.3 |
| `label_smoothing_factor` | 0.1 |
| `early_stopping_patience` | 2 |
| `use_class_weights` | false |
| `rebalance_train_3class` | false |

### 2.3 Trainer metrics (validation)

From `make_compute_metrics()` in `train.py`:

- **Accuracy:** `sklearn.metrics.accuracy_score(y_true, y_pred)` with `y_pred = argmax(logits)`.
- **Macro precision / recall / F1:** `sklearn.metrics.precision_recall_fscore_support(..., average="macro", zero_division=0)`.
- **Per-class** precision, recall, F1: same function with `average=None`, `labels=range(num_labels)`.

Best checkpoint: `metric_for_best_model` from config (default `eval_f1`), `greater_is_better=True`, early stopping on validation.

---

## 3. Dataset numbers (published to website)

From `docs/data/model_records.json` → `dataset`:

| Field | Value |
|-------|--------|
| `train_samples` | 10,455 |
| `val_samples` | 485 |
| `test_samples` | 2,289 |
| `total_samples_approx` | 13,229 |
| `label_balance_train` | low 3,485 / med 3,485 / high 3,485 |
| `pii_categories` | 24 |
| `data_sources` | 6 |
| `source_names` | NVIDIA Nemotron-PII pool; Enron email pool; Faker business synthetic; safe negatives; Patronus stub set; user feedback merges |

**Note:** Some reports (e.g. PR/ROC summaries) record a different **evaluation row count** (`test_split_size`: 3,391 in `reports/pr_curve_summary.json`) than `test_samples` above. Treat **13,229 / 2,289** as the high-level dataset story in `model_records.json`, and **3,391** as the size of the specific split used when that report was generated.

---

## 4. Metrics currently in `docs/data/model_records.json`

### 4.1 Holdout / threshold-check style metrics (`metrics`)

These align with **`core-sentinel-guardrail/reports/binary_threshold_check.json`** at **recommended threshold 0.45**:

| Metric | Value |
|--------|--------|
| `accuracy` | 0.9121203184901209 |
| `precision` | 0.9446945337620579 (risky-positive precision in that report) |
| `recall` | 0.9588772845953003 |
| `f1` | 0.9517330741820538 |
| `macro_f1` | 0.7308007476173427 |
| `safe_recall` | 0.4740061162079511 |
| `fpr_non_high_as_high` | 0.5259938837920489 |
| `auc` | 0.7514706327301757 |
| `risky_auprc` | 0.9890162341941844 |
| `recommended_threshold` | 0.45 |
| `selection_rule` | max macro_f1, then safe_recall, then risky_precision, then min fpr |

### 4.2 Calibration sweep metrics (`calibration`)

From validation sweep in **`reports/threshold_calibration.json`** (binary mode), **selected** row at threshold **0.1**:

| Metric | Value |
|--------|--------|
| `prob_threshold_risky` | 0.1 |
| `accuracy` | 0.9992138364779874 |
| `precision` | 0.9984301412872841 |
| `recall` | 1.0 |
| `f1` | 0.9992144540455618 |
| `fpr` | 0.0015723270440251573 |
| `selection_rule` | max_recall_under_fpr_cap |

### 4.3 Release gate flags (`release_gate`)

| Field | Value |
|-------|--------|
| `gate_passed` | false |
| `promote_allowed` | false |
| `recommended_rollout` | canary_10pct |

### 4.4 Artifact sources referenced by `model_records.json`

| Key | Path |
|-----|------|
| holdout threshold check | `core-sentinel-guardrail/reports/binary_threshold_check.json` |
| calibration | `core-sentinel-guardrail/reports/threshold_calibration.json` |
| PR curve | `core-sentinel-guardrail/reports/pr_curve_summary.json` |
| release gate (unseen) | `core-sentinel-guardrail/reports/release_readiness_report_unseen.json` |

---

## 5. Formulas: binary threshold metrics (calibration & threshold checks)

Implemented in `calibrate_thresholds.py` as `_binary_metrics_at_threshold()`:

- Labels: `y_true = (label > 0)` (risky = positive class).
- Predictions: `y_pred = (P(risky) >= threshold)`.

Confusion counts:

- `TP` = predicted risky and true risky  
- `TN` = predicted safe and true safe  
- `FP` = predicted risky but true safe  
- `FN` = predicted safe but true risky  

Then:

\[
\text{precision} = \frac{TP}{TP + FP} \quad (\text{if } TP+FP>0)
\]

\[
\text{recall} = \frac{TP}{TP + FN} \quad (\text{if } TP+FN>0)
\]

\[
F1 = \frac{2 \cdot \text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}} \quad (\text{if } \text{precision} + \text{recall} > 0)
\]

\[
\text{accuracy} = \frac{TP + TN}{N}
\]

\[
\text{FPR} = \frac{FP}{FP + TN} \quad (\text{if } FP+TN>0)
\]

`binary_threshold_check.json` additionally reports **safe_recall**, **risky_precision**, etc., derived from the same confusion matrix for the binary problem.

**Why two different FPR values appear:** Validation calibration FPR (~0.16%) and holdout “FPR” (~52%) refer to **different datasets and definitions** (validation vs. larger holdout / different label mix). The UI separates “calibration FPR” vs “holdout FPR” for that reason.

### 5.1 What to do when reviewers compare those two FPRs (professor confusion)

They are **not two estimates of the same quantity**. Treat them as **two different experiments**:

| Label you should use aloud | JSON / UI location | What it measures |
|----------------------------|-------------------|------------------|
| **Validation / calibration FPR** | `model_records.json` → `calibration.fpr` (and related threshold sweep on **val**) | False-alarm rate among **non-risky** labels when τ is chosen **on the validation split** to satisfy the calibration objective (small **n**, often easier distribution). |
| **Holdout / test FPR** | `model_records.json` → `metrics.fpr_non_high_as_high` (from `binary_threshold_check.json`) | Same **formula** \(FP/(FP+TN)\) but on the **held-out test split** at the **recommended binary threshold** τ — often **harder / more imbalanced**, so the number can be much larger. |

**Do / say this:**

1. **Never** imply “calibration FPR is wrong because it doesn’t match holdout FPR.” They are **different splits and often different τ contexts**.
2. **Always pair a number with its label:** say “**holdout test FPR at τ = 0.45**” vs “**validation FPR at the calibrated τ**,” not “the FPR.”
3. **One slide / table:** show **both** columns side by side with full names; add a footnote: *same formula, different data and threshold selection step*.
4. **If someone asks which is “the real” FPR:** for **deployment honesty**, prioritize **holdout / test** metrics for **generalization**; use **calibration** metrics for **how τ was chosen** during training (tuning), not as a second headline that must match.

**Optional UI:** `docs/ml/index.html` already pulls **both** into the FPR logic (`fprHoldout` vs `fprCalibration`); when demoing, expand or caption which number is on screen.

---

## 6. PR / ROC summaries (committed excerpts)

From `reports/pr_curve_summary.json` (generated `2026-04-17T15:07:12`):

| Class | `auprc` | Notes |
|-------|---------|--------|
| safe | 0.5139250312661671 | |
| risky | 0.9890162341941844 | Matches `risky_auprc` in `model_records.json` |

`roc_auc_macro` in that file is `null` (macro ROC AUC not populated for this run).

---

## 7. Training run summary on disk (`reports/train_eval_summary.json`)

Validation metrics after training (best/last entries — **on the validation split**, not holdout test):

| Metric (eval) | ~Value |
|---------------|--------|
| `eval_accuracy` | 0.9992138364779874 |
| `eval_f1` | 0.9992138359920963 |
| `eval_precision_safe` / `eval_recall_risky` | etc. | See file for full list |

These match the **calibration** block’s validation-scale numbers when the same split is used; they are **not** a substitute for holdout test metrics in §4.1.

---

## 8. Runtime policy and thresholds (`config/risk_policy.json`)

Committed values (abridged; see file for full JSON):

| Area | Committed values |
|------|------------------|
| `block_threshold` | 80 |
| `inference_binary_labels.prob_threshold_risky` | **0.1** |
| `inference_binary_labels.max_fpr` | 0.6 |
| `inference_binary_labels.selected_by` | max_recall_under_fpr_cap |
| `strict_decision_rules.recall_first_mode` | true |
| `strict_decision_rules.medium_multi_signal_block_min` | 2 |
| `strict_decision_rules.tiny_text_min_chars` | 12 |
| `strict_decision_rules.tiny_text_min_confidence` | 0.85 |
| `strict_decision_rules.tiny_text_requires_signal` | true |
| `med.action_override` (under `thresholds.med`) | warn_only |
| `critical_secret.min_token_entropy_bits` | 4.2 |
| `_calibrated_at` | 2026-04-17T14:52:28.326488+00:00 |

Per-tier `thresholds` (low/med/high/safe/risky) include PR-derived fields such as `warn`, `block`, `auprc`, `derived_from` — used by broader policy and legacy 3-class paths; **binary runtime thresholding** for `P(risky)` is governed by `inference_binary_labels` as above.

---

## 9. Release gate configuration (`config/release_gate.json`)

| Criteria | Value |
|----------|--------|
| `blind_pass_rate_min` | 0.92 |
| `blind_medium_warn_recall_min` | 0.85 |
| `blind_risky_block_recall_min` | 0.98 |
| `safe_false_positive_rate_max` | 0.03 |

Canary: start 10%, step 10%, 24 h windows, 2 consecutive passes. Rollback thresholds listed in file.

---

## 10. Runtime risk score (0–100) — formulas in `infer.py`

### 10.1 Heuristic `compute_risk_score(risk, triggers)`

- Base: **low** → 0, **med** → 30, **high** → 70.
- Additives from trigger string matching (capped at 100), including e.g.:
  - passport / confidential / salary / business context: +10–20
  - SSN / national ID: +30
  - financial identifiers: +25
  - email / phone / address: +15
  - IP / device / token / JWT / bearer: +15

### 10.2 Binary model path (`score_clipboard_with_pii`)

1. `p_risky` = softmax probability for label **risky**.
2. `merged = round(p_risky × 100)` clipped to [0, 100].
3. `severity_floor = compute_risk_score(risk, triggers)` from regex/NER + policy.
4. **`merged_score_for_penalty = max(merged, severity_floor)`** so rules can raise the score when the model is underconfident.
5. Further adjustments (caps/floors): e.g. HIPAA-derived critical-label paths may force high scores; contact-only cap; tiny-text downgrades; strict multi-signal block rules — see `infer.py` around `risk_score = merged_score_for_penalty`.

### 10.3 UI action vs score

- `RISK_SCORE_SILENT_MAX = 30`: if final action is not already `block`, scores **above 30** tend to yield **warn** via `_finalize_action_for_risk_score` (critical secrets force **block**).

### 10.4 Optional dynamic score (3-class probability blend)

If `GUARDRAIL_DYNAMIC_RISK_SCORE` is enabled, `_expected_risk_score_from_3class_probs` uses:

\[
\text{score} \approx \bigl( p_{high} \cdot 1.0 + p_{med} \cdot 0.52 + p_{low} \cdot 0.08 \bigr) \times 100
\]

(rounded to int), optionally blended with tier-only scores — see `infer.py`.

---

## 11. Blind eval snapshot (`reports/release_readiness_report_unseen.json`)

Example committed summary (see file for full `mismatches`):

| Metric | Value |
|--------|--------|
| `pass_rate` | 0.875 (105/120) |
| `safe_false_positive_rate` | 0.0 |
| `medium_warn_recall` | 0.625 |
| `risky_block_recall` | 1.0 |

---

## 12. Benchmark delta table (professor Q&A ready)

From `core-sentinel-guardrail/benchmark_report_hipaa_full.txt` on the full Arrow test fallback (`n=3391`).
Baseline is **Regex+NER**; deltas are shown in **percentage points (pp)** and relative `%`.

| Metric | Regex+NER | ML val-calib | ML test-rec | Full val-calib | Full test-rec |
|--------|-----------|--------------|-------------|----------------|---------------|
| Accuracy | 0.9030 | 0.8832 (**-1.98 pp**, -2.19%) | 0.8641 (**-3.89 pp**, -4.31%) | 0.9071 (**+0.41 pp**, +0.45%) | 0.9068 (**+0.38 pp**, +0.42%) |
| Precision | 0.9108 | 0.9270 (**+1.62 pp**, +1.78%) | 0.9437 (**+3.29 pp**, +3.61%) | 0.9077 (**-0.31 pp**, -0.34%) | 0.9087 (**-0.21 pp**, -0.23%) |
| Recall (TPR) | 0.9896 | 0.9452 (**-4.44 pp**, -4.49%) | 0.9034 (**-8.62 pp**, -8.71%) | 0.9987 (**+0.91 pp**, +0.92%) | 0.9971 (**+0.75 pp**, +0.76%) |
| F1 | 0.9485 | 0.9360 (**-1.25 pp**, -1.32%) | 0.9231 (**-2.54 pp**, -2.68%) | 0.9510 (**+0.25 pp**, +0.26%) | 0.9508 (**+0.23 pp**, +0.24%) |
| F2 | 0.9727 | 0.9415 (**-3.12 pp**, -3.21%) | 0.9112 (**-6.15 pp**, -6.32%) | 0.9791 (**+0.64 pp**, +0.66%) | 0.9780 (**+0.53 pp**, +0.54%) |
| FNR (lower is better) | 0.0104 | 0.0548 (**+4.44 pp**, +426.9% worse) | 0.0966 (**+8.62 pp**, +828.8% worse) | 0.0013 (**-0.91 pp**, -87.5% better) | 0.0029 (**-0.75 pp**, -72.1% better) |
| AUC-ROC | 0.5406 | 0.8462 (**+0.3056**, +56.5%) | 0.8462 (**+0.3056**, +56.5%) | 0.5409 (**+0.0003**, +0.1%) | 0.5409 (**+0.0003**, +0.1%) |
| AUPRC | 0.9107 | 0.9789 (**+0.0682**, +7.49%) | 0.9789 (**+0.0682**, +7.49%) | 0.9108 (**+0.0001**, +0.01%) | 0.9108 (**+0.0001**, +0.01%) |

Professor-facing interpretation:

- ML-only increases precision/ranking (AUC/AUPRC) but loses recall at stricter operating points on this pattern-heavy slice.
- Full OR-fusion recovers and slightly improves recall/F1 over Regex+NER baseline.
- On this fallback benchmark, true negatives are relatively scarce; quote FPR/Specificity with that caveat.

---

## 13. Files to update when you retrain

1. Run training, evaluation, calibration, and `publish_website_records.py` (or your CI/sync path) so **`docs/data/model_records.json`** and **`reports/*.json`** refresh.  
2. **`config/risk_policy.json`** is updated by `calibrate_thresholds.py --apply` when calibration selects a new binary threshold.  
3. **Model weights** remain local under `models/tinybert_guardrail/` (not in git); document the run id / commit in your release notes if needed.

---

*This file is a static export of repository state and code-defined formulas. Regenerate or edit it after major metric or policy changes if you want the doc to stay in sync.*
