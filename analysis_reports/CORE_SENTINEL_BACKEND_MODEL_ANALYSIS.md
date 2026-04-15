# Core Sentinel Backend Model Analysis Report

Author-ready technical summary for academic review.
This report is generated from the current backend implementation and calibration scripts.

## 1) Backend Decision Pipeline

Core Sentinel uses a layered decision stack:

1. Clipboard text preprocessing and sliding-window model inference.
2. Model probability interpretation (TinyBERT softmax outputs).
3. Regex/PII override layer (strong and medium pattern detections).
4. Risk aggregation and scalar risk score computation.
5. Policy enforcement (silent/warn/block), including class-specific action caps.
6. Final UI action gating and telemetry logging.

## 2) Model and Probability-to-Class Mapping

The classifier is a fine-tuned TinyBERT model (`tinybert_guardrail`).

For 3-class outputs (`low`, `med`, `high`), class mapping uses policy thresholds:

- If `prob_high > prob_threshold_high` -> `high`
- Else if `prob_med > prob_threshold_med` -> `med`
- Else -> `low`

Current configured values from `config/risk_policy.json`:

- `prob_threshold_high = 0.75`
- `prob_threshold_med = 0.40`

These values are applied consistently in both inference and calibration helper logic.

## 3) Decision Rules (Allow/Warn/Block)

### 3.1 Strict high-risk branch

When top risk label is `high`, stricter early rules are applied:

- If `pii_count >= 2` and `prob_high >= 0.30` -> `block`
- Else if `prob_high >= 0.35` -> `block`
- Else if `prob_high >= 0.20` -> `warn`

### 3.2 Config-driven branch

If strict branch does not block/warn first:

- If `prob_high >= min_prob_high_block` -> `block`
- Else if `prob_high >= min_prob_high_warn` OR `prob_med >= min_prob_med_warn` -> `warn`
- Else -> `allow`

Default policy loader values:

- `min_prob_high_block = 0.80`
- `min_prob_high_warn = 0.50`
- `min_prob_med_warn = 0.60`

Note: runtime policy may override these via `risk_policy.json`.

## 4) Per-Class Threshold Logic (9-class mode support)

For per-class probabilities, each class has warn/block cutoffs:

- If any class `prob >= class.block` -> `block`
- Else if any class `prob >= class.warn` -> `warn`
- Else -> `allow`

Class overrides in policy can cap action levels (`silent`, `warn`, `block`) per label family.

## 5) Risk Score Formula (0-100)

Scalar risk score is computed as:

`risk_score = base(risk) + sum(trigger_weights)` with cap at 100.

Base score:

- `low -> 0`
- `med -> 30`
- `high -> 70`

Trigger increments (selected):

- SSN/National ID: `+30`
- Credit card / IBAN / routing / financial identifiers: `+25`
- Email / phone / address: `+15`
- IP / device / token / auth / JWT / bearer: `+15`
- Passport number: `+15`
- Confidential marker: `+15`
- Salary info: `+20`

Final clamp:

- `risk_score = min(100, risk_score)`

## 6) Long-Text Window Penalty Heuristic

For multi-window inference, per-window model scores are derived and adjusted to reduce sparse false positives:

- `final_score = max(window_scores)`
- `trigger_rate = (# windows with score > 30) / total_windows`
- If `trigger_rate < 0.25` and `final_score < 80`:
  - `final_score = int(final_score * trigger_rate * 2)`

Additional very-long-text safeguard:

- If tokens >= 180, all windows trigger, minimum window score >= 70, and merged score <= 70:
  - cap `final_score` to `<= 30`

Risk-tier mapping from adjusted scalar:

- `<45 -> low`
- `<70 -> med`
- `>=70 -> high`

## 7) Critical Secret Detection Formula

Critical-secret path can force high-risk/block behavior independent of normal probability thresholds.

### 7.1 Pattern-based

Regex signatures include API keys/tokens/password patterns such as:

- OpenAI/GitHub/AWS-like keys
- JWT/Bearer tokens
- Password assignment patterns
- Private key blocks
- DB connection strings with credentials

### 7.2 Entropy-based

For long alphanumeric tokens (`length >= 36`), Shannon entropy per character is computed:

`H = -sum(p_i * log2(p_i))`

If `H >= min_token_entropy_bits`, classify as critical secret.

Current default/active entropy threshold:

- `min_token_entropy_bits = 4.2`

## 8) Final Action Enforcement Rules

After merging decisions and score:

1. Critical secret -> always `block`.
2. If `risk_score >= block_threshold`, force `block`.
   - Current runtime `block_threshold` in config: `80`.
3. Low risk is forced to `silent`.
4. Medium risk cannot stay `block`; downgraded to `warn`.
5. Contact-only high cases are capped:
   - If contact-only and `risk_score > 70`, set `risk_score = 70` and action `warn`.
6. Final silent/warn gate:
   - If action is already `block`, keep block.
   - Else if `risk_score > 30`, action `warn`.
   - Else action `silent`.

## 9) Measurements and Evaluation Methodology

Evaluation pipeline (`evaluate_test.py`) computes:

- Accuracy
- Macro F1
- Per-class precision/recall/F1
- PR curves per class
- AUPRC per class
- Threshold at precision >= 0.85
- Recall at that threshold
- Confusion matrix
- Classification report

PR summary is written to:

- `core-sentinel-guardrail/reports/pr_curve_summary.json`

## 10) Threshold Calibration Method

Calibration script (`calibrate_thresholds.py`) uses two paths:

1. **PR-summary application mode**
   - Reads `pr_curve_summary.json`
   - Sets per-class:
     - `block = threshold_at_precision_085`
     - `warn = max(0.30, block - 0.15)`

2. **Validation sweep mode**
   - Sweeps `(th, tm)` over `[0.05, 0.95]` grid
   - Objective: maximize high-risk recall under `FPR <= max_fpr` (default 5%)
   - Reports recommended thresholds and resulting recall/FPR.

## 11) Latency Measurement Method

`benchmark_latency.py` measures wall-clock end-to-end inference latency for:

- tokenize -> model forward -> regex override -> policy decision

Reported metrics:

- Mean latency (ms)
- P50, P95, P99 (ms)

Method notes:

- Warmup iterations (default 10) to stabilize runtime
- Iterative scoring with unique text suffixes to avoid cache effects

## 12) What to Tell a Professor (Concise)

Core Sentinel is not a single-threshold classifier. It is a policy-driven ensemble pipeline:

- TinyBERT posterior probabilities give semantic risk.
- Regex and secret heuristics provide deterministic high-recall guardrails.
- A calibrated risk policy maps probabilities to actions with class-specific overrides.
- A scalar explainable score (0-100) is computed with explicit trigger weights.
- Final decisions are constrained by safety rules (critical secret escalation, medium no-block rule, contact-only cap, and silent threshold gate).
- Evaluation and calibration are reproducible via PR/AUPRC, confusion matrices, and FPR-capped threshold sweeps.

---

Generated for academic presentation from current repository implementation.
