# Core Sentinel: Professor Technical Report (ML, Thresholds, and Detection Logic)

## 1) Executive answer to your professor’s core question

Core Sentinel is a **hybrid risk engine**. It does not rely on one detector.

It combines:
1. **Deterministic pattern detection** (regex + policy rules, including HIPAA-critical IDs),
2. **Entity/Span signals** (NER-style signals available in benchmark mode),
3. **TinyBERT classification** (binary or 3-class depending on training mode),
4. **Policy fusion** (strict decision rules + risk score gates + class overrides).

So the deployed behavior is not “ML only” and not “regex only”. It is a policy-fused system tuned for safety.

---

## 2) What exactly is the ML model and how does it work?

### 2.1 Model family
- Base encoder: TinyBERT (`huawei-noah/TinyBERT_General_4L_312D`) fine-tuned for sequence classification.
- Two operating label modes are supported in this project:
  - **Binary:** `safe` vs `risky`.
  - **3-class:** `low`, `med`, `high`.

### 2.2 Inference math (softmax)
Given logits $z_i$, probability for class $i$ is:

$$
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

### 2.3 Long text handling
Text is chunked into overlapping windows (max sequence length around 128 tokens by default, stride overlap configurable). Each window is scored, then merged using a “worst-window” rule (prefer high-risk window evidence).

Conceptually:

$$
p(y\mid x) = \text{merge}\big(p(y\mid w_1), p(y\mid w_2), \dots, p(y\mid w_n)\big)
$$

### 2.4 Decision conversion inside runtime
For 3-class mode, class probabilities are converted to buckets using calibrated thresholds:
- if $p_{high} > \tau_h$ => high
- else if $p_{med} > \tau_m$ => med
- else => low

For binary mode:
- if $p_{risky} \ge \tau_r$ => risky
- else => safe

But that is only the first stage. Final action is post-processed by risk policy logic.

---

## 3) Nemotron dataset: how labeling is defined

Nemotron rows contain text and span metadata. The project converts spans to risk labels via rule mapping:

- Parse spans for each row.
- If a span contains `critical_secret` => label as **high**.
- Else if critical types appear (`ssn`, `credit_card`, `api_key`, `password`) => **high**.
- Else if span count $\ge 4$ => **high**.
- Else if span count $\ge 2$ => **med**.
- Else => **low**.

So Nemotron label generation is **programmatic weak supervision from spans**, not manual human labeling per row.

### 3.1 Nemotron split behavior in pipeline
- Nemotron `train` and `test` are loaded.
- A validation split is formed from train using stratified split when possible.
- Optional caps are applied for train/val/test sizes.

---

## 4) How training is done

### 4.1 Pipeline summary
1. Build dataset pool from Nemotron + other real/synthetic sources (depending on selected data source mode).
2. Normalize labels to project risk schema.
3. Split train/validation/test with stratification where possible.
4. Tokenize with TinyBERT tokenizer.
5. Train with weighted cross-entropy (class-weight support) and optional rebalancing.
6. Early stopping and metric-based model selection.

### 4.2 Loss with class weighting
Weighted cross-entropy is used when enabled:

$$
\mathcal{L} = -\sum_{c} w_c \, y_c \log p_c
$$

where $w_c$ are class weights (manual or derived from class frequencies).

### 4.3 Why this matters
Because risk data is naturally imbalanced (many non-sensitive, fewer severe sensitive patterns), weighting/rebalancing prevents the model from collapsing to majority-class behavior.

---

## 5) What is eval vs validation vs holdout vs blind/unseen?

This repository uses multiple evaluation concepts:

1. **Validation set**
   - Used during calibration and model tuning.
   - Example: threshold sweep for binary calibration uses validation outputs.

2. **Test set**
   - Standard post-training evaluation set.
   - Used for PR/ROC and classification reports.

3. **Source holdout (data pipeline mode)**
   - In `multi_real_synthetic`, selected data sources can be excluded from train and kept as holdout test to measure domain shift.

4. **Blind set / unseen blind set (release gate)**
   - Operational gate datasets used before release decisions.
   - Seen/curated blind can pass while unseen blind can fail (this happened in artifacts).

So “eval” is not one thing; each set answers a different question.

---

## 6) How threshold values are measured

## 6.1 Binary threshold calibration objective
Binary threshold sweep evaluates grid thresholds and computes:
- accuracy,
- precision,
- recall,
- F1,
- FPR.

Then it applies selection rule:
- **maximize recall under FPR cap** (current default).

Current calibration artifact shows:
- $n_{val}=1272$
- FPR cap = 0.05
- selected threshold $\tau_r = 0.1$
- recall = 1.0
- FPR ≈ 0.00157

This is explicitly recall-first safety tuning.

## 6.2 PR-derived thresholds
From test PR curve summary, threshold at target precision is recorded per class.
For risky class in current artifact:
- risky AUPRC ≈ 0.9890
- threshold at precision 0.85 ≈ 0.0493
- recall at that threshold = 1.0

## 6.3 Combined runtime thresholds (not a single scalar)
Combined runtime decision is multi-gate, including:
- `inference_binary_labels.prob_threshold_risky = 0.1`
- `inference_3class_labels.prob_threshold_high = 0.825`
- strict rules (e.g., high block/warn probs and multi-signal rules)
- risk score gates (`block_threshold = 80`, silent/warn score gate)
- class overrides and hard triggers.

So comparing “combined threshold vs ML threshold” is not 1-to-1:
- **ML-only threshold** = one probability cutoff (e.g., 0.1 risky).
- **Combined threshold** = policy graph with multiple cutoffs and override rules.

---

## 7) ROC: how values are determined

ROC is computed from scores and labels by sweeping threshold across score range.

For positive class (high or risky depending mode):

$$
TPR = \frac{TP}{TP + FN}, \quad FPR = \frac{FP}{FP + TN}
$$

Each threshold gives one $(FPR,TPR)$ point. The curve is all points; AUC is area under it.

In this project, ROC for high-class route is built from softmax probability of positive class and true binary labels (one-vs-rest style for that class).

---

## 8) MER/HIPAA detection and MRN detection

("MER" interpreted as medical-entity/HIPAA record detection path.)

### 8.1 Is MRN detected by ML or regex?
MRN is explicitly captured by HIPAA-critical regex rule:
- pattern includes `mrn` / `medical record number` + 5–12 digits.
- when matched, it is tagged as critical trigger and forced toward block path.

So MRN detection is primarily deterministic regex in current runtime path.

### 8.2 HIPAA-specific critical patterns in policy path
Runtime includes HIPAA-critical regex triggers for:
- MRN,
- DEA,
- NPI,
- Health Plan ID,
- age > 89 cues.

These triggers can force high risk / block regardless of model uncertainty.

### 8.3 Do we have HIPAA regex?
Yes. HIPAA-related regexes are present and active in runtime logic.

---

## 9) ML-only vs Regex+NER-only vs Combined

## 9.1 Conceptual comparison

### A) Regex+NER-only
Strengths:
- deterministic,
- low false positive for known patterns,
- excellent for fixed formats (SSN, emails, API key patterns).

Weaknesses:
- misses contextual/implicit risk,
- misses unseen formats/paraphrases,
- brittle against adversarial formatting.

### B) ML-only
Strengths:
- captures semantic/contextual risk,
- generalizes beyond handcrafted patterns.

Weaknesses:
- can raise false positives on ambiguous business text,
- can miss specific structured secrets without explicit cues,
- requires threshold tuning and continuous calibration.

### C) Combined
Strengths:
- deterministic hard blocks for critical known patterns,
- ML catches contextual/unseen cases,
- policy lets you tune recall/FPR tradeoff operationally.

Weaknesses:
- more complex behavior (multi-threshold),
- requires careful governance to avoid over-blocking.

## 9.2 What happens if ML is removed?
If ML is removed and only regex + NER remain:
- You retain high precision on pattern-identifiable secrets.
- You lose semantic/context detection and generalization.
- Medium-risk contextual statements (without explicit patterns) are more likely to be missed.

In short: ML is significant for recall on non-pattern cases and robustness to phrasing variation.

---

## 10) Interpreting current benchmark artifacts carefully

The benchmark script includes five modes (Regex+NER, ML-only calibrated/holdout, Full calibrated/holdout).

Important nuance:
- In benchmark implementation, Full mode multiplies ML score by regex/NER prefilter (hard gate). If prefilter misses, ML is effectively muted there.
- Runtime product path is richer than that benchmark gate; runtime does not reduce to a single `score * prefilter` formula.

So benchmark “full” numbers should be presented with this caveat to avoid misrepresenting deployed behavior.

---

## 11) Current threshold tradeoff rationale (what matters most)

The current project priorities are safety-first:
1. keep risky recall very high,
2. constrain FPR to acceptable cap,
3. enforce hard block on critical secrets/HIPAA-critical IDs,
4. keep medium-risk behavior warn-oriented,
5. suppress nuisance alerts for tiny benign snippets via strict tiny-text logic.

This is why binary calibration selected very low risky threshold (0.1): it preserved full recall under FPR cap in validation artifact.

---

## 12) Binary model role: why it helps

Binary model (`safe` vs `risky`) simplifies operational control:
- easier thresholding,
- stronger recall-focused calibration,
- clearer release gating metrics.

What matters most in binary mode:
- risky recall,
- safe-class false positive rate,
- calibration stability on unseen blind sets.

This repository’s release reports show exactly this tradeoff: risky recall remains strong while medium-warning sensitivity can drop on unseen data.

---

## 13) How values in release gate are defined

Release gate criteria currently check:
- blind pass rate minimum,
- medium warn recall minimum,
- risky block recall minimum,
- safe false positive rate maximum.

Configured targets include:
- blind pass rate min 0.92,
- medium warn recall min 0.85,
- risky block recall min 0.98,
- safe false positive max 0.03.

Observed outcomes:
- curated blind report: gate passed,
- unseen blind report: gate failed (mostly medium warn recall shortfall), rollout recommendation: canary.

---

## 14) Backend connection flow (detailed, not just file names)

1. Clipboard text enters scoring pipeline.
2. Critical secret and HIPAA-critical regex checks run early.
3. TinyBERT produces class probabilities on sliding windows.
4. Window outputs are merged; optional long-text penalty logic prevents one-window spikes from dominating.
5. Regex/KB triggers and model signal are fused into risk tier + risk score.
6. Policy layer applies strict rules, class ceilings, and UI action mapping.
7. Final action (`silent`/`warn`/`block`) is emitted.
8. Event telemetry is recorded with hashed text and context metadata.
9. Offline retraining/calibration updates thresholds and policy for next cycle.

---

## 15) Validation and dataset definitions used in labeling/training

### 15.1 Data families used in this repository
- Nemotron PII (span-derived weak labels),
- Patronus/enterprise-style labeled data,
- BigCode PII,
- business/enron-like real corpora,
- kaggle sensitive,
- financial PII tables,
- ai4privacy variants,
- synthetic hard negatives / adversarial risky samples,
- user feedback merges.

### 15.2 Label normalization
Labels are normalized into project risk schema (`low/med/high` or binary mapping).

### 15.3 Rebalancing and floors
Pipeline includes:
- class rebalancing,
- min-count enforcement for eval splits,
- optional source holdout coverage enforcement,
- optional synthetic top-up where needed.

This is meant to avoid misleading metrics caused by class collapse.

---

## 16) Direct answers to likely professor viva questions

### Q: Does ML play a significant role?
Yes, especially for non-pattern contextual risk and recall beyond deterministic regex patterns.

### Q: Is MRN caught by model?
Primarily by HIPAA-critical regex in current runtime; model can contribute context but deterministic trigger is the strong path.

### Q: How were thresholds chosen?
By validation/test sweeps with explicit objective functions (recall-first under FPR cap for binary) and PR-derived thresholds for class policies.

### Q: What is combined threshold compared to ML-only threshold?
ML-only uses one probability cutoff; combined uses multi-stage policy with multiple thresholds, strict rules, and override gates.

### Q: What is eval vs holdout?
Validation = tuning/calibration; test = standard evaluation; holdout/blind/unseen = generalization and release-readiness checks.

---

## 17) Final technical conclusion

The project is a **policy-governed hybrid guardrail**. The ML model is not optional if your goal is broad contextual detection and better recall on unseen phrasing. Regex and HIPAA-critical patterns are essential for deterministic hard-block safety, while ML provides the generalization layer. The threshold system is intentionally recall-first and is controlled by validation constraints plus release gates to manage operational risk.

---

## 18) Solid examples: why ML improves recall on non-pattern and phrasing variation

Below are concrete examples you can present directly.

### Example A — Non-pattern contextual risk (no explicit hard regex token)

Text:
"Please summarize this employee compensation sheet with full names, monthly pay, bonus, and bank details before I send it to the assistant."

- Regex-only likely outcome:
   - May miss or under-score if no explicit account number/card/SSN pattern appears.
- ML contribution:
   - Sees semantic co-occurrence of compensation + personal identifiers + banking context.
- Combined outcome:
   - Higher chance of warn/block than regex-only for this kind of implicit leakage.

### Example B — Phrasing variation of same risky intent

Text 1:
"Wire request: SWIFT BOFAUS3N with account number 100000000123."

Text 2 (paraphrase):
"Process an international transfer using BOFAUS3N and beneficiary account 100000000123."

- Regex-only:
   - Usually catches Text 1 strongly due to known tokens.
   - May degrade on Text 2 if formatting/keywords deviate from strict pattern expectations.
- ML contribution:
   - Learns transfer/banking semantics and keeps risky score high across paraphrases.
- Combined:
   - Keeps deterministic strength when regex hits and improves robustness when wording shifts.

### Example C — Medium-risk personal context not always regex-hard

Text:
"Ariana from HR is on medical leave and available on personal email for urgent updates."

- Regex-only:
   - Might catch only email token and produce weak/partial signal.
- ML contribution:
   - Captures person + department + health context as sensitive personal narrative.
- Combined:
   - Better warn recall for human-written narrative compared with pattern-only rules.

### Example D — Obfuscated formatting to evade simple patterns

Text:
"Customer SSN is 123 45 6789 and alternate format 123*45*6789 in legacy notes."

- Regex-only:
   - Can fail if obfuscation/spacing variants are not covered by exact regex.
- ML contribution:
   - Context around identity fields can still push risk upward.
- Combined:
   - Best chance to maintain recall under adversarial formatting.

### Example E — Why unseen-set failures still prove ML necessity

From your unseen release report, many medium rows were predicted silent (examples like:
"Jonas from Support will join the planning review...").

Interpretation for viva:
- This does **not** mean ML is unnecessary.
- It shows threshold/domain-shift calibration still needs improvement for medium narrative cases.
- Even with that gap, risky-block recall remained high in the same unseen report.

### Example F — Quantitative proof from your benchmark artifact

From current benchmark output:
- Regex+NER recall is very low on evaluated candidates.
- ML-only recall is dramatically higher.

How to say this to professor:
"Pattern systems are precision-strong for known formats but recall-weak for contextual text. ML adds the recall layer and robustness to paraphrase; combined policy preserves deterministic hard blocks while extending coverage."
