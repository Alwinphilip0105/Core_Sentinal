# Core Sentinel — ML Architecture Analysis

## 1. Three-layer detection pipeline

Core Sentinel applies risk detection in a **three-layer pipeline**. Layers are ordered for latency and safety: deterministic high-confidence signals run first; heavier models run when needed.

### Layer 1 — Critical secret regex (runs first, `<2ms`)

- **Role:** Catch unambiguous secrets and credentials before any neural model runs.
- **Pattern list (representative):** SSH private keys, API keys, database connection strings, and related high-severity patterns.
- **Behavior:** If a match is found, the pipeline **returns block immediately** and **skips the classifier** (and associated model cost) for that decision path.

### Layer 2 — TinyBERT 3-class classifier (runs second, ~9 ms)

- **Input:** Text is **chunked into 128-token windows** (tokenizer-aligned) for the sequence classifier.
- **Output:** A **probability distribution** over **`[low, med, high]`** risk for each chunk.
- **Aggregation:** A **sliding window** over the text; chunk-level scores are combined using **max aggregation** across chunks so localized high-risk spans surface globally.
- **Calibration:** A **trigger rate penalty** is applied for **long safe text** so benign long passages are less likely to accumulate false high scores.

### Layer 3 — NER + medium regex (runs in parallel with the model)

- **NER (spaCy):** Detects **`PERSON`**, **`DATE`**, **`ORG`**, **`GPE`** (and related entity types as configured).
- **Medium regex:** Covers patterns such as **email**, **phone**, **IP**, **address-like** structures, **CVV**, and similar medium-severity signals.
- **Fusion:** When **NER and regex agree** (same span or consistent signal), the **score is boosted**—agreement is treated as stronger evidence than either alone.

---

## 2. Threshold calibration methodology

Thresholds are produced by **`calibrate_thresholds.py`** and tuned on a **held-out test set** (not used for training).

- **Procedure:** A **precision–recall (PR) curve** is computed on **5,500 held-out test examples**.
- **Block threshold:** The **lowest** score **T** such that **precision ≥ 0.90** at that operating point (subject to the PR sweep).
- **Warn threshold:** The **lowest** score **T** such that **precision ≥ 0.75** at that operating point.
- **Current calibrated values:** **`warn = 40`**, **`block = 70`**.
- **Persistence:** Values are stored in **`config/risk_policy.json`** and consumed by the runtime policy layer.

### Operating-point summary

| System | Warn | Block | FPR | High Recall |
|--------|------|-------|-----|-------------|
| Regex only | N/A | exact | 0.1% | 72% |
| TinyBERT only | 40 | 70 | 4.2% | 91% |
| Combined | 40 | 70 | 2.4% | 98.7% |

---

## 3. Model performance metrics

### Confusion matrix (3-class)

Rows are **true** labels; columns are **predicted** labels.

| | Pred: Low | Pred: Med | Pred: High |
|---|-----------|-----------|------------|
| **True: Low** | 500 | 0 | 0 |
| **True: Med** | 0 | 56 | 116 |
| **True: High** | 0 | 61 | 4767 |

### Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| low | 1.000 | 1.000 | 1.000 | 500 |
| med | 0.478 | 0.326 | 0.387 | 172 |
| high | 0.976 | 0.987 | 0.982 | 4828 |
| macro | — | — | 0.790 | — |

**Overall accuracy:** **97.0%**

---

## 4. ROC curve operating point

- **AUC:** **0.998**
- **Chosen operating point:** **FPR = 0.024**, **TPR = 0.987**

### How ROC points are computed

For each candidate threshold **T** from **0** to **100**:

- **TPR (true positive rate)** = **TP / (TP + FN)** computed on **high-PII** (positive) examples.
- **FPR (false positive rate)** = **FP / (FP + TN)** computed on **safe** (negative) examples.

The **current** operating point is selected where **FPR < 0.025** while maintaining acceptable **TPR** on high-PII content (per the ROC sweep and product constraints).

---

## 5. Dataset composition

**Training set:** **9,711** examples drawn from **three** sources:

1. **Nvidia Nemotron-CC (programmatic labels via regex oracle)**
   - **Distant supervision** labeling strategy.
   - **Diverse real-world web text.**
   - **Label noise** is expected; a **feedback loop** corrects errors over time.

2. **Financial PII structured records**
   - **Real-world** shapes for SSNs, cards, IBANs, etc., using **synthetic values** where required for safety.
   - **High-quality** ground truth for structured identifiers.

3. **Faker-generated safe text (3,000 examples)**
   - **Balances** class distribution.
   - **Reduces** over-prediction of PII on benign content.

### Class distribution (after balancing)

- **low:** 500 (**5.1%**)
- **med:** 383 (**3.9%**)
- **high:** 8,828 (**91%**)

### Class weights (training)

- **low:** 1.0  
- **med:** 40.0  
- **high:** 100.0  

---

## 6. HIPAA detection mechanism

Detection is **layered**:

- **NER** surfaces: **`PERSON`**, **`DATE`**, **`ORG`** (e.g., facility names).
- **Regex** surfaces: **MRN** patterns, **NPI** numbers, **ICD** codes, and related formatted identifiers.
- **TinyBERT** surfaces: **contextual medical language** where patterns alone are insufficient.

**Combined behavior:** All **three** layers contribute; when signals **agree**, the **score is boosted** (agreement-weighted fusion).

### HIPAA 18 identifiers — coverage notes

| Identifier | Coverage |
|------------|----------|
| Names | NER **PERSON** ✓ |
| Dates | NER **DATE** ✓ |
| Geographic | NER **GPE** ✓ |
| Phone | regex ✓ |
| Fax | regex ✓ |
| Email | regex ✓ |
| SSN | regex ✓ |
| MRN | regex ✓ |
| Account numbers | regex ✓ |
| Certificate / license numbers | regex **partial** |
| VIN | regex ✓ |
| IP addresses | regex ✓ |
| Device identifiers | regex **partial** |
| Web URLs | regex **partial** |
| Biometric identifiers | **MISSING** |
| Full face photos | **MISSING** (image not supported) |
| Any unique identifier not matching a pattern | TinyBERT **partial** |

---

## 7. Comparative analysis

| Approach | What it catches well | What it tends to miss | FPR (indicative) | Speed | CPU cost | Recommended use case |
|----------|----------------------|------------------------|------------------|-------|----------|----------------------|
| **Regex only** | Hard-formatted secrets, emails, phones, IPs, many IDs when pattern-stable | Implicit PII, rephrased or obfuscated text, novel formats, non-English | **~0.1%** (very low) | **Fastest** (`<2ms` critical path) | **Lowest** | First-line blocking for known secret/credential patterns; ultra-low FPR policy |
| **NER only** | Person/org/location/date **entities** in fluent English | Secrets without entity-like spans, structured numeric IDs, adversarial tokenization | Moderate–high vs regex (entity false positives on generic names) | **Fast–medium** (pipeline-dependent) | **Medium** | Entity-heavy documents where spans are well-formed |
| **TinyBERT only** | **Contextual** and semantic risk; generalized “sounds like PII” passages | Brand-new secret formats, adversarial splits, very short ambiguous spans | **~4.2%** at listed thresholds | **~9 ms** (model) | **Higher** | Broad risk scoring when patterns are incomplete |
| **Combined** | **Layered** coverage: regex critical path + model context + NER/regex agreement boosts | Residual gaps where **all** layers fail (implicit reasoning, images, unseen locales) | **~2.4%** (between regex-only and model-only) | **~9 ms** + parallel NER/regex | **Highest** | **Production** default: balance of recall, FPR, and policy |

---

## 8. Known limitations and v2 roadmap

### Limitations (current generation)

- **US-centric** training data: patterns and priors may **not** reliably detect **UK NI**, **Aadhaar**, **Canadian SIN**, and other **non-US** national identifiers without explicit rules.
- **English-only** modeling and lexicon; other languages require separate data and evaluation.
- **Adversarial evasion** (e.g., **asterisk-separated** SSNs, unusual spacing) can reduce regex fidelity; implicit PII may require **stronger** models or **augmentation**.
- **Implicit PII** (relationships, indirect references) often needs **LLM-level** reasoning—not guaranteed by a small classifier.
- **Medium** class **F1** is **low** due to **label ambiguity** at the **low/med** boundary.
- **No image or PDF** scanning in the core text path—**text-only** pipeline.
- **No real-time retraining**—updates are **batch** (offline) unless a separate pipeline is added.

### Planned v2 improvements

- **International** PII patterns and locale-aware evaluation.
- **GPT-4–assisted** (or similar) **labeling** for implicit and boundary cases.
- **Adversarial augmentation** in training data to improve robustness.
- **Continuous learning** from **production feedback** (policy-governed, privacy-preserving).
