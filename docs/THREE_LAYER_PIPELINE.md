# Three-layer pipeline: independent examples, skips, and accuracy

This document explains how **Layer 1 (regex)**, **Layer 2 (NER)**, and **Layer 3 (TinyBERT)** complement each other in Core Sentinel. Demo scenarios live in `core-sentinel-guardrail/layer_demo.py`; production scoring uses `core-sentinel-guardrail/infer.py` (`score_clipboard_with_pii`), which merges model scores with **severity floors** from rules so a low `P(risky)` does not always erase rule-based risk.

---

## 1. Where each layer works *most* independently

These are **illustrative**: in production, layers are **combined** in a risk aggregator (policy + score), not voted on in isolation.

### A. Regex / pattern rules — strong alone on **structured** secrets

**Example (from `layer_demo.py` scenario 2):**

> `Employee record: SSN 123-45-6789, hired 2024.`

- **Regex:** Matches `\d{3}-\d{2}-\d{4}` SSN shape → deterministic hit.
- **NER:** May tag “SSN”, years, or noise — not required for the decision.
- **ML:** Usually also “risky,” but the **pattern** is already enough for critical handling in policy.

**When regex is the star:** Standard formats (SSN, email, phone, card-like runs, JWT-shaped strings) that you encoded in `REGEX_PATTERNS` / `risk_mapping.py`.

---

### B. NER — strong when **names / orgs / locations** matter but **no regex format**

**Example (scenario 4):**

> `Please forward this to Sarah Chen at the downtown office. Her manager David Park approved the transfer.`

- **Regex:** No SSN/email/phone/card pattern → **no match** (or empty).
- **NER:** `Sarah Chen`, `David Park` as **PERSON** → PII-relevant entities.
- **ML:** Often still flags contextual risk; in the demo, decision logic uses NER + ML together.

**When NER is the star:** Unstructured proper nouns and orgs where you don’t have a regex for “full name.”

**Caveat:** NER alone fires on **benign** names (e.g. “Guido van Rossum” in a Python tutorial). Production mitigates this with **policy**, **context keywords**, and **ML** — not “PERSON = always block.”

---

### C. TinyBERT — strong when **context** is sensitive but **no pattern and weak NER**

**Example (scenario 9 — adversarial / obfuscated):**

> `my social is one two three dash four five dash six seven eight nine and my cc is four five three two 1234 5678 9012`

- **Regex:** **Skips** — digits spelled out, no `123-45-6789` style string.
- **NER:** Often **unhelpful** — scattered CARDINAL tokens, not “SSN” semantics.
- **ML:** Trained on paraphrases / risk → can still label **risky** from intent (“my social is…”, “my cc is…”).

**When ML is the star:** Paraphrased PII, implicit HR/finance/health narrative, non-standard formatting.

**Caveat:** ML can miss borderline text or over-flag safe business prose — tune **threshold**, **data**, and **rules**.

---

## 2. What each layer *skips* (typical failure modes)

| Layer | Often **misses** | Why |
|--------|------------------|-----|
| **Regex** | Spelled-out numbers, unicode tricks, novel templates, “salary discussion” with no digits | Only what patterns encode |
| **NER** | Non-entity risky semantics; secrets without entity types; **false sense of safety** when no entities | Labels spans, not “is this a leak?” |
| **ML** | Rare long-tail patterns; may FP on benign if threshold low | Statistical; needs examples + calibration |

**Defense in depth:** Regex misses → NER or ML may catch; NER noisy → regex + ML + policy downgrades; ML misses → regex/critical-secret paths in `infer.py` can still **floor** severity.

---

## 3. “Who skips what?” in one glance

- **Skip regex layer** (conceptually): Chunk has **no** matching pattern — e.g. pure prose salary talk with no email/SSN shape.
- **Skip relying on NER**: No person/org/date entities, or only generic entities in safe text.
- **Skip relying on ML alone**: You still want **deterministic** blocks for known secret formats → **regex + critical-secret / hard-block** paths in production.

---

## 4. Increasing **accuracy** (system-level, not one metric)

Accuracy here means **correct allow / warn / block** with acceptable **misses (FN)** and **nuisance flags (FP)**.

1. **Per-layer tuning**  
   - Regex: add high-precision patterns; avoid overly broad patterns that FP.  
   - NER: restrict which entity types escalate risk; require **context** (e.g. medical + PERSON) for warns.  
   - ML: more **balanced** training data; threshold via `prob_threshold_risky` and calibration reports.

2. **Fusion policy** (production)  
   - Use **`max(P(risky), severity_floor)`** so rules aren’t ignored when the model is underconfident.  
   - Keep **medium → warn**, **critical → block** consistent with product goals.

3. **Evaluation**  
   - Holdout + **blind** sets; track **precision/recall/FPR/FNR** separately; recall-only is misleading without an FP budget.

4. **Operational**  
   - Mine **false negatives** into training + regression tests; mine **false positives** into safe-negative pools.

---

## 5. Try it locally

```bash
cd core-sentinel-guardrail
.\.venv\Scripts\python.exe layer_demo.py
```

Review the printed **per-layer** timings and the **SUMMARY** counts (only regex / only NER / only ML / all layers).

---

*For deployment behavior (strict mode, tiny-text, hard-block triggers), see `config/risk_policy.json` and `infer.py`.*
