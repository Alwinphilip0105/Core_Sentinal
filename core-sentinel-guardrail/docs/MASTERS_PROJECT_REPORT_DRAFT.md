# Core Sentinel: A Hybrid Guardrail for Preventing Sensitive Data Leakage in AI Prompt Workflows

## Abstract
The rapid adoption of large language models (LLMs) has introduced a new privacy risk: users frequently paste confidential information into AI chat interfaces. This project presents Core Sentinel, a Windows-native clipboard guardrail that intercepts paste actions, analyzes clipboard text using a hybrid detection pipeline, and enforces policy outcomes (`silent`, `warn`, or `block`) before sensitive data is submitted to external AI systems. Core Sentinel combines deterministic regex-based secret detection, NER and medium-risk pattern signals, and TinyBERT-based contextual classification in a policy-fused runtime. The system is calibrated for safety-first behavior using precision-recall threshold selection and release-gated evaluation on blind and unseen datasets. Experimental results indicate strong high-risk detection (high-class recall ~0.987, AUC ~0.998), while highlighting ongoing challenges in medium-risk boundary classification. The project demonstrates a practical privacy-by-design architecture for AI usage, balancing detection quality, low-latency user experience, and operational governance.

## 1. Introduction
### 1.1 Background
LLM tools are increasingly integrated into daily workflows for coding, writing, analysis, and communication. In real use, users often include sensitive content in prompts unintentionally, such as API keys, patient identifiers, payroll details, account data, and internal business information. Once pasted into a remote AI service, the data exposure risk may become irreversible.

### 1.2 Problem Statement
Existing privacy controls are commonly post-hoc (e.g., audits, DLP logs) or rely on user awareness. There is a need for a proactive, low-latency guardrail that operates at the exact moment of paste and prevents accidental disclosure before transmission.

### 1.3 Project Objectives
This project aims to:
- Detect structured and contextual sensitive data in clipboard text.
- Enforce configurable policy outcomes (`silent`, `warn`, `block`) in real time.
- Maintain interactive desktop performance suitable for everyday use.
- Support calibration, retraining, and release governance for continual improvement.

### 1.4 Contributions
Core contributions of this work include:
- A three-layer hybrid detection pipeline (critical regex, TinyBERT classifier, NER+medium regex fusion).
- A Windows-native interception and remediation workflow.
- Safety-first threshold calibration and release-readiness gate integration.
- A feedback-aware operational loop for future model and policy updates.

## 2. System Overview
Core Sentinel is implemented as a clipboard-first privacy guardrail for Windows environments. The runtime flow is:
1. Intercept paste event and collect current clipboard text.
2. Confirm target context is likely an AI/chat prompt surface.
3. Run layered risk detection and aggregate scores.
4. Apply policy thresholds and deterministic overrides.
5. Execute action: allow silently, warn user, or block paste.
6. Offer remediation options (mask/edit).
7. Log event metadata for analysis and retraining workflows.

This design reframes privacy protection as an inline decision system rather than an offline classifier.

## 3. Architecture and Detection Pipeline
### 3.1 Layer 1: Critical Secret Regex (Fast Path)
The first stage uses deterministic regex for high-confidence secret patterns (e.g., private keys, API credentials, structured critical identifiers). If matched, the system can block immediately, minimizing both latency and false negatives for known high-severity patterns.

### 3.2 Layer 2: TinyBERT 3-Class Classification
The second stage applies a fine-tuned TinyBERT model (`low`, `med`, `high` risk classes). Clipboard text is chunked into token windows, scored per chunk, and aggregated with a worst-window preference so localized high-risk evidence is not diluted in long text.

### 3.3 Layer 3: NER + Medium Regex Fusion
In parallel with model inference, NER and medium-risk regex patterns detect entity-level and formatted indicators (e.g., person/date/org/location, phone, email, IP, address-like patterns). Agreement between NER and regex signals increases confidence through score boosting.

### 3.4 Policy Fusion and Final Action
Final decisions are not model-only. Runtime policy combines:
- model probabilities,
- deterministic triggers,
- regex/NER signals,
- risk score gates,
- strict override rules.

This policy-fused approach improves safety over single-detector systems by preserving deterministic precision and contextual generalization.

## 4. Data Pipeline and Labeling Strategy
### 4.1 Dataset Composition
Training data is built from mixed sources including weakly labeled web-scale text (e.g., Nemotron-style span-derived supervision), structured financial PII records, and synthetic safe examples to control false positives.

Reported working scale in current artifacts:
- Training examples: approximately 9.7k (in one tracked setup).
- Alternative merged artifacts: up to ~13k across pipeline variants.

### 4.2 Labeling Logic
Labeling combines source-provided metadata and rule-based mapping into project risk schema:
- Binary mode: `safe`, `risky`.
- Multi-class mode: `low`, `med`, `high`.

Weak supervision from span counts and critical-type mappings is used where manual labels are unavailable, enabling scalable data generation with known noise characteristics.

### 4.3 Class Imbalance Handling
To reduce majority-class collapse, the pipeline uses:
- rebalancing/top-up strategies,
- class weights in loss,
- stratified splits where feasible.

This is essential for safety tasks where severe examples are less frequent but operationally critical.

## 5. Model Training and Inference
### 5.1 Model
Base encoder: TinyBERT (`huawei-noah/TinyBERT_General_4L_312D`) fine-tuned for sequence classification.

### 5.2 Training
The training process includes tokenization, weighted cross-entropy optimization, validation-driven checkpointing, and output artifact publishing to runtime model directories.

### 5.3 Inference on Long Text
Given clipboard input \(x\), with windows \(w_1, w_2, ..., w_n\), the model computes class probabilities per window and merges evidence:

\[
p(y|x) = \text{aggregate}(p(y|w_1), p(y|w_2), ..., p(y|w_n))
\]

Worst-window aggregation is used to preserve high-risk local spans in long prompt text.

## 6. Threshold Calibration and Policy Selection
### 6.1 Calibration Method
Thresholds are tuned through precision-recall and ROC sweeps on held-out validation/test splits. A recall-first objective under false-positive constraints is used for safety deployment.

### 6.2 Current Calibrated Points (Observed)
From current project references:
- Warn threshold: ~40
- Block threshold: ~70
- Binary risky probability threshold: ~0.10 (for one calibrated binary route)

### 6.3 Rationale
The guardrail prioritizes minimizing false negatives on high-risk content while keeping safe-class false positives within acceptable operational limits. This is appropriate for pre-send privacy enforcement.

## 7. Evaluation Results
### 7.1 Multi-Class Test Metrics (Observed Artifacts)
- Overall accuracy: ~97.0%
- High class: precision ~0.976, recall ~0.987, F1 ~0.982
- Medium class: lower F1 (~0.387), indicating boundary ambiguity and domain-shift sensitivity
- AUC: ~0.998 at selected operating context

### 7.2 Comparative Behavior
- Regex-only: fastest, lowest FPR on known formats, weaker contextual recall.
- ML-only: stronger contextual recall, higher standalone false positives.
- Combined policy: best practical balance for production safety.

### 7.3 Release Readiness Insight
Curated blind sets can pass while unseen blind sets expose medium-class generalization gaps, reinforcing the need for strict gating and canary rollout logic before broad release.

## 8. HIPAA and Sensitive Domain Coverage
The system includes HIPAA-relevant detection through:
- entity signals (e.g., names, dates, locations, organizations),
- regex for key identifiers (e.g., MRN/NPI/ICD-related formats depending on configuration),
- contextual model support for non-template medical narratives.

Coverage is substantial for text-based identifiers but incomplete for multimodal identifiers (e.g., images/biometrics), which are outside the current runtime scope.

## 9. Implementation Layers
Core Sentinel is organized into functional layers:
- Input/Interception: Windows hook, clipboard listeners, active window detection.
- Scoring/Policy: inference engine, risk mapping, policy loader, HIPAA/rule modules.
- UI/Remediation: risk bubble, remediation dialogs, toast and status indicators.
- Data/ML: dataset builders, training, evaluation, calibration, error analysis, release gate.
- Feedback/Operations: telemetry logging, feedback merge, retraining, sync/publishing.

This modular architecture supports maintainability and iterative upgrades.

## 10. Challenges and Limitations
Current limitations include:
- Medium-risk class ambiguity and reduced robustness on unseen phrasing.
- Locale sensitivity (primarily US/English-centric data in current form).
- Adversarial formatting that may bypass brittle regex variants.
- Text-only runtime path (no native image/PDF identifier extraction in core loop).
- Batch-style retraining rather than full online continual learning.

## 11. Future Work
Planned improvements:
- International identifier and multilingual expansions.
- Stronger augmentation for obfuscation and adversarial patterns.
- Better medium-class labeling quality via assisted annotation workflows.
- Improved blind-set synthesis and domain-shift evaluation.
- Privacy-preserving feedback loops for continuous calibration.
- Optional multimodal document-risk pipeline integration.

## 12. Conclusion
Core Sentinel demonstrates that a practical AI privacy guardrail should be built as a layered policy system rather than a standalone classifier. By combining deterministic secret detection, contextual ML inference, and policy-governed decision logic, the project provides meaningful real-time protection against accidental prompt leakage in Windows AI workflows. The current results show strong high-risk performance and production-oriented operational maturity, while also identifying clear pathways for medium-risk robustness and broader generalization in future versions.

## References
1. TinyBERT: Huawei Noah’s Ark Lab (`huawei-noah/TinyBERT_General_4L_312D`).
2. Project documentation and evaluation artifacts in `core-sentinel-guardrail/docs`.
3. Calibration and evaluation scripts in `core-sentinel-guardrail` runtime and ML pipeline modules.
