# Core Sentinel Capstone Defense Package

## 1) Executive summary
Core Sentinel is a Windows-first clipboard privacy guardrail for AI and LLM prompts. It intercepts paste events, scans clipboard text for PII and secrets, scores the risk with a hybrid rules-plus-ML pipeline, and then decides whether to allow, warn, or block the paste. The project is designed as a safety system, not just a classifier.

At a high level, the system combines:
- deterministic regex and knowledge-base rules,
- span-level PII mapping,
- TinyBERT-based text classification,
- policy thresholds and strict decision rules,
- remediation UI for masking or editing,
- feedback capture, retraining, calibration, and release gating.

Primary goal: prevent sensitive data from entering LLM chat windows while keeping the user experience fast and practical.

---

## 2) One-sentence project pitch
Core Sentinel is a Windows clipboard privacy shield that detects sensitive content before paste, assigns a risk level, and blocks or warns on unsafe AI prompt input.

---

## 3) Why this project exists
AI assistants are useful, but users often paste confidential data into prompts without realizing it. That can expose:
- personal identifiers,
- account and payment data,
- passwords and API keys,
- health or HR data,
- internal business details.

Core Sentinel exists to reduce that risk at the point of paste, when the user can still intervene.

---

## 4) End-to-end system flow

1. User copies text and pastes into an AI or chat window.
2. The Windows hook and clipboard listener capture the event.
3. The system checks whether the foreground app looks like an LLM/chat context.
4. Text is analyzed with layered detection:
   - regex and KB rules,
   - PII span detection / mapping,
   - TinyBERT scoring,
   - optional Gemini augmentation for document scanning workflows.
5. Risk is aggregated into a decision: silent, warn, or block.
6. A UI bubble and remediation dialog explain the issue.
7. The event is logged for telemetry.
8. User corrections can be fed back into retraining.
9. Calibration and release gates validate the system before promotion.

---

## 5) Architecture explanation

### A. Input and interception layer
Relevant files:
- [core-sentinel-guardrail/win_paste_hook.py](core-sentinel-guardrail/win_paste_hook.py)
- [core-sentinel-guardrail/clipboard_listener.py](core-sentinel-guardrail/clipboard_listener.py)
- [core-sentinel-guardrail/clipboard_monitor.py](core-sentinel-guardrail/clipboard_monitor.py)
- [core-sentinel-guardrail/windows_clipboard_app.py](core-sentinel-guardrail/windows_clipboard_app.py)
- [core-sentinel-guardrail/active_window_llm.py](core-sentinel-guardrail/active_window_llm.py)

Role:
- detect paste attempts,
- inspect foreground window context,
- obtain clipboard content,
- trigger analysis only when needed.

### B. Scoring and policy layer
Relevant files:
- [core-sentinel-guardrail/infer.py](core-sentinel-guardrail/infer.py)
- [core-sentinel-guardrail/risk_mapping.py](core-sentinel-guardrail/risk_mapping.py)
- [core-sentinel-guardrail/risk_policy_loader.py](core-sentinel-guardrail/risk_policy_loader.py)
- [core-sentinel-guardrail/hipaa_identifiers.py](core-sentinel-guardrail/hipaa_identifiers.py)
- [core-sentinel-guardrail/kb_loader.py](core-sentinel-guardrail/kb_loader.py)
- [core-sentinel-guardrail/gemini_scanner.py](core-sentinel-guardrail/gemini_scanner.py)

Role:
- detect sensitive spans,
- map labels to risk,
- merge policy thresholds,
- choose the final action.

### C. UI and user remediation layer
Relevant files:
- [core-sentinel-guardrail/ui_risk_bubble.py](core-sentinel-guardrail/ui_risk_bubble.py)
- [core-sentinel-guardrail/ui_remediation_dialog.py](core-sentinel-guardrail/ui_remediation_dialog.py)
- [core-sentinel-guardrail/ui_bubble_toolbar.py](core-sentinel-guardrail/ui_bubble_toolbar.py)
- [core-sentinel-guardrail/ui_sentinel_settings.py](core-sentinel-guardrail/ui_sentinel_settings.py)
- [core-sentinel-guardrail/toast.py](core-sentinel-guardrail/toast.py)
- [core-sentinel-guardrail/traffic_light_indicator.py](core-sentinel-guardrail/traffic_light_indicator.py)

Role:
- show risk feedback,
- let users mask or edit content,
- keep alerts lightweight and understandable.

### D. Data, training, and evaluation layer
Relevant files:
- [core-sentinel-guardrail/data.py](core-sentinel-guardrail/data.py)
- [core-sentinel-guardrail/train.py](core-sentinel-guardrail/train.py)
- [core-sentinel-guardrail/evaluate_test.py](core-sentinel-guardrail/evaluate_test.py)
- [core-sentinel-guardrail/calibrate_thresholds.py](core-sentinel-guardrail/calibrate_thresholds.py)
- [core-sentinel-guardrail/run_error_analysis.py](core-sentinel-guardrail/run_error_analysis.py)
- [core-sentinel-guardrail/build_unseen_blind_eval_pack.py](core-sentinel-guardrail/build_unseen_blind_eval_pack.py)
- [core-sentinel-guardrail/release_readiness_gate.py](core-sentinel-guardrail/release_readiness_gate.py)

Role:
- build datasets,
- train TinyBERT,
- evaluate on validation/test/blind sets,
- calibrate thresholds,
- block unsafe releases.

### E. Feedback, telemetry, and publishing layer
Relevant files:
- [core-sentinel-guardrail/feedback_store.py](core-sentinel-guardrail/feedback_store.py)
- [core-sentinel-guardrail/merge_feedback_to_training.py](core-sentinel-guardrail/merge_feedback_to_training.py)
- [core-sentinel-guardrail/retrain_publish.py](core-sentinel-guardrail/retrain_publish.py)
- [core-sentinel-guardrail/publish_website_records.py](core-sentinel-guardrail/publish_website_records.py)
- [core-sentinel-guardrail/guardrail_logs.py](core-sentinel-guardrail/guardrail_logs.py)
- [core-sentinel-guardrail/sentinel_sync_daemon.py](core-sentinel-guardrail/sentinel_sync_daemon.py)
- [core-sentinel-guardrail/sync_supabase_guardrail.py](core-sentinel-guardrail/sync_supabase_guardrail.py)

Role:
- capture user feedback,
- log decisions,
- sync metrics,
- support retraining and dashboard publishing.

---

## 6) File-by-file walkthrough

### Core runtime
- [core-sentinel-guardrail/main.py](core-sentinel-guardrail/main.py)
  - Main application entry point.
  - Bootstraps environment, UI, hooks, inference worker, and retraining notification.
- [core-sentinel-guardrail/guardrail_runtime.py](core-sentinel-guardrail/guardrail_runtime.py)
  - Runtime helpers and integration glue.
- [core-sentinel-guardrail/user_settings.py](core-sentinel-guardrail/user_settings.py)
  - Stores user preferences and runtime settings.
- [core-sentinel-guardrail/access_control.py](core-sentinel-guardrail/access_control.py)
  - Controls where and when guardrail logic is applied.
- [core-sentinel-guardrail/input_field_rect.py](core-sentinel-guardrail/input_field_rect.py)
  - Supports locating text input geometry.
- [core-sentinel-guardrail/win_input_anchor.py](core-sentinel-guardrail/win_input_anchor.py)
  - Helps anchor UI elements to Windows input areas.
- [core-sentinel-guardrail/font_clamp.py](core-sentinel-guardrail/font_clamp.py)
  - Maintains readable UI sizing.

### Inference and rules
- [core-sentinel-guardrail/infer.py](core-sentinel-guardrail/infer.py)
  - Core scoring engine.
  - Performs sliding-window inference, rule fusion, risk scoring, and final action selection.
- [core-sentinel-guardrail/risk_mapping.py](core-sentinel-guardrail/risk_mapping.py)
  - Maps labels to risk levels and defines regex-based override triggers.
- [core-sentinel-guardrail/risk_policy_loader.py](core-sentinel-guardrail/risk_policy_loader.py)
  - Loads and merges policy thresholds.
- [core-sentinel-guardrail/kb_loader.py](core-sentinel-guardrail/kb_loader.py)
  - Loads custom phrase/rule knowledge-base entries.
- [core-sentinel-guardrail/hipaa_identifiers.py](core-sentinel-guardrail/hipaa_identifiers.py)
  - HIPAA-related identifier support.
- [core-sentinel-guardrail/pii_remediation.py](core-sentinel-guardrail/pii_remediation.py)
  - Remediation and masking helpers.

### ML and data
- [core-sentinel-guardrail/data.py](core-sentinel-guardrail/data.py)
  - Builds Arrow datasets, balances classes, and prepares training splits.
- [core-sentinel-guardrail/train.py](core-sentinel-guardrail/train.py)
  - Fine-tunes TinyBERT and saves the model.
- [core-sentinel-guardrail/evaluate_test.py](core-sentinel-guardrail/evaluate_test.py)
  - Evaluates the test split and writes reports.
- [core-sentinel-guardrail/calibrate_thresholds.py](core-sentinel-guardrail/calibrate_thresholds.py)
  - Calibrates thresholds for binary and 3-class decisions.
- [core-sentinel-guardrail/run_error_analysis.py](core-sentinel-guardrail/run_error_analysis.py)
  - Produces confusion matrix and sampled errors.
- [core-sentinel-guardrail/build_unseen_blind_eval_pack.py](core-sentinel-guardrail/build_unseen_blind_eval_pack.py)
  - Creates unseen blind-eval and safe-only sets with leakage filtering.
- [core-sentinel-guardrail/release_readiness_gate.py](core-sentinel-guardrail/release_readiness_gate.py)
  - Final safety gate for rollout decisions.
- [core-sentinel-guardrail/binary_threshold_check.py](core-sentinel-guardrail/binary_threshold_check.py)
  - Validates binary decision thresholds.
- [core-sentinel-guardrail/binary_regression_tests.py](core-sentinel-guardrail/binary_regression_tests.py)
  - Regression coverage for binary behavior.
- [core-sentinel-guardrail/adversarial_tests.py](core-sentinel-guardrail/adversarial_tests.py)
  - Stress tests against tricky or adversarial prompts.
- [core-sentinel-guardrail/performance_audit.py](core-sentinel-guardrail/performance_audit.py)
  - Tracks efficiency and runtime behavior.

### UI
- [core-sentinel-guardrail/ui_risk_bubble.py](core-sentinel-guardrail/ui_risk_bubble.py)
  - Floating risk indicator bubble.
- [core-sentinel-guardrail/ui_remediation_dialog.py](core-sentinel-guardrail/ui_remediation_dialog.py)
  - Explains detected issues and supports masking/editing.
- [core-sentinel-guardrail/ui_bubble_toolbar.py](core-sentinel-guardrail/ui_bubble_toolbar.py)
  - Toolbar actions for the bubble UI.
- [core-sentinel-guardrail/ui_sentinel_settings.py](core-sentinel-guardrail/ui_sentinel_settings.py)
  - User settings dialog.
- [core-sentinel-guardrail/toast.py](core-sentinel-guardrail/toast.py)
  - Lightweight notifications.
- [core-sentinel-guardrail/traffic_light_indicator.py](core-sentinel-guardrail/traffic_light_indicator.py)
  - Traffic-light style status indicator.
- [core-sentinel-guardrail/character_widget.py](core-sentinel-guardrail/character_widget.py)
  - UI support widget.
- [core-sentinel-guardrail/ui_bubble_toolbar.py](core-sentinel-guardrail/ui_bubble_toolbar.py)
  - Action buttons and tooltip behavior.

### Windows integration
- [core-sentinel-guardrail/win_paste_hook.py](core-sentinel-guardrail/win_paste_hook.py)
  - Low-level paste/keyboard hook.
- [core-sentinel-guardrail/clipboard_listener.py](core-sentinel-guardrail/clipboard_listener.py)
  - Clipboard change listener.
- [core-sentinel-guardrail/clipboard_monitor.py](core-sentinel-guardrail/clipboard_monitor.py)
  - Clipboard monitoring loop.
- [core-sentinel-guardrail/windows_clipboard_app.py](core-sentinel-guardrail/windows_clipboard_app.py)
  - Windows clipboard application integration.
- [core-sentinel-guardrail/active_window_llm.py](core-sentinel-guardrail/active_window_llm.py)
  - Detects whether the active app is likely an LLM/chat surface.
- [core-sentinel-guardrail/win_input_anchor.py](core-sentinel-guardrail/win_input_anchor.py)
  - Aligns UI to the target input area.

### Telemetry, feedback, and publishing
- [core-sentinel-guardrail/feedback_store.py](core-sentinel-guardrail/feedback_store.py)
  - Stores user feedback and corrections.
- [core-sentinel-guardrail/merge_feedback_to_training.py](core-sentinel-guardrail/merge_feedback_to_training.py)
  - Merges feedback into training data.
- [core-sentinel-guardrail/retrain_publish.py](core-sentinel-guardrail/retrain_publish.py)
  - Retrains and publishes model artifacts.
- [core-sentinel-guardrail/publish_website_records.py](core-sentinel-guardrail/publish_website_records.py)
  - Publishes dashboard/website records.
- [core-sentinel-guardrail/sentinel_sync_daemon.py](core-sentinel-guardrail/sentinel_sync_daemon.py)
  - Background sync process.
- [core-sentinel-guardrail/sync_supabase_guardrail.py](core-sentinel-guardrail/sync_supabase_guardrail.py)
  - Syncs guardrail telemetry to Supabase.
- [core-sentinel-guardrail/guardrail_logs.py](core-sentinel-guardrail/guardrail_logs.py)
  - Structured logging helpers.

### Optional external scanner
- [core-sentinel-guardrail/gemini_scanner.py](core-sentinel-guardrail/gemini_scanner.py)
  - Optional Gemini-based PII scanner used when an API key is available.

### Reports and dashboards
- [core-sentinel-guardrail/README.md](core-sentinel-guardrail/README.md)
  - Subproject overview and usage.
- [README.md](README.md)
  - Repository-level overview.
- [REPO_LAYOUT.md](REPO_LAYOUT.md)
  - Workspace structure map.
- [docs/ML_SYSTEM_REFERENCE.md](docs/ML_SYSTEM_REFERENCE.md)
  - Formal ML and threshold reference.
- [docs/THREE_LAYER_PIPELINE.md](docs/THREE_LAYER_PIPELINE.md)
  - Explains the layered detection design.
- [docs/PRODUCTION_AND_RELEASE.md](docs/PRODUCTION_AND_RELEASE.md)
  - Production and release workflow.
- [docs/WINDOWS_SETUP.md](docs/WINDOWS_SETUP.md)
  - Windows setup notes.
- [docs/index.html](docs/index.html)
  - Public landing page.
- [docs/hub.html](docs/hub.html)
  - Documentation hub.
- [docs/ml/index.html](docs/ml/index.html)
  - ML dashboard page.
- [docs/demo/index.html](docs/demo/index.html)
  - Demo page.
- [docs/admin/index.html](docs/admin/index.html)
  - Admin dashboard.

---

## 7) Machine learning pipeline explanation

### 7.1 Dataset construction
The dataset pipeline builds labeled training, validation, and test splits from multiple sources. It includes synthetic and real-world material, then balances the classes so the model sees a stable distribution during training.

Typical sources discovered in the repository include:
- Nemotron PII,
- Enron-like email data,
- business-real data,
- Kaggle sensitive data,
- Patronus / enterprise PII,
- BigCode PII,
- AI4Privacy-style categories,
- user feedback rows,
- curated extra phrase pools.

Key design choice: the system does not depend on a single dataset. It uses mixed sources to improve generalization.

### 7.2 Model training
The model is TinyBERT-based sequence classification. The training pipeline:
- tokenizes the text,
- applies weighted loss or class balancing,
- fine-tunes on labeled spans/text,
- evaluates on validation data,
- stores the trained artifact under the models directory.

### 7.3 Inference
The inference path uses sliding windows for longer text so the model is not limited to a short token span.

For a text $x$ with windows $w_1, w_2, \dots, w_n$:

$$
p(y \mid x) = \text{aggregate}\big(p(y \mid w_1), p(y \mid w_2), \dots, p(y \mid w_n)\big)
$$

This matters because clipboard content can be longer than a single model context window.

### 7.4 Risk scoring
The model output is not used alone. It is fused with:
- regex matches,
- KB triggers,
- critical secret detection,
- policy thresholds,
- long-text penalties,
- special overrides.

A practical interpretation is:

$$
\text{risk score} = f(\text{model probabilities}, \text{regex signals}, \text{secret rules}, \text{policy thresholds})
$$

The system then maps the final risk to an action:
- silent,
- warn,
- block.

### 7.5 Threshold calibration
Calibration tunes the decision boundary to prioritize safety.

For binary risky detection, the objective is essentially:
- maximize risky recall,
- keep false positives within an acceptable cap.

This is appropriate for a safety guardrail, because missing sensitive data is worse than warning too often.

---

## 8) Mathematical explanation for viva

### A. Classification probabilities
The classifier outputs logits $z_i$ for each class. Softmax converts these to probabilities:

$$
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

### B. Binary decision rule
If the risky probability exceeds threshold $\tau$, the system blocks or warns:

$$
\text{action} =
\begin{cases}
\text{block or warn}, & p_{risk} \ge \tau \\
\text{silent}, & p_{risk} < \tau
\end{cases}
$$

### C. Recall and false positive rate
For risky content, recall is:

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

False positive rate is:

$$
\text{FPR} = \frac{FP}{FP + TN}
$$

For a guardrail, high recall is crucial because false negatives are more dangerous than extra warnings.

### D. Macro F1
Macro F1 treats each class equally:

$$
F1_{macro} = \frac{1}{K}\sum_{k=1}^{K} \frac{2PR}{P+R}
$$

This is useful when classes are imbalanced.

---

## 9) Evaluation and results summary
The repository artifacts indicate a mature evaluation pipeline.

Observed report values include:
- dataset size around 13.2k samples,
- train/validation/test split around 10,455 / 485 / 2,289,
- three balanced classes in the main record set,
- risky AUPRC around 0.989,
- binary calibration with risky threshold around 0.1,
- blind-set release gate pass on one curated evaluation set,
- fail and canary recommendation on an unseen blind set.

Interpretation:
- The model is strong on risky detection.
- Generalization remains a real concern on unseen phrases, which is why release gating matters.

---

## 10) Best defense narrative
Use this framing:

> Core Sentinel is a layered clipboard safety system that protects users from accidentally pasting sensitive data into AI prompts. It combines deterministic PII rules, ML-based text classification, policy thresholds, and user remediation. It is trained, calibrated, evaluated, and release-gated like a production safety product, not just a demo model.

If asked why it is novel:
- it acts at the moment of paste,
- it is Windows-native,
- it blends rules and ML,
- it supports feedback-driven retraining,
- it includes operational release gates.

---

## 11) 40-minute presentation structure

### Slide 1: Title
- Project name
- Your name
- Tagline: AI clipboard privacy guardrail

### Slide 2: Problem statement
- People paste sensitive data into AI tools
- Leakage risk is real
- Need point-of-paste protection

### Slide 3: Proposed solution
- Layered guardrail
- warn/block before paste
- remediation support

### Slide 4: System architecture
- Windows hook
- clipboard monitor
- inference engine
- UI feedback
- telemetry loop

### Slide 5: Detection pipeline
- regex
- KB rules
- model inference
- policy logic

### Slide 6: Dataset and training
- sources
- balancing
- TinyBERT fine-tuning

### Slide 7: Threshold calibration
- why thresholds matter
- safety-first tuning

### Slide 8: Evaluation
- metrics
- confusion matrix
- blind set results

### Slide 9: Release readiness
- gate checks
- canary rollout logic

### Slide 10: Demo flow
- copy text
- paste into LLM
- detection bubble appears
- user remediates

### Slide 11: Limitations
- unseen phrasing
- domain shift
- false positives

### Slide 12: Future work
- broader models
- richer policy learning
- enterprise integrations

### Slide 13: Conclusion
- privacy-by-design
- safety at the point of paste

---

## 12) Likely panel questions and strong answers

### Q1. Why not use only regex?
A. Regex is fast and precise for known patterns, but it cannot generalize to novel phrasing or context. The model adds coverage beyond fixed patterns.

### Q2. Why not use only ML?
A. ML alone can miss critical secrets or produce unstable results. Deterministic rules provide high-confidence protection for known dangerous patterns.

### Q3. Why TinyBERT?
A. It is lightweight enough for interactive desktop inference while still providing contextual classification power.

### Q4. Why do you prioritize recall over accuracy?
A. A guardrail should avoid missing risky content. False negatives are more harmful than extra warnings.

### Q5. What happens on long text?
A. The system uses sliding windows so content beyond the model’s native context length is still examined.

### Q6. How do you handle false positives?
A. Through threshold calibration, rule tuning, policy overrides, and remediation UI that lets the user correct content.

### Q7. How do you know the model is ready to ship?
A. It must pass evaluation and release gates on blind or unseen sets, not just training or validation metrics.

### Q8. What is the role of Gemini?
A. It is an optional enhancement for document-scanning workflows, not the core runtime dependency.

### Q9. What if the text contains both safe and risky content?
A. The policy layer aggregates the strongest signals and prioritizes safety.

### Q10. What is the main limitation?
A. Generalization on unseen prompts and domain shift. The unseen blind-set failure demonstrates why calibration and gating are necessary.

---

## 13) Limitations
Be honest and explicit:
- Unseen phrasing can evade model confidence.
- Some docs and artifacts reflect different label modes, so interpretation must be careful.
- Windows-native behavior limits portability.
- External model or API augmentation may depend on network/API availability.
- Safety tuning can create false positives.

These are acceptable limitations for a real defense if you show that they are recognized and mitigated.

---

## 14) Future scope
Good future directions:
- stronger generalization with larger or domain-adapted models,
- better active learning from feedback,
- more robust blind-set generation,
- enterprise policy integration,
- multi-platform support,
- richer explanation and audit logs,
- improved document-level scanning.

---

## 15) Demo script

1. Open a chat or text area.
2. Paste a safe sentence: system stays silent.
3. Paste a medium-risk sentence: user gets a warning.
4. Paste a risky item like an API key or SSN: system blocks.
5. Show remediation dialog and masking.
6. Show telemetry or logs.
7. Explain how feedback improves future performance.

---

## 16) One-page cheat sheet

### What to say first
- Core Sentinel prevents accidental leakage of sensitive data into AI prompts.

### Core pipeline
- Hook → Analyze → Score → Decide → Remediate → Log → Learn.

### Main technical strengths
- Windows-native
- Hybrid rules + ML
- Threshold calibrated
- Feedback-aware
- Release-gated

### Main metrics to mention
- about 13.2k samples
- risky AUPRC around 0.989
- binary risky threshold around 0.1 in calibration
- blind-set pass on curated data
- canary recommendation on unseen blind data

### Most important files
- [core-sentinel-guardrail/main.py](core-sentinel-guardrail/main.py)
- [core-sentinel-guardrail/infer.py](core-sentinel-guardrail/infer.py)
- [core-sentinel-guardrail/data.py](core-sentinel-guardrail/data.py)
- [core-sentinel-guardrail/train.py](core-sentinel-guardrail/train.py)
- [core-sentinel-guardrail/evaluate_test.py](core-sentinel-guardrail/evaluate_test.py)
- [core-sentinel-guardrail/calibrate_thresholds.py](core-sentinel-guardrail/calibrate_thresholds.py)
- [core-sentinel-guardrail/release_readiness_gate.py](core-sentinel-guardrail/release_readiness_gate.py)
- [docs/ML_SYSTEM_REFERENCE.md](docs/ML_SYSTEM_REFERENCE.md)

### Best closing line
- This project demonstrates a production-style safety architecture for AI input privacy, with measurable evaluation, operational gating, and a user-centered remediation loop.

---

## 17) Final defense statement
Core Sentinel is a practical, production-oriented privacy guardrail for AI usage on Windows. Its value is not only in detecting PII, but in preventing accidental disclosure at the moment of paste, while supporting remediation, learning, and safe release practices.
