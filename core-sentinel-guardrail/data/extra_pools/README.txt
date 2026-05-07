Extra pools (optional training augmentation)
===========================================

Loaded by data.load_extra_pool_rows_for_training during build_and_save when this directory exists.

Tracked files:
  contextual_embedded_pii.jsonl  — synthetic narrative snippets (biz email, support, legal);
                                    each line JSON: {"text": "...", "risk": "low|med|high", optional "label"}.
                                    Rebuild: python tools/gen_contextual_embedded_pii.py
  financial_balanced_p1.jsonl  — synthetic earnings/budget vs wire/claims-style lines for FINANCIAL class balance.
                                    Rebuild: python tools/gen_financial_balanced_p1.py. P1 merge loads this automatically when tracked and env path is unset (see docs/P1_ROADMAP.md); opt out with GUARDRAIL_FINANCIAL_BALANCED_USE_DEFAULT_POOL=0.

Local / generated (untracked unless you add them):
  high_extra.jsonl               — fixed risk "high"
  hard_negative_low.jsonl        — fixed risk "low"
  hard_negative_med.jsonl        — fixed risk "med"
  weak_category_examples.jsonl     — per-line risk; generator: tools/gen_weak_category_examples.py

Env overrides:
  GUARDRAIL_EXTRA_POOL_DIR       — alternate directory
  GUARDRAIL_EXTRA_POOL_MAX_PER_FILE — cap per file (default 8000)
