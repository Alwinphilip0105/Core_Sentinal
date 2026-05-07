# P1 data and calibration roadmap

Kill-switch for **all** P1 loaders merged in `multi_real_synthetic`: set  
`GUARDRAIL_ENABLE_P1_DATASETS=0` (or `false` / `no`). Default is enabled; rows load only when paths / dataset ids are set.

## P1-A — HEALTH (i2b2 2014 de-identification)

- **Env:** `GUARDRAIL_I2B2_2014_DIR` → directory of challenge XML notes. Optional caps: `GUARDRAIL_I2B2_MAX_ROWS`, `GUARDRAIL_P1_MAX_CHARS`.
- **Distribution:** Obtained via [i2b2.org](https://www.i2b2.org) with their DUA — not redistributed here.
- **Loader:** `datasets_p1.load_i2b2_2014_xml_dir` unwraps `<PHI>` tags for text; `risk` is `high` when PHI markup was present, else `med`.

## P1-B — AUTH (secret-style patterns)

- **JSONL:** `GUARDRAIL_AUTH_SECRETS_JSONL` — lines with `text` and either `risk` or `is_positive` (rotated/synthetic placeholders for sharing).
- **Optional Hugging Face:** `GUARDRAIL_AUTH_HF_DATASET` (optional `:config`), `GUARDRAIL_AUTH_HF_SPLIT`, `GUARDRAIL_AUTH_HF_MAX_ROWS`. Requires `pip install datasets`.

## P1-C — FINANCIAL (balanced)

- **Env:** `GUARDRAIL_FINANCIAL_BALANCED_JSONL` (optional path) plus optional `GUARDRAIL_FIN_BALANCED_MAX_ROWS`.
- When unset, the loader uses `data/extra_pools/financial_balanced_p1.jsonl` if present. Set `GUARDRAIL_FINANCIAL_BALANCED_USE_DEFAULT_POOL=0` to skip that bundled file without setting a custom path.
- **Generator:** `python tools/gen_financial_balanced_p1.py` writes `data/extra_pools/financial_balanced_p1.jsonl`.

## P1-D — LOCATION / NAME precision gate

- **Policy key:** `gate_geo_name_ml_without_ner_regex` in `config/risk_policy.json` (default `true` in code defaults).
- **Behavior:** After contact-only scoring cap, uncorroborated ML-only NAME/LOCATION-style windows downgrade a **block** toward **warn** and cap score (see `infer.py`).

## P1-E — Live clipboard calibration

1. Collect **500–1000** consented clipboard events (export existing telemetry or add a labeling export).
2. Label each paste for intended UX action (`allow` / `warn` / `block`) or risk tier aligned with prod policy.
3. Produce PR metrics (for example via `evaluate_test.py`) suitable for `reports/pr_curve_summary.json`, then run `python calibrate_thresholds.py` to merge calibrated class thresholds into `config/risk_policy.json`. Optionally use `python calibrate_thresholds.py --validation-sweep` for Arrow-split sweeps documented in `calibrate_thresholds.py`.
