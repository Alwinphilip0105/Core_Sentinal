# Growing the dataset from multi-system feedback and retraining

End-to-end loop for **3-class** (`low` / `med` / `high`) TinyBERT training when using **`multi_real_synthetic`** in `data.py`.

**Monitoring scores over time:** see **`RISK_TELEMETRY.md`** (`logs/risk_telemetry.jsonl` + local dashboard + optional webhook).

## What gets merged

1. **Bubble feedback** (`logs/feedback_store.jsonl`) — predicted vs **correct** label (low/med/high).
2. **Full text** (`logs/feedback_fulltext.jsonl`) — **complete** pasted text per feedback event (one JSON line per submit; last line per hash wins). **New since this file was added**; older feedback may only have previews.
3. **Hash index** (`logs/hash_index.jsonl`) — **fallback**: up to **500 characters** if no row exists in `feedback_fulltext.jsonl` for that hash.
4. **Curated JSONL pools** (`data/extra_pools/*.jsonl`) — optional **high** lines and **hard-negative** low/med lines you maintain by hand (see `data/extra_pools/README.txt`). Merged after feedback.

Export prefers **fulltext → preview**. Set env **`GUARDRAIL_FEEDBACK_NO_FULLTEXT=1`** when recording feedback to skip writing full text (e.g. shared machine); export then uses previews only.

## Steps

### A. Collect feedback on many machines

- Run the guardrail; use **Correct** / **Wrong** on the bubble when it appears.
- Merge logs via **git** (private repo): at minimum `feedback_store.jsonl`, **`feedback_fulltext.jsonl`**, and `hash_index.jsonl` under `logs/` (pull / merge carefully).

### B. Export → training file

From `core-sentinel-guardrail`:

```bash
python merge_feedback_to_training.py
```

This writes **`data/user_feedback/export.jsonl`** — one JSON object per line: `{"text": "...", "risk": "low"|"med"|"high"}`.

Optional: `--mark-used` marks those feedback rows as `used_for_training` so you do not double-count next time.

Override path: `GUARDRAIL_USER_FEEDBACK_JSONL` or `merge_feedback_to_training.py -o path.jsonl`.

### C. Rebuild Arrow data (includes your feedback)

```bash
# Set extra data sources as needed (see CLIPBOARD_GUARDRAIL.md)
python data.py multi_real_synthetic
```

You should see log lines like: `[user_feedback] merged N labeled rows ...` and, if pools exist, `[extra_pools] merged M curated rows ...`.

### D. Train and evaluate

```bash
python train.py
python evaluate_test.py
python calibrate_thresholds.py
```

Weights go to `models/tinybert_guardrail/` (gitignored). Copy to other machines or redeploy as you prefer.

## 9-class models

`export.jsonl` is **3-class risk**. The `ai4privacy_*` data sources in `data.py` do **not** automatically ingest this file. Use **multi_real_synthetic** for feedback-augmented 3-class runs, or add a separate mapping from risk → PII labels later.

## Security

Feedback text can contain PII. Keep the repo **private** and treat `export.jsonl` like sensitive data.
