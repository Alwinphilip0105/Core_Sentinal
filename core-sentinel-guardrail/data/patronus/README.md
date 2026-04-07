# Patronus AI / EnterprisePII data

Place Patronus AI or **EnterprisePII** files here to use them as a data source.

- **`sample_enterprise_pii.jsonl`** – minimal 6-row sample to verify the pipeline. Replace or add real data for training.

## Getting the data

- **EnterprisePII** (3k annotated enterprise excerpts, business-sensitive PII) is not on Hugging Face. You can obtain it from:
  - [llm-foundry](https://github.com/mosaicml/llm-foundry) (MosaicML / Databricks)
  - Patronus AI platform (enterprise)
- Save one or more of:
  - **JSON**: array of `{"text": "...", "sensitivity": "public"|"internal"|"confidential"|"restricted"}`
  - **JSONL**: one such object per line
  - **CSV**: columns `text` (or `content`/`excerpt`/`prompt`) and `sensitivity` (or `label`/`risk`/`category`)

## Risk mapping

- `public`, `internal` → **low**
- `confidential` → **med**
- `restricted`, `sensitive` → **high**

## Run

From `core-sentinel-guardrail`:

```bash
python data.py patronus
# or custom path:
python data.py patronus path/to/your/enterprise_pii.jsonl
```

Then train as usual: `python train.py`
