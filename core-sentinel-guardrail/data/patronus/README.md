# Patronus AI / EnterprisePII data

Place Patronus AI or **EnterprisePII** files here to use them as a data source.

- **`sample_enterprise_pii.jsonl`** – minimal 6-row sample to verify the pipeline. Replace or add real data for training.

## Create the directory

**PowerShell (Windows):**

```powershell
New-Item -ItemType Directory -Force -Path data/patronus
```

**bash:**

```bash
mkdir -p data/patronus
```

## Getting the data

- **EnterprisePII** (thousands of annotated enterprise excerpts) is **not** guaranteed to exist as a public `datasets.load_dataset("PatronusAI/EnterprisePII")` on the Hub. **Confirm any dataset ID** on [huggingface.co/datasets](https://huggingface.co/datasets) before scripting downloads.
- Typical sources called out by Patronus / MosaicML:
  - [llm-foundry](https://github.com/mosaicml/llm-foundry) (MosaicML / Databricks) — follow their layout for evaluation data.
  - [Patronus AI](https://www.patronus.ai/) — product / research access (e.g. contact via their site).
- If you **do** have a valid Hugging Face dataset ID (public or after `huggingface-cli login`):

```python
from datasets import load_dataset

ds = load_dataset("YOUR_ORG/YOUR_DATASET", split="train")  # verify split name
ds.to_json("data/patronus/enterprise_pii.jsonl")
```

Use filenames ending in `.json`, `.jsonl`, or `.csv` so `data.py` picks them up.

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

## What not to do

- Do **not** add a `README.json` “placeholder” here — it is not a training file and can confuse git history. If you have nothing yet, leave the folder empty or keep only this README and `sample_enterprise_pii.jsonl` (if present).
- Avoid `pip install ... --break-system-packages` unless you know you need it (e.g. some Linux distros); prefer a **venv** and `pip install datasets`.
