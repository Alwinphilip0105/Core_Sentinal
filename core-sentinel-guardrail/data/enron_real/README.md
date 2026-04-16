# Enron / EDRM Enron email corpus

Place Enron or EDRM-style email/body text here for real-data training or the manual-labeling candidate pool.

## Where to place files

Put files in this directory: **`core-sentinel-guardrail/data/enron_real/`** (paths below assume your shell’s current directory is **`core-sentinel-guardrail`**).

**Create the folder**

```powershell
New-Item -ItemType Directory -Force -Path data/enron_real
```

```bash
mkdir -p data/enron_real
```

## Optional: download public corpora (verify IDs on the Hub first)

These are **examples** only—dataset names and configs change. Check each [dataset card](https://huggingface.co/datasets) for license, splits, and column names. Large `*.jsonl` files under `data/` are usually **gitignored**; keep them local or in private storage.

### Option 1 — Enron slice from The Pile (streaming)

The [EleutherAI/pile](https://huggingface.co/datasets/EleutherAI/pile) dataset exposes a subset config (often `enron_emails`). Rows typically include a **`text`** field, which `data.py` can load. Cap rows so the first run does not download the whole Pile.

```python
import json
from datasets import load_dataset

ds = load_dataset("EleutherAI/pile", "enron_emails", split="train", streaming=True)
out_path = "data/enron_real/enron_emails.jsonl"
n_max = 10_000
count = 0
with open(out_path, "w", encoding="utf-8") as f:
    for row in ds:
        if count >= n_max:
            break
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        count += 1
print(f"Wrote {count} lines to {out_path}")
```

If `load_dataset` fails, open the dataset card and confirm the **config name** (e.g. `enron_emails`) and required trust / login.

### Option 2 — Kaggle “Enron email” dumps

```bash
pip install kaggle
# configure ~/.kaggle/kaggle.json with API credentials
kaggle datasets download -d wcukierski/enron-email-dataset
unzip enron-email-dataset.zip -d data/enron_real/
```

Normalize to `.csv` / `.jsonl` with a body column if needed (see below).

### Option 3 — PII-labeled Enron-style prompts (if published)

**Confirm** the dataset ID on the Hub before running (names like `LLM-PBE/enron-email` may move or require login):

```python
from datasets import load_dataset

ds = load_dataset("LLM-PBE/enron-email", split="train")
ds.to_json("data/enron_real/enron_pii_prompts.jsonl")
print(len(ds), ds.column_names)
```

**Unlabeled rows** (no `risk` / `label` / …) are kept for the **candidate pool** export but do **not** add labeled training rows until you map labels to `low`/`med`/`high` (see label mapping below).

## Supported file types

- **.csv** – CSV with a text/body column (and optional label column)
- **.json** – JSON array of objects with text and optional label
- **.jsonl** – One JSON object per line
- **.txt** – Plain text: each file is split by double newline into snippets; snippets ≥20 chars are loaded as rows. If no paragraphs, the whole file is one row if long enough.

## Cap on `enron_pii_prompts.jsonl`

`load_enron_real()` reads at most **`GUARDRAIL_ENRON_MAX_ROWS`** lines from **`enron_pii_prompts.jsonl`** (default **20000**) so very large Hub exports do not blow memory or build time. Other files in this folder are not capped. When the limit applies, the loader prints: `[enron_real] capped at N rows (GUARDRAIL_ENRON_MAX_ROWS)`.

## Expected columns (labeled data)

- **Text**: `text`, `content`, `body`, `excerpt`, `prompt`, `input`, or `message`
- **Label** (optional): `risk`, `label`, `category`, `classification`, or `sensitivity`  
  If missing, rows are treated as **unlabeled candidates** for the labeling queue.

## Accepted label values

Same mapping as Patronus: `public`/`internal` → low; `confidential` → med; `restricted`/`sensitive` → high.

## Examples

**JSONL (labeled):**
```json
{"body": "Re: Q3 numbers. Attached draft for board.", "sensitivity": "confidential"}
{"text": "Lunch at 12?", "label": "internal"}
```

**CSV (unlabeled):**
```csv
body
Meeting moved to 3pm in Bldg 2.
Please send the NDA by EOD.
```

**TXT:** One email or paragraph per file, or one file with paragraphs separated by blank lines.

## Run

Used automatically when building the real candidate pool or running `multi_real_synthetic`:

```bash
python data.py multi_real_synthetic
```

Or build only the pool: from Python, `from core_sentinel_guardrail.data import build_real_candidate_pool` then `build_real_candidate_pool()` (or run the dataset pipeline).
