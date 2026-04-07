# Enron / EDRM Enron email corpus

Place Enron or EDRM-style email/body text here for real-data training or the manual-labeling candidate pool.

## Where to place files

Put files in this directory: `data/enron_real/`

## Supported file types

- **.csv** – CSV with a text/body column (and optional label column)
- **.json** – JSON array of objects with text and optional label
- **.jsonl** – One JSON object per line
- **.txt** – Plain text: each file is split by double newline into snippets; snippets ≥20 chars are loaded as rows. If no paragraphs, the whole file is one row if long enough.

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
