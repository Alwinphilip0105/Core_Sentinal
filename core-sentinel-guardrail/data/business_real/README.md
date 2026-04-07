# Optional business / Enron-style real data

Place manually labeled enterprise or Enron-style snippets here. Used for real-data training or the manual-labeling candidate pool.

## Where to place files

Put files in this directory: `data/business_real/`

## Supported file types

- **.csv** – CSV with text and optional label column
- **.json** – JSON array of objects
- **.jsonl** – One JSON object per line

## Expected columns (labeled data)

- **Text**: `text`, `content`, `excerpt`, `prompt`, `input`, or `body`
- **Label** (optional): `risk`, `label`, `category`, `classification`, or `sensitivity`  
  If missing, rows are treated as **unlabeled candidates** for the labeling queue.

## Accepted label values

Same mapping as Patronus: `public`/`internal` → low; `confidential` → med; `restricted`/`sensitive` → high.

## Examples

**JSONL (labeled):**
```json
{"text": "Q3 revenue was $2.1M. Key accounts: Acme Corp.", "risk": "confidential"}
{"text": "Internal team lunch next Tuesday.", "sensitivity": "internal"}
```

**CSV (unlabeled):**
```csv
text
Board packet draft attached—confidential.
Meeting notes: John, Sarah. Timeline TBD.
```

## Run

Used automatically when building the real candidate pool or running `multi_real_synthetic`:

```bash
python data.py multi_real_synthetic
```
