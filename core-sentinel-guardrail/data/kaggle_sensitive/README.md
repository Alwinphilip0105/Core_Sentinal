# Kaggle / structured-sensitive data

Place Kaggle or other structured sensitive-datasets here. Used for real-data training or the manual-labeling candidate pool.

## Where to place files

Put files in this directory: `data/kaggle_sensitive/`

## Supported file types

- **.csv** – CSV with text and optional label column
- **.json** – JSON array of objects
- **.jsonl** – One JSON object per line

## Expected columns (labeled data)

- **Text**: `text`, `content`, `excerpt`, `prompt`, `input`, `description`, or `body`
- **Label** (optional): `risk`, `label`, `category`, `classification`, or `sensitivity`  
  If missing, rows are treated as **unlabeled candidates** for the labeling queue.

## Accepted label values

Mapped to low/med/high like Patronus: `public`/`internal` → low; `confidential` → med; `restricted`/`sensitive` → high.

## Examples

**CSV (labeled):**
```csv
text,label
"Customer account 12345 balance inquiry",internal
"SSN and routing on file for wire",restricted
```

**JSONL (unlabeled):**
```json
{"description": "Quarterly revenue summary for leadership."}
{"text": "API key rotation schedule."}
```

## Run

Used automatically when building the real candidate pool or running `multi_real_synthetic`:

```bash
python data.py multi_real_synthetic
```
