# Kaggle AI4privacy-PII (English)

Place the Kaggle dataset here for **English-only** 9-class PII training.

## Download (Kaggle CLI)

Install the [Kaggle CLI](https://github.com/Kaggle/kaggle-api) (`pip install kaggle`), then configure API credentials (kaggle.json in `~/.kaggle`).

**Bash / Git Bash:**
```bash
cd "path/to/core-sentinel-guardrail/data/ai4privacy_kaggle"
kaggle datasets download verracodeguacas/ai4privacy-pii
unzip -o ai4privacy-pii.zip
```

**PowerShell:**
```powershell
cd "E:\Rutgers_Class\Sem 4\Project\Core_Sentinal\core-sentinel-guardrail\data\ai4privacy_kaggle"
kaggle datasets download verracodeguacas/ai4privacy-pii
Expand-Archive -Path ai4privacy-pii.zip -DestinationPath . -Force
```

Or download manually from: https://www.kaggle.com/datasets/verracodeguacas/ai4privacy-pii/data and extract into this directory.

## Run

From `core-sentinel-guardrail`: `python data.py ai4privacy_kaggle_en`

Expected columns: `source_text` (or `text`/`prompt`), `target_text` (or `masked`), `language`. Rows with `language` in `en`, `en-US`, `en-GB` are kept.
