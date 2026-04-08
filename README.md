# Core Sentinel

Risk-aware clipboard guardrail for LLM workflows: local TinyBERT scoring, PII-aware remediation UI, and optional feedback-driven retraining.

## Repository layout

| Path | Description |
|------|-------------|
| [`core-sentinel-guardrail/`](core-sentinel-guardrail/) | Application code (`main.py`), inference, training, configs |
| [`docs/`](docs/) | Documentation index (links to all guides) |

## Quick start

```bash
cd core-sentinel-guardrail
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
python main.py
```

- **Model weights** are not committed (see `.gitignore`). Download or train per [`core-sentinel-guardrail/README.md`](core-sentinel-guardrail/README.md).
- **Secrets**: use env / `.env` (gitignored), not committed keys.

## Documentation

| Guide | Topic |
|-------|--------|
| [Documentation index](docs/README.md) | Table of contents for all Markdown guides |
| [Guardrail README](core-sentinel-guardrail/README.md) | Training, datasets, 3-class vs 9-class |
| [Clipboard guardrail](core-sentinel-guardrail/CLIPBOARD_GUARDRAIL.md) | Paste flow, window detection, training loop |
| [GitHub feedback sync](core-sentinel-guardrail/GITHUB_FEEDBACK_SYNC.md) | Private repo, which log files to commit |
| [Feedback → training loop](core-sentinel-guardrail/FEEDBACK_TRAINING_LOOP.md) | Export feedback, rebuild data, `train.py` |

## Add this project to GitHub

1. Create a **private** repository on [GitHub](https://github.com/new) (recommended if logs may contain sensitive text).
2. On your machine, from the **repository root** (`Core_Sentinal`):

   ```bash
   git init
   git add .
   git commit -m "Initial commit: Core Sentinel guardrail"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```

3. If the repo already exists with a README, use `git pull origin main --allow-unrelated-histories` once, then push.

GitHub will render `README.md` at the repo root and show linked Markdown files with navigation.

## License

Add a `LICENSE` file if you open-source the project; otherwise keep the repository private.
