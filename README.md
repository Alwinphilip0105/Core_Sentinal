# Core Sentinel — ML Guardrail for LLM Chat Windows

> A Grammarly-style floating overlay that detects and protects Personally Identifiable Information (PII) when pasting content into LLM chat windows like ChatGPT, Claude, Gemini, and Comet.

**Repository layout:** see **[REPO_LAYOUT.md](REPO_LAYOUT.md)** (folders, entry points, what belongs where).

---

## What it does

Core Sentinel sits quietly in the corner of your screen. The moment you paste sensitive content into an LLM window, it intercepts, scores, and gives you options — redact, rephrase, encrypt, or proceed. It only activates when you are using an LLM, leaving all other applications completely unaffected.

---

## Demo

### 1. Floating pill — idle vs active

The pill stays in the corner at low opacity. When you hover, it darkens and shows options. The color changes based on risk level.

| State | Color | Meaning |
|-------|-------|---------|
| Grey pill | No color | Not in LLM window — not monitoring |
| Green dot | Green | Monitoring active — LLM detected |
| Amber badge | Amber | Medium risk paste detected |
| Red badge + number | Red | High risk — N issues found |

---

### 2. Safe paste — no interruption

Paste: `"The meeting is on Tuesday at 3pm in conference room B."`

Result: Spinner briefly appears, returns to idle. No badge, no popup. Event logged silently to `events.csv` (or the primary structured log when legacy CSV is disabled).

---

### 3. Medium risk — warn only

Paste: `"Please contact John at john.smith@company.com or call 212-555-0134."`

Result: Badge appears on pill with count. Click the pill to open the panel.

![Panel showing issues](screenshot_panel_issues.png)

The panel shows each detected PII item with Fix / Skip buttons. The score wheel shows 70 in amber.

---

### 4. High risk — block with remediation

Paste: `"My SSN is 123-45-6789 and card 4532-1234-5678-9012"`

Result: Panel opens immediately showing HIGH risk items. Four actions available at the bottom:

| Button | What it does |
|--------|-------------|
| **Redact** | Partially masks PII — `****-****-****-9012` |
| **Rephrase** | Rewrites naturally — `"their payment method"` |
| **Fix all** | Replaces with label — `[credit card pattern]` |
| **Snooze ▾** | Pause alerts for 5 min / 10 min / 30 min / 1 hr |

---

### 5. Panel — safe state

![Panel safe state](screenshot_panel_safe.png)

When no PII is detected the panel shows a shield and "No PII detected" with general best practice tips.

---

### 6. Toolbar on hover

Hover over the pill to reveal the vertical toolbar:

![Toolbar](screenshot_toolbar.png)

| Icon | Action | Shortcut |
|------|--------|----------|
| ⏻ | Toggle monitoring on/off | Alt+G |
| ✎ | Rephrase clipboard text | Alt+R |
| ▓ | Redact PII from clipboard | Alt+D |
| 🔒 | Encrypt clipboard content | Alt+E |
| 📄 | Scan a file for PII | Alt+S |
| ⚙ | Open settings | Alt+, |

---

### 7. Settings panel

![Settings](screenshot_settings.png)

| Setting | Default | Description |
|---------|---------|-------------|
| Monitor LLM windows only | ON | Only activates in ChatGPT, Claude, etc. |
| Block high-risk pastes | ON | Shows panel for high-risk content |
| Show badge count | ON | Red number on pill |
| Auto-redact on block | OFF | Automatically redacts without asking |
| Sound on block | OFF | Audio alert on high-risk detection |
| Sensitivity | Medium | Low / Medium / High / Strict |

---

### 8. Document scan

Click the scan icon in the toolbar to scan any file for PII.

Supported formats: PDF · DOCX · XLSX · PPTX · PNG · JPG · TXT

![Document scan report](screenshot_doc_scan.png)

The scan report shows:

- Overall risk score and level
- Every page where PII was found
- Exact PII spans highlighted with their class
- Gemini AI findings (if API key is set)

---

## Model performance

Trained on 9,711 examples (17k raw, balanced to 3-class). **Test set: 5,500 examples.**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| **low** | 1.000 | 1.000 | **1.000** | 500 |
| **med** | 0.478 | 0.326 | 0.387 | 172 |
| **high** | 0.976 | 0.987 | **0.982** | 4828 |

- **Overall accuracy:** 97.0%
- **Macro F1:** 0.790

**Training improvements (v1 → v2):** Added 3,000 synthetic low-risk examples (Faker), Financial PII XLSX as a 4th training source, and a balanced split (6,000 high / 3,000 med / 3,000 low). Low-class F1 improved from 0.00 to 1.00; macro F1 from 0.662 to 0.790; accuracy 96.9% → 97.0%.

**Key design decision:** The model errs toward over-flagging medium content as high rather than under-flagging. A false positive (unnecessary warning) is far less harmful than a false negative (missed SSN or API key).

---

## Architecture

```
Clipboard paste (Ctrl+V)
        │
        ▼
active_window_llm.py ──► Not LLM? → Skip entirely
        │
        ▼ (LLM detected)
guardrail_runtime.py ──► Duplicate in last 60s? → Skip
        │
        ▼
InferenceWorker (QThread)
        │
        ├── infer.py (TinyBERT sliding window, 128 tokens/chunk)
        ├── risk_mapping.py (regex: SSN, card, JWT, email...)
        └── kb_loader.py (company-specific rules)
        │
        ▼
get_action() → silent / warn / block
        │
        ▼
ui_risk_bubble.py ──► update pill color + badge
        │
        ▼ (warn or block)
ui_remediation_dialog.py ──► show panel with Fix/Skip/Redact
        │
        ▼
feedback_store.py ──► log user correction
events.csv ──────────► log every decision (optional legacy CSV)
```

---

## Quick start

### Prerequisites

- Windows 10/11 (x64)
- Python 3.11
- Visual C++ Redistributable 2022 — [download](https://aka.ms/vs/17/release/vc_redist.x64.exe)

### Install

```powershell
git clone https://github.com/Alwinphilip0105/Core_Sentinal.git
cd Core_Sentinal

python -m venv .venv
.venv\Scripts\Activate.ps1

pip install -r core-sentinel-guardrail/requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Configure (optional)

Create a `.env` file in the project root (you can copy **`.env.example`** to `.env` and edit):

```env
# Gemini AI for document scanning (free tier)
# Get key at: https://aistudio.google.com/app/apikey
GEMINI_API_KEY=your-key-here

# Supabase for cloud sync (free tier)
# Get at: https://supabase.com
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key

# Optional: mirror each risk telemetry row to Supabase (see docs/sql/risk_telemetry.sql)
# GUARDRAIL_TELEMETRY_SUPABASE=1

# Optional: POST each telemetry JSON to your own HTTPS endpoint (Zapier, Edge Function, etc.)
# GUARDRAIL_TELEMETRY_WEBHOOK_URL=https://example.com/ingest

# Optional tuning
GUARDRAIL_BLOCKING_PRELOAD=0
GUARDRAIL_DEBUG_WINDOW=0
```

For **GitHub sync, backup scope, and recovery** after disk loss or a new machine, see **[docs/BACKUP_AND_RECOVERY.md](docs/BACKUP_AND_RECOVERY.md)**.

For a **production-style workflow** (branching, secrets, tests before ship, ML release order, demo checklist), see **[docs/PRODUCTION_AND_RELEASE.md](docs/PRODUCTION_AND_RELEASE.md)**. Contributors: **[CONTRIBUTING.md](CONTRIBUTING.md)**.

### Run

```powershell
cd Core_Sentinal
.venv\Scripts\python.exe run_guardrail.py
```

The pill appears in the bottom-right corner. Open ChatGPT or Claude in your browser and start pasting.

---

## Training your own model

```powershell
cd core-sentinel-guardrail

# Build dataset
python data.py

# Train TinyBERT classifier
$env:ALLOW_TORCH_LOAD_PRE26 = '1'
python train.py

# Evaluate with PR curve
python evaluate_test.py

# Calibrate thresholds from PR curve
python calibrate_thresholds.py

# Run smoke tests
python adversarial_tests.py
```

---

## Company knowledge base

Create `config/company_policy.yaml` with your company's custom rules:

```yaml
company: "Acme Corp"
version: "1.0"

custom_pii_patterns:
  - name: "Employee ID"
    regex: "EMP-\\d{6}"
    risk: "high"
  - name: "Project code"
    regex: "PRJ-[A-Z]{3}-\\d{4}"
    risk: "med"

forbidden_terms:
  - "confidential"
  - "internal only"
  - "trade secret"

sensitivity_overrides:
  FINANCIAL: "block"
  AUTH: "block"
  NAME: "warn"
```

Set the path:

```powershell
$env:GUARDRAIL_KB_PATH = "config/company_policy.yaml"
```

To deploy across multiple companies, host the YAML in a private GitHub repo and set:

```powershell
$env:GUARDRAIL_KB_URL = "https://raw.githubusercontent.com/org/repo/main/policy.yaml"
$env:GUARDRAIL_KB_TOKEN = "your-github-token"
```

---

## Feedback and relearning

Every paste decision is logged. Users can correct wrong flags using the Correct / Wrong buttons that appear after each detection.

```
User clicks "Wrong" on a false positive
        │
        ▼
feedback_store.jsonl ← {text_hash, predicted: "high", correct: "low"}
        │
        ▼ (after 30 corrections)
Tray notification: "30 corrections ready — run train.py"
        │
        ▼
python train.py  ← fine-tunes on corrections, 3 epochs LR=1e-5
        │
        ▼
Improved model deployed to models/tinybert_guardrail
```

---

## Supported LLM windows

Detected automatically by window title and browser URL:

- ChatGPT (`chat.openai.com`)
- Claude (`claude.ai`)
- Gemini (`gemini.google.com`)
- Microsoft Copilot (`copilot.microsoft.com`)
- Perplexity (`perplexity.ai`)
- Comet browser
- Poe (`poe.com`)
- Any URL containing `/chat` or `assistant`

Add custom LLMs without editing code:

```powershell
$env:GUARDRAIL_EXTRA_LLM_TITLES = "myapp,internal-gpt,llama"
```

---

## Masking modes

| Mode | Example input | Example output |
|------|--------------|----------------|
| Partial (default) | `john@company.com` | `j***n@company.com` |
| Partial | `212-555-0134` | `***-***-0134` |
| Partial | `123-45-6789` | `***-**-6789` |
| Partial | `4532-1234-5678-9012` | `****-****-****-9012` |
| Partial | `sk-abc123secret` | `sk-a****cret` |
| Stars | `John Smith` | `**** *****` |
| Full | `John Smith` | `[NAME]` |
| Rephrase | `John Smith` | `the person` |

---

## Project structure

```
Core_Sentinal/
├── run_guardrail.py              ← entry point (delegates to launchers/)
├── launchers/                    ← launcher implementation (PyTorch DLL, main import)
├── REPO_LAYOUT.md                ← folder map (read this first)
├── scripts/                      ← dev helpers (e.g. manual_probe.py)
├── docs/                         ← guides, Windows setup, Supabase SQL, static hub HTML
├── legacy/rutgers_demo/          ← small older demo (not the full guardrail)
├── .env                          ← API keys (not in git)
├── .venv/                        ← Python environment (optional location; see REPO_LAYOUT.md)
└── core-sentinel-guardrail/
    ├── main.py                   ← PyQt6 app + keyboard hook
    ├── infer.py                  ← TinyBERT inference + sliding window
    ├── risk_mapping.py           ← regex PII patterns
    ├── risk_policy_loader.py     ← threshold config loader
    ├── kb_loader.py              ← company knowledge base
    ├── access_control.py         ← role-based paste permission
    ├── feedback_store.py         ← user correction logging
    ├── pii_remediation.py        ← mask / rephrase / encrypt
    ├── text_extractor.py         ← document text extraction
    ├── gemini_scanner.py         ← Gemini AI PII detection
    ├── document_scanner.py       ← full document scan pipeline
    ├── active_window_llm.py      ← LLM window detection
    ├── guardrail_runtime.py      ← dedup + snooze state
    ├── ui_risk_bubble.py         ← floating pill widget
    ├── ui_remediation_dialog.py  ← side panel
    ├── toast.py                  ← in-app notifications
    ├── data.py                   ← dataset builder
    ├── train.py                  ← TinyBERT training
    ├── evaluate_test.py          ← PR curve + metrics
    ├── calibrate_thresholds.py   ← threshold derivation
    ├── adversarial_tests.py      ← smoke tests
    ├── config/
    │   ├── risk_policy.json      ← per-class thresholds
    │   ├── pii_policy.json       ← allow/warn/block rules
    │   ├── pii_to_risk.json      ← 9-class → low/med/high mapping
    │   └── company_policy_template.yaml
    ├── models/
    │   └── tinybert_guardrail/   ← trained model weights (gitignored)
    ├── logs/
    │   ├── events.csv            ← optional legacy decision log
    │   ├── feedback_store.jsonl  ← user corrections
    │   └── scan_reports/         ← HTML document scan reports
    └── reports/
        ├── train_eval_summary.json
        ├── pr_curve_summary.json
        └── train_log_history.json
```

Generated `reports/` and most `logs/` artifacts are listed in `.gitignore`; regenerate locally after training or evaluation.

---

## Demo test cases

Copy and paste each into ChatGPT or Claude to test:

```
# Safe — no alert expected
The weather in New York today is partly cloudy with 
temperatures around 72 degrees.

# Medium — amber badge expected  
Please contact John at john.smith@company.com or 
call 212-555-0134 if you need to reschedule.

# High — block expected
My SSN is 123-45-6789 and date of birth 04/15/1990.

# Critical — immediate block expected
API_KEY=sk-abc123XYZsecretkey9999
DB_PASSWORD=MyP@ssw0rd123!

# Financial — block expected
Please charge card 4532-1234-5678-9012 expiry 09/27.

# Long text — sliding window test (SSN buried in text)
This is a regular update. Sales are up 15%. 
The engineering team shipped 3 features. 
My Social Security Number is 987-65-4321 urgent.
The finance team closed books on time.

# Mixed language — robustness test
Hola, mi nombre es Carlos. My credit card is 
5425-2334-3010-9903. Por favor procesa.
```

---

## Further documentation

| Guide | Topic |
|-------|--------|
| [Documentation index](docs/README.md) | Table of contents for Markdown guides |
| [Guardrail README](core-sentinel-guardrail/README.md) | Label modes (3-class / 9-class), datasets, training |
| [Clipboard guardrail](core-sentinel-guardrail/CLIPBOARD_GUARDRAIL.md) | Paste flow, window detection, calibration |
| [GitHub feedback sync](core-sentinel-guardrail/GITHUB_FEEDBACK_SYNC.md) | Which log files to sync in a private repo |
| [Feedback → training](core-sentinel-guardrail/FEEDBACK_TRAINING_LOOP.md) | Merge feedback into training data |
| [Risk telemetry](core-sentinel-guardrail/RISK_TELEMETRY.md) | `risk_telemetry.jsonl`, dashboard server |

---

## Screenshots

Add `screenshot_*.png` files at the repository root (paths used above) so the Demo images render on GitHub.

---

## License

MIT License — free to use, modify, and distribute. Full text: [`LICENSE`](LICENSE).

---

*Built as a machine learning capstone project — Rutgers University, Spring 2026*
