# Clipboard Guardrail – End-to-End Flow

When the user presses **Ctrl+V** with the Windows clipboard app running:

1. **Hook** – The app has registered a global keyboard hook (via `keyboard`) for key events. On Ctrl+V (key down for `v` while Ctrl is pressed), the hook runs before the key is passed to the active application.

2. **Read & score** – The handler reads the current clipboard text with `pyperclip.paste()`, then calls `score_clipboard_with_pii(text, policy_override)`. That runs the PII model (and regex overrides), maps labels to risk, applies policy, and returns `risk`, `decision`, `block`, `message`, `triggers`.

3. **Log** – A JSON line is appended to `logs/clipboard_events.jsonl` with `timestamp`, `risk`, `decision`, `block`, `pii_override_applied`, `triggers`, `policy_mode`, and `text_hash` (no raw text).

4. **Popup & suppress**  
   - **Low risk / allow** – No popup (or a small toast if `--show-allow-toast`). The hook returns “allow,” so Ctrl+V is **not** suppressed and the paste goes through.  
   - **Medium risk / warn** – A MessageBox: “Clipboard Guardrail Warning” with `message` and **Yes (Proceed)** / **No (Cancel)**. If the user clicks **Yes**, paste is allowed (key not suppressed). If **No**, the hook suppresses the key and the paste does **not** happen.  
   - **High risk / block** – A MessageBox: “Clipboard Guardrail Blocked” with `message` and **OK** only. The hook always suppresses Ctrl+V; paste is **not** allowed.

5. **Policy mode**  
   - **Default** (`--mode default`): High risk uses “warn” (Yes/No).  
   - **Strict** (`--mode strict`): High risk uses “block” (OK only, no override). Implemented by passing `policy_override={"high": {"allow_warn_instead": False}}` into `score_clipboard_with_pii`.

## Example flows

- **Low-risk text** (e.g. “Team standup at 9am”) → Allow → No popup (or toast) → Paste proceeds.  
- **Medium-risk text** (e.g. “Meeting with John Smith in Newark”) → Warn → Popup with “names or general personal details” → User chooses Yes → Paste proceeds; No → Paste suppressed.  
- **High-risk text with triggers** (e.g. “SSN 123-45-6789 and account details”) → In default mode: Warn popup (with “e.g., SSN pattern” if triggers present); in strict mode: Block popup → Paste always suppressed.

## Run commands

From `core-sentinel-guardrail`:

```bash
# Default: high -> warn (user can still proceed)
python windows_clipboard_app.py

# Strict: high -> block (no override)
python windows_clipboard_app.py --mode strict

# Optional: show a small popup even on allow
python windows_clipboard_app.py --show-allow-toast
```

**Note:** The app needs to run with sufficient privileges so the global hook can capture Ctrl+V. On Windows, suppression works when the hook returns `False` for that key event.

---

## Risk-Aware Assistant (bubble + URL-based agent)

Run `python main.py` for the floating RiskBubble and remediation dialog. Detection uses the **active tab URL** when possible so the bubble shows the real agent name (ChatGPT, Claude, Gemini) instead of "Brave".

**To enable URL-based agent detection:** start Brave (or Chrome/Edge) with remote debugging:

- **Windows (PowerShell):**  
  `Start-Process "C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe" -ArgumentList "--remote-debugging-port=9222"`
- Or create a shortcut that runs:  
  `brave.exe --remote-debugging-port=9222`

Then run `python main.py`. When you paste in a ChatGPT tab, the bubble will show **ChatGPT** (from the URL) instead of **Brave**. Optional: set `GUARDRAIL_CDP_PORT=9233` if you use a different port.

---

## Training, calibration, and dataset rebuild (full loop)

From `core-sentinel-guardrail`, typical order:

1. **Evaluate on the test split** (metrics + PR curve + `reports/pr_curve_summary.json`):

   ```bash
   python evaluate_test.py
   ```

2. **Apply calibrated thresholds** from the PR summary into `config/risk_policy.json`:

   ```bash
   python calibrate_thresholds.py
   ```

3. **Point at Financial PII xlsx** and **rebuild** Arrow datasets. The env var must be a folder containing `Training_Set.xlsx` and/or `Testing_Set.xlsx`. Use the **`multi_real_synthetic`** data source so `load_financial_pii_xlsx()` is merged (see `data.py`).

   **PowerShell (session only):**

   ```powershell
   $env:GUARDRAIL_FINANCIAL_PII_PATH = "C:\path\to\financial_pii_folder"
   python data.py multi_real_synthetic
   ```

   **cmd.exe:**

   ```bat
   set GUARDRAIL_FINANCIAL_PII_PATH=C:\path\to\financial_pii_folder
   python data.py multi_real_synthetic
   ```

   **bash:**

   ```bash
   export GUARDRAIL_FINANCIAL_PII_PATH=/path/to/financial_pii_folder
   python data.py multi_real_synthetic
   ```

4. **Retrain** the classifier:

   ```bash
   python train.py
   ```

5. **Evaluate again** to see if F1 (and related metrics) improved:

   ```bash
   python evaluate_test.py
   ```

6. **Refresh thresholds** after retraining (re-reads the new `pr_curve_summary.json` from step 5):

   ```bash
   python calibrate_thresholds.py
   ```

   For the **legacy validation-set sweep** (writes `threshold_calibration.json` instead), use:

   ```bash
   python calibrate_thresholds.py --validation-sweep
   ```

7. **Smoke test** inference + guardrail behavior:

   ```bash
   python adversarial_tests.py
   ```

**Note:** Step 3 alone (`python data.py` with no args) defaults to **nemotron**, not the multi-source + Financial PII mix—use `multi_real_synthetic` as above.
