# Risk telemetry (low / med / high) and growing accuracy

## What gets recorded

Every clipboard score appends one line to **`logs/risk_telemetry.jsonl`** (and still writes **`logs/guardrail.db`** as before). Each line is JSON:

- `timestamp`, `text_hash` (SHA-256 of text — **no raw paste stored here**)
- `risk`: `low` | `med` | `high`
- `risk_score`: 0–100 (same heuristic as the UI)
- `action`: `silent` | `warn` | `block`
- `pii_class`, `confidence`, `score_context` (`clipboard`, `document`, or `plain`)

Disable JSONL only: set **`GUARDRAIL_RISK_TELEMETRY=0`**.

## View batches in a browser (local)

From `core-sentinel-guardrail`:

```bash
python tools/risk_dashboard_server.py
```

Open **http://127.0.0.1:8765** — filter by risk, refresh, or leave the tab open (auto-refresh every 15s).

## “Online” log (your own endpoint)

Set **`GUARDRAIL_TELEMETRY_WEBHOOK_URL`** to an HTTPS URL. Each score **POST**s the same JSON record (fire-and-forget background thread). You can:

- Point it at a small Cloudflare Worker / AWS Lambda / Flask app that stores rows in a database.
- Use **Zapier**, **Make**, or **Supabase Edge Functions** to append to a table.

No extra Python dependencies are required for the webhook.

## Cloud storage (Supabase)

You already have two ways to get data off the machine:

1. **SQLite → Supabase table `guardrail_events`**  
   When **`SUPABASE_URL`** and **`SUPABASE_ANON_KEY`** are set, **`sentinel_sync_daemon`** (started from `main.py`) copies new rows from **`logs/guardrail.db`** (`scoring_events`) every 60s. Same hashes and actions; good for a warehouse or your hosted dashboards.

2. **Telemetry JSON → Supabase table `risk_telemetry`** (optional, mirrors JSONL)  
   - In Supabase SQL Editor, run **`docs/sql/risk_telemetry.sql`** to create the `payload` jsonb table and RLS policies.  
   - Set **`GUARDRAIL_TELEMETRY_SUPABASE=1`** (same `SUPABASE_*` env vars as above).  
   - Each score inserts one row with the full telemetry object in **`payload`** (same shape as one line of `risk_telemetry.jsonl`).  
   - Override table name with **`GUARDRAIL_TELEMETRY_SUPABASE_TABLE`** if needed.

If you also use the scoring sync, you may see overlapping information in **`guardrail_events`** vs **`risk_telemetry`**; keep both, or use only the stream you prefer for analytics.

## Training: dataset size and accuracy

**Telemetry is for monitoring**, not training labels. To improve the model:

1. Use bubble **Correct / Wrong** feedback when the label is off — that goes to **`logs/feedback_store.jsonl`** (and optional full text in **`feedback_fulltext.jsonl`**).
2. Run **`python merge_feedback_to_training.py`** → **`data/user_feedback/export.jsonl`**.
3. Run **`python data.py multi_real_synthetic`**, then **`python train.py`**, then **`python evaluate_test.py`**.

See **`FEEDBACK_TRAINING_LOOP.md`** for the full loop and multi-machine git merge notes.
