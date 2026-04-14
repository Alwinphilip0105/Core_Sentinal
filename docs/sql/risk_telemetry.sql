-- Optional: store each risk telemetry row (same JSON as logs/risk_telemetry.jsonl) in Supabase.
-- Run in Supabase: SQL Editor → New query → Paste → Run
--
-- App env: GUARDRAIL_TELEMETRY_SUPABASE=1
--          SUPABASE_URL, SUPABASE_ANON_KEY (same as sentinel sync)
-- Optional: GUARDRAIL_TELEMETRY_SUPABASE_TABLE=my_table (default: risk_telemetry)

CREATE TABLE IF NOT EXISTS risk_telemetry (
  id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  received_at timestamptz DEFAULT now() NOT NULL,
  payload jsonb NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_risk_telemetry_received ON risk_telemetry (received_at DESC);

ALTER TABLE risk_telemetry ENABLE ROW LEVEL SECURITY;

-- Desktop app uses anon key: allow inserts from the client (tighten for production, e.g. auth-only).
CREATE POLICY "anon can insert risk_telemetry"
ON risk_telemetry FOR INSERT
WITH CHECK (true);

-- Optional: allow dashboard/service to read (or restrict to authenticated users).
CREATE POLICY "anon can select risk_telemetry"
ON risk_telemetry FOR SELECT
USING (true);
