-- Run in Supabase: SQL Editor → New query → Paste → Run
-- Knowledge base entries for hub Knowledge Base Manager (docs/hub.html)

CREATE TABLE IF NOT EXISTS kb_entries (
  id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  filename text NOT NULL,
  category text NOT NULL,
  risk_level text DEFAULT 'warn',
  patterns jsonb DEFAULT '[]'::jsonb,
  raw_length integer DEFAULT 0,
  uploaded_by text,
  uploaded_at timestamptz DEFAULT now(),
  active boolean DEFAULT true
);

ALTER TABLE kb_entries ENABLE ROW LEVEL SECURITY;

CREATE POLICY "anon can read active kb"
ON kb_entries FOR SELECT
USING (active = true);

CREATE POLICY "anon can insert kb"
ON kb_entries FOR INSERT
WITH CHECK (true);

CREATE POLICY "anon can delete kb"
ON kb_entries FOR DELETE
USING (true);
