"""
Local web dashboard for logs/risk_telemetry.jsonl (low / med / high scores over time).

Run from core-sentinel-guardrail:
  python tools/risk_dashboard_server.py
  python tools/risk_dashboard_server.py --port 8765

Open http://127.0.0.1:8765 in a browser. No raw clipboard text is stored — only hashes.

For an "online" log, set GUARDRAIL_TELEMETRY_WEBHOOK_URL to POST each record to your server,
or sync this JSONL to private storage and open the dashboard where the file lives.
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

_guardrail_dir = Path(__file__).resolve().parent.parent
if str(_guardrail_dir) not in sys.path:
    sys.path.insert(0, str(_guardrail_dir))

from guardrail_logs import RISK_TELEMETRY_JSONL  # noqa: E402


def _read_records(limit: int = 2000) -> list[dict]:
    path = RISK_TELEMETRY_JSONL
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows: list[dict] = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _summary(records: list[dict]) -> dict:
    c = {"low": 0, "med": 0, "high": 0, "unknown": 0}
    for r in records:
        k = str(r.get("risk", "unknown")).lower()
        if k in c:
            c[k] += 1
        else:
            c["unknown"] += 1
    return {"by_risk": c, "total": len(records)}


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Core Sentinel — risk telemetry</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 1rem 1.5rem; background: #fafafa; color: #222; }
    h1 { font-size: 1.25rem; }
    .bar { display: flex; gap: 1rem; flex-wrap: wrap; align-items: center; margin-bottom: 1rem; }
    .pill { padding: 0.2rem 0.6rem; border-radius: 6px; font-size: 0.85rem; }
    .low { background: #e8f5e9; color: #1b5e20; }
    .med { background: #fff8e1; color: #f57f17; }
    .high { background: #ffebee; color: #b71c1c; }
    table { border-collapse: collapse; width: 100%; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
    th, td { text-align: left; padding: 0.45rem 0.6rem; border-bottom: 1px solid #eee; font-size: 0.85rem; }
    th { background: #f5f5f5; }
    code { font-size: 0.78rem; word-break: break-all; }
    select, button { padding: 0.35rem 0.6rem; }
    .hint { color: #666; font-size: 0.85rem; max-width: 48rem; }
  </style>
</head>
<body>
  <h1>Risk telemetry</h1>
  <p class="hint">Rows come from <code>logs/risk_telemetry.jsonl</code> (hash-only, no raw text).
  Use bubble <strong>Correct/Wrong</strong> feedback + <code>merge_feedback_to_training.py</code> to grow the training set and retrain.</p>
  <div class="bar">
    <label>Filter <select id="risk">
      <option value="all">all</option>
      <option value="low">low</option>
      <option value="med">med</option>
      <option value="high">high</option>
    </select></label>
    <button type="button" id="reload">Refresh</button>
    <span id="summary"></span>
  </div>
  <table>
    <thead><tr><th>Time</th><th>Risk</th><th>Score</th><th>Action</th><th>Context</th><th>PII class</th><th>Hash</th></tr></thead>
    <tbody id="tbody"></tbody>
  </table>
  <script>
    async function load() {
      const r = document.getElementById('risk').value;
      const res = await fetch('/api/records?risk=' + encodeURIComponent(r) + '&limit=500');
      const data = await res.json();
      document.getElementById('summary').textContent =
        'Total (loaded): ' + data.summary.total +
        ' — low ' + data.summary.by_risk.low +
        ', med ' + data.summary.by_risk.med +
        ', high ' + data.summary.by_risk.high;
      const tb = document.getElementById('tbody');
      tb.innerHTML = '';
      for (const row of data.records) {
        const tr = document.createElement('tr');
        const risk = (row.risk || '').toLowerCase();
        const cls = risk === 'high' ? 'high' : (risk === 'med' ? 'med' : 'low');
        tr.innerHTML = '<td>' + escapeHtml(row.timestamp || '') + '</td>' +
          '<td><span class="pill ' + cls + '">' + escapeHtml(row.risk || '') + '</span></td>' +
          '<td>' + escapeHtml(String(row.risk_score ?? '')) + '</td>' +
          '<td>' + escapeHtml(row.action || '') + '</td>' +
          '<td>' + escapeHtml(row.score_context || '') + '</td>' +
          '<td>' + escapeHtml(String(row.pii_class || '').slice(0, 80)) + '</td>' +
          '<td><code>' + escapeHtml((row.text_hash || '').slice(0, 20)) + '…</code></td>';
        tb.appendChild(tr);
      }
    }
    function escapeHtml(s) {
      const d = document.createElement('div');
      d.textContent = s;
      return d.innerHTML;
    }
    document.getElementById('reload').onclick = load;
    document.getElementById('risk').onchange = load;
    load();
    setInterval(load, 15000);
  </script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/api/records":
            q = parse_qs(parsed.query)
            filt = (q.get("risk") or ["all"])[0].lower()
            try:
                lim = min(5000, max(1, int((q.get("limit") or ["500"])[0])))
            except ValueError:
                lim = 500
            recs = _read_records(limit=50_000)
            if filt in ("low", "med", "high"):
                recs = [r for r in recs if str(r.get("risk", "")).lower() == filt]
            recs = recs[-lim:]
            summ = _summary(recs)
            out = json.dumps({"records": recs, "summary": summ}, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
            return
        self.send_error(404)


def main() -> int:
    ap = argparse.ArgumentParser(description="Risk telemetry dashboard (local JSONL)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    path = RISK_TELEMETRY_JSONL
    print(f"Telemetry file: {path} (exists: {path.exists()})")
    print(f"Dashboard: http://{args.host}:{args.port}/")
    server = HTTPServer((args.host, args.port), _Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
