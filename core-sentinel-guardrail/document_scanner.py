"""
Full document scan: extract text, local PII scoring, optional Gemini, SQLite + HTML report.
"""

from __future__ import annotations

import os
from pathlib import Path

# Load .env file if it exists (for local development)
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

import html
import json
import sqlite3
import sys
import traceback
import webbrowser
from datetime import datetime

_GUARDRAIL_ROOT = Path(__file__).resolve().parent
if str(_GUARDRAIL_ROOT) not in sys.path:
    sys.path.insert(0, str(_GUARDRAIL_ROOT))

from gemini_scanner import merge_local_and_gemini, scan_with_gemini  # noqa: E402
from infer import score_clipboard_with_pii  # noqa: E402
from text_extractor import extract  # noqa: E402

DB_PATH = _GUARDRAIL_ROOT / "logs" / "scan_results.db"


def _init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scans (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            scanned_at  TEXT,
            file_name   TEXT,
            file_path   TEXT,
            file_size   INTEGER,
            page_count  INTEGER,
            method      TEXT,
            overall_risk TEXT,
            risk_score  INTEGER,
            issue_count INTEGER,
            pii_classes TEXT,
            gemini_used INTEGER,
            report_path TEXT,
            findings    TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def scan_document(
    file_path: str,
    use_gemini: bool = True,
    progress_cb=None,
) -> dict:
    """
    Full document scan pipeline.
    progress_cb(pct, message) for UI updates.
    """
    path = Path(file_path)
    if not path.exists():
        return {"error": f"File not found: {file_path}", "file": file_path}

    def _progress(pct: int, msg: str) -> None:
        if progress_cb:
            progress_cb(pct, msg)
        print(f"[scan] {pct}% {msg}")

    try:
        _init_db()

        _progress(10, f"Extracting text from {path.name}...")
        extracted = extract(file_path)

        print(f"[scan] extracted text length: {len(extracted.get('text', '') or '')}")
        print(f"[scan] page count: {extracted.get('page_count', 0)}")
        pages = extracted.get("pages", [])
        print(f"[scan] pages to score: {len(pages)}")

        if extracted.get("error"):
            return {
                "error": extracted["error"],
                "file": file_path,
                "overall_risk": "unknown",
                "issue_count": 0,
                "report_path": None,
            }

        _progress(30, "Running local PII detection...")
        page_results = []

        for i, page in enumerate(pages):
            pct = 30 + int((i / max(len(pages), 1)) * 30)
            _progress(pct, f"Scanning page {i + 1} of {len(pages)}...")

            page_text = page.get("text", "").strip()
            if not page_text:
                continue

            result = score_clipboard_with_pii(page_text, context="document")
            result["page"] = page.get("page", i + 1)
            result["text_preview"] = page_text[:200]
            page_results.append(result)

        gemini_findings: list = []
        if use_gemini and os.environ.get("GEMINI_API_KEY"):
            _progress(65, "Running Gemini AI analysis...")
            full_text = extracted.get("text", "")
            gemini_findings = scan_with_gemini(full_text)

        _progress(80, "Merging findings...")

        overall_risk = "low"
        if any(r.get("action") == "block" for r in page_results):
            overall_risk = "high"
        elif any(r.get("action") == "warn" for r in page_results):
            overall_risk = "med"

        if gemini_findings:
            merged_doc = merge_local_and_gemini(
                {
                    "risk": overall_risk,
                    "spans": [s for r in page_results for s in r.get("spans", [])],
                },
                gemini_findings,
            )
            overall_risk = merged_doc.get("risk", overall_risk)

        flagged_pages = [r for r in page_results if r.get("action") in ("warn", "block")]

        all_classes = list(
            {s.get("class", "") for r in page_results for s in r.get("spans", [])}
            | {f.get("type", "") for f in gemini_findings}
        )
        all_classes = [c for c in all_classes if c]

        max_score = max((r.get("risk_score", 0) for r in page_results), default=0)

        scan_result = {
            "file": str(path),
            "file_name": path.name,
            "file_size": extracted.get("file_size", 0),
            "page_count": extracted.get("page_count", 1),
            "method": extracted.get("method", "unknown"),
            "overall_risk": overall_risk,
            "risk_score": max_score,
            "issue_count": len(flagged_pages) + len(gemini_findings),
            "pii_classes": all_classes,
            "flagged_pages": flagged_pages,
            "gemini_findings": gemini_findings,
            "gemini_used": bool(gemini_findings),
            "scanned_at": datetime.now().isoformat(),
            "summary": (f"{len(flagged_pages)} of {len(pages)} pages contain PII"),
        }

        _progress(90, "Saving to database...")
        row_id = _save_to_db(scan_result)

        _sync_to_supabase(scan_result)

        _progress(95, "Generating report...")
        report_path = _generate_html_report(scan_result, file_path=file_path)
        scan_result["report_path"] = report_path
        _update_db_report_path(row_id, report_path)

        _progress(100, "Done!")
        return scan_result

    except Exception as e:
        traceback.print_exc()
        return {
            "error": str(e),
            "file": file_path,
            "overall_risk": "unknown",
            "issue_count": 0,
            "report_path": None,
        }


def _save_to_db(result: dict) -> int:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        """
        INSERT INTO scans
        (scanned_at, file_name, file_path, file_size,
         page_count, method, overall_risk, risk_score,
         issue_count, pii_classes, gemini_used,
         report_path, findings)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            result["scanned_at"],
            result["file_name"],
            result["file"],
            result["file_size"],
            result["page_count"],
            result["method"],
            result["overall_risk"],
            result["risk_score"],
            result["issue_count"],
            ",".join(result["pii_classes"]),
            1 if result["gemini_used"] else 0,
            result.get("report_path", ""),
            json.dumps(result["flagged_pages"]),
        ),
    )
    conn.commit()
    row_id = int(cur.lastrowid)
    conn.close()
    return row_id


def _update_db_report_path(row_id: int, report_path: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE scans SET report_path = ? WHERE id = ?", (report_path, row_id))
    conn.commit()
    conn.close()


def _sync_to_supabase(result: dict) -> None:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_ANON_KEY")
    if not url or not key:
        return
    try:
        from supabase import create_client

        sb = create_client(url, key)
        sb.table("document_scans").insert(
            {
                "scanned_at": result["scanned_at"],
                "file_name": result["file_name"],
                "overall_risk": result["overall_risk"],
                "risk_score": result["risk_score"],
                "issue_count": result["issue_count"],
                "pii_classes": ",".join(result["pii_classes"]),
                "gemini_used": result["gemini_used"],
            }
        ).execute()
        print("[supabase] scan synced")
    except Exception as e:
        print(f"[supabase] sync failed (ok): {e}")


def _generate_html_report(result: dict, *, file_path: str | None = None) -> str:
    path = Path(file_path) if file_path else Path(result.get("file", "") or "")
    risk_color = {
        "high": "#C62828",
        "med": "#E65100",
        "low": "#2E7D32",
    }.get(result["overall_risk"], "#666")

    _HIGHLIGHT_MAX = 80

    cards = ""
    for page in result.get("flagged_pages", []):
        action_color = "#C62828" if page.get("action") == "block" else "#E65100"
        spans_html = ""
        section_warnings_html = ""
        for span in page.get("spans", []):
            raw_match = str(span.get("match", ""))
            c = html.escape(str(span.get("class", "")))
            if len(raw_match) <= _HIGHLIGHT_MAX:
                m = html.escape(raw_match[:40])
                spans_html += (
                    f'<span style="background:#ffebee;'
                    f'color:#c62828;padding:1px 4px;'
                    f'border-radius:3px;margin:2px;">'
                    f"{m} "
                    f"<small>({c})"
                    f"</small></span> "
                )
            else:
                nch = len(raw_match)
                section_warnings_html += (
                    f'<div style="background:#fff8e1;border:1px solid #ffe082;'
                    f"border-radius:6px;padding:8px;margin:6px 0;font-size:12px;"
                    f'color:#5d4037;">'
                    f"<strong>Section warning</strong> ({c}) — "
                    f"{nch} characters flagged as high-sensitivity; "
                    f"not shown as a single highlight."
                    f"</div>"
                )
        prev = html.escape(str(page.get("text_preview", ""))[:150])
        cards += f"""
        <div style="border:1px solid #eee;border-radius:8px;
                    padding:12px;margin:8px 0;
                    border-left:4px solid {action_color};">
          <div style="font-weight:600;color:{action_color};">
            Page {html.escape(str(page.get("page", "?")))} —
            {html.escape(str(page.get("action", "")).upper())}
          </div>
          <div style="color:#666;font-size:12px;margin:4px 0;">
            {prev}...
          </div>
          <div style="margin-top:6px;">{spans_html}</div>
          {section_warnings_html}
        </div>"""

    gemini_cards = ""
    for f in result.get("gemini_findings", []):
        t = html.escape(str(f.get("type", "")))
        tx = html.escape(str(f.get("text", ""))[:50])
        rs = html.escape(str(f.get("reason", "")))
        gemini_cards += f"""
        <div style="border:1px solid #e3f2fd;border-radius:8px;
                    padding:10px;margin:6px 0;
                    border-left:4px solid #1565C0;">
          <b style="color:#1565C0;">{t}</b>
          <span style="background:#e3f2fd;padding:1px 6px;
                       border-radius:3px;margin-left:8px;">
            {tx}
          </span>
          <div style="color:#888;font-size:11px;margin-top:4px;">
            {rs}
          </div>
        </div>"""

    file_stem = path.stem if path.name else "report"
    fname = html.escape(result["file_name"])
    scanned = html.escape(result["scanned_at"][:19].replace("T", " "))
    overall = html.escape(result["overall_risk"].upper())
    summary = html.escape(result["summary"])
    method_u = html.escape(str(result["method"]).upper())

    html_doc = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Sentinel Scan — {fname}</title>
<style>
  body{{font-family:-apple-system,sans-serif;
       max-width:800px;margin:0 auto;padding:24px;
       background:#f8f9fa;color:#1a1a1a;}}
  .header{{background:#1C1C1E;color:white;
           padding:20px;border-radius:12px;
           margin-bottom:20px;}}
  .badge{{display:inline-block;padding:4px 12px;
          border-radius:20px;font-weight:600;
          background:{risk_color};color:white;
          font-size:14px;}}
  .stat{{display:inline-block;margin:0 16px 0 0;}}
  .stat-val{{font-size:28px;font-weight:700;}}
  .stat-lbl{{font-size:11px;color:#999;}}
  h2{{color:#1C1C1E;margin:20px 0 8px;}}
</style>
</head><body>
<div class="header">
  <div style="font-size:20px;font-weight:700;
              margin-bottom:8px;">
    Sentinel Document Scan
  </div>
  <div style="color:#aaa;font-size:13px;">
    {fname} ·
    {scanned}
  </div>
</div>

<div style="background:white;padding:20px;
            border-radius:12px;margin-bottom:16px;">
  <span class="badge">{overall}</span>
  <span style="margin-left:12px;font-size:20px;
               font-weight:700;color:{risk_color};">
    Risk Score: {result['risk_score']}
  </span>
  <div style="margin-top:16px;">
    <div class="stat">
      <div class="stat-val">{result['page_count']}</div>
      <div class="stat-lbl">PAGES</div>
    </div>
    <div class="stat">
      <div class="stat-val">{result['issue_count']}</div>
      <div class="stat-lbl">ISSUES</div>
    </div>
    <div class="stat">
      <div class="stat-val">{method_u}</div>
      <div class="stat-lbl">METHOD</div>
    </div>
    <div class="stat">
      <div class="stat-val">
        {'✓' if result['gemini_used'] else '—'}
      </div>
      <div class="stat-lbl">GEMINI AI</div>
    </div>
  </div>
  <div style="margin-top:12px;color:#666;font-size:13px;">
    {summary}
  </div>
</div>

{f'<h2>Local model findings</h2>{cards}' if cards else ''}
{f'<h2>Gemini AI findings</h2>{gemini_cards}' if gemini_cards else ''}

{'<div style="background:#e8f5e9;padding:16px;border-radius:8px;color:#2E7D32;font-weight:600;">✓ No PII detected in this document</div>' if not cards and not gemini_cards else ''}

<div style="margin-top:32px;color:#aaa;
            font-size:11px;text-align:center;">
  Generated by Core Sentinel Guardrail
</div>
</body></html>"""

    out_dir = _GUARDRAIL_ROOT / "logs" / "scan_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"scan_{file_stem}_{ts}.html"
    out_path.write_text(html_doc, encoding="utf-8")
    return str(out_path)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        result = scan_document(sys.argv[1])
        print("Risk:", result.get("overall_risk"))
        print("Issues:", result.get("issue_count"))
        print("Report:", result.get("report_path"))
        rp = result.get("report_path")
        if rp:
            p = Path(rp)
            if p.is_file():
                webbrowser.open(p.as_uri())
    else:
        print("Usage: python document_scanner.py <file>")
