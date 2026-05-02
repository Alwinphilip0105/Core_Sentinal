"""
Full document scan: extract text, local PII scoring, SQLite + HTML report.
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

from infer import score_clipboard_with_pii  # noqa: E402
from risk_mapping import strong_regex_pii_spans  # noqa: E402
from sentinel_sync_daemon import is_placeholder_supabase_url  # noqa: E402
from text_extractor import extract  # noqa: E402

DB_PATH = _GUARDRAIL_ROOT / "logs" / "scan_results.db"

# Regex span classes that force block + high risk when found on full-page scan
_CRITICAL_REGEX_CLASSES = {
    "SSN pattern",
    "credit card pattern",
    "API key pattern",
    "JWT pattern",
    "IBAN pattern",
    "IBAN",
}

_ACTION_RANK = {"silent": 0, "warn": 1, "block": 2}
_RISK_RANK = {"low": 0, "med": 1, "high": 2}


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def _offset_spans(spans: list[dict], delta: int) -> list[dict]:
    out: list[dict] = []
    for s in spans:
        d = dict(s)
        try:
            d["start"] = int(s.get("start", 0)) + delta
            d["end"] = int(s.get("end", 0)) + delta
        except (TypeError, ValueError):
            continue
        out.append(d)
    return out


def _worse_action(a: str, b: str) -> str:
    aa = str(a).lower()
    bb = str(b).lower()
    ra = _ACTION_RANK.get(aa, 0)
    rb = _ACTION_RANK.get(bb, 0)
    return a if ra >= rb else b


def _higher_risk(a: str, b: str) -> str:
    ra = _RISK_RANK.get(str(a).lower(), 0)
    rb = _RISK_RANK.get(str(b).lower(), 0)
    return a if ra >= rb else b


def _worse_decision(a: str, b: str) -> str:
    """Prefer block > warn > allow for merged chunk decisions."""
    order = {"allow": 0, "warn": 1, "block": 2}
    aa = str(a).lower()
    bb = str(b).lower()
    return a if order.get(aa, 0) >= order.get(bb, 0) else b


def _iter_chunk_ranges(text: str, chunk_size: int = 500, overlap: int = 100):
    """Same windows as _chunk_text; yields (start_offset, chunk) for correct span mapping."""
    if len(text) <= chunk_size:
        yield 0, text
        return
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        yield start, text[start:end]
        start += chunk_size - overlap


def _merge_chunk_metadata(chunk_results: list[dict]) -> dict:
    if not chunk_results:
        return {
            "risk": "low",
            "pii_labels": [],
            "pii_risk_before_override": "low",
            "pii_override_applied": False,
            "decision": "allow",
            "block": False,
            "action": "silent",
            "message": "",
            "triggers": [],
            "risk_score": 0,
            "suggestions": [],
            "critical_secret_detected": False,
            "token_count": 0,
            "chunks_scored": 0,
            "text_truncated": False,
            "window_scores": [],
            "spans": [],
        }
    merged = dict(chunk_results[0])
    for r in chunk_results[1:]:
        merged["risk_score"] = max(int(merged.get("risk_score") or 0), int(r.get("risk_score") or 0))
        merged["risk"] = _higher_risk(str(merged.get("risk", "low")), str(r.get("risk", "low")))
        merged["action"] = _worse_action(str(merged.get("action", "silent")), str(r.get("action", "silent")))
        merged["critical_secret_detected"] = bool(
            merged.get("critical_secret_detected") or r.get("critical_secret_detected")
        )
        merged["chunks_scored"] = int(merged.get("chunks_scored") or 0) + int(r.get("chunks_scored") or 0)
        merged["token_count"] = max(
            int(merged.get("token_count") or 0),
            int(r.get("token_count") or 0),
        )
        merged["text_truncated"] = bool(merged.get("text_truncated") or r.get("text_truncated"))
        merged["pii_override_applied"] = bool(merged.get("pii_override_applied") or r.get("pii_override_applied"))
        merged["decision"] = _worse_decision(str(merged.get("decision", "allow")), str(r.get("decision", "allow")))
        merged["block"] = bool(merged.get("block") or r.get("block"))
    seen_t: set[str] = set()
    tr_union: list[str] = []
    for r in chunk_results:
        for t in r.get("triggers") or []:
            ts = str(t)
            if ts not in seen_t:
                seen_t.add(ts)
                tr_union.append(ts)
    merged["triggers"] = tr_union
    merged["pii_labels"] = merged.get("pii_labels") or []
    merged["suggestions"] = merged.get("suggestions") or []
    merged["window_scores"] = merged.get("window_scores") or []
    return merged


def _dedupe_spans(spans: list[dict]) -> list[dict]:
    seen: set[tuple[int, int, str]] = set()
    out: list[dict] = []
    for s in spans:
        try:
            st = int(s.get("start", 0))
            en = int(s.get("end", 0))
        except (TypeError, ValueError):
            continue
        key = (st, en, str(s.get("class", "")))
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def scan_page_text_local(page_text: str) -> dict:
    """
    Chunked model scoring + full-page strong-regex merge (document scan).
    """
    chunk_results: list[dict] = []
    merged_spans: list[dict] = []

    for start_off, chunk in _iter_chunk_ranges(page_text):
        if len(chunk.strip()) < 20:
            continue
        r = score_clipboard_with_pii(chunk, context="document")
        chunk_results.append(r)
        merged_spans.extend(_offset_spans(r.get("spans") or [], start_off))

    result = _merge_chunk_metadata(chunk_results)
    result["spans"] = _dedupe_spans(merged_spans)

    regex_spans = strong_regex_pii_spans(page_text)
    if regex_spans:
        result["spans"] = list(result.get("spans") or [])
        result["spans"].extend(regex_spans)
        result["spans"] = _dedupe_spans(result["spans"])

        found_critical = any(s.get("class") in _CRITICAL_REGEX_CLASSES for s in regex_spans)
        if found_critical:
            result["action"] = "block"
            result["risk"] = "high"
            if result.get("risk_score", 0) < 80:
                result["risk_score"] = 85
            result["risk_score"] = max(int(result.get("risk_score") or 0), 100)
            result["block"] = True
            result["decision"] = "block"

    return result


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
            report_path TEXT,
            findings    TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def scan_document(
    file_path: str,
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

            result = scan_page_text_local(page_text)
            result["page"] = page.get("page", i + 1)
            result["text_preview"] = page_text[:200]
            page_results.append(result)

        _progress(65, "Aggregating findings...")

        overall_risk = "low"
        if any(r.get("action") == "block" for r in page_results):
            overall_risk = "high"
        elif any(r.get("action") == "warn" for r in page_results):
            overall_risk = "med"

        flagged_pages = [r for r in page_results if r.get("action") in ("warn", "block")]

        all_classes = list(
            {s.get("class", "") for r in page_results for s in r.get("spans", [])}
        )
        all_classes = [c for c in all_classes if c]

        max_score = max((r.get("risk_score", 0) for r in page_results), default=0)

        total_span_issues = sum(len(r.get("spans", [])) for r in page_results)

        scan_result = {
            "file": str(path),
            "file_name": path.name,
            "file_size": extracted.get("file_size", 0),
            "page_count": extracted.get("page_count", 1),
            "method": extracted.get("method", "unknown"),
            "overall_risk": overall_risk,
            "risk_score": max_score,
            "issue_count": total_span_issues,
            "pii_classes": all_classes,
            "flagged_pages": flagged_pages,
            "scanned_at": datetime.now().isoformat(),
            "summary": (
                f"{total_span_issues} PII span(s); {len(flagged_pages)} of {len(pages)} page(s) flagged"
            ),
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
         issue_count, pii_classes,
         report_path, findings)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
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
    url = (os.environ.get("SUPABASE_URL") or "").strip()
    key = (os.environ.get("SUPABASE_ANON_KEY") or "").strip()
    if not url or not key:
        return
    if is_placeholder_supabase_url(url):
        return
    try:
        from supabase import create_client

        sb = create_client(url, key)
        sb.table("document_scans").insert(
            {
                "scanned_at": result["scanned_at"],
                "file_name": result["file_name"],
                "overall_risk": result["overall_risk"],
                "risk_score": int(result["risk_score"] or 0),
                "issue_count": int(result["issue_count"] or 0),
                "pii_classes": ",".join(result["pii_classes"]),
            }
        ).execute()
        print("[supabase] document_scans insert OK")
    except Exception as e:
        msg = str(e).lower()
        hint = ""
        if "document_scans" in msg or "pgrst205" in msg or "schema cache" in msg:
            hint = (
                " Create the table: run core-sentinel-guardrail/supabase/document_scans.sql "
                "in the Supabase SQL Editor."
            )
        print(f"[supabase] document_scans sync failed: {e}{hint}")


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
  </div>
  <div style="margin-top:12px;color:#666;font-size:13px;">
    {summary}
  </div>
</div>

{f'<h2>Local model findings</h2>{cards}' if cards else ''}

{'<div style="background:#e8f5e9;padding:16px;border-radius:8px;color:#2E7D32;font-weight:600;">✓ No PII detected in this document</div>' if not cards else ''}

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


def generate_paste_report(text: str, result: dict) -> Path:
    """
    Mini HTML report for a scored clipboard snapshot (background monitor / LLM context).
    Saved under logs/paste_reports/paste_{timestamp}.html
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = _GUARDRAIL_ROOT / "logs" / "paste_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    score = int(result.get("risk_score", 0) or 0)
    action = str(result.get("action", "silent"))
    spans = result.get("spans") if isinstance(result.get("spans"), list) else []

    spans_html = "".join(
        f"""
        <tr>
          <td>{html.escape(str(s.get("class", "?")))}</td>
          <td><code>{html.escape(str(s.get("match", ""))[:50])}</code></td>
          <td>{html.escape(str(s.get("risk", "?")))}</td>
          <td>{html.escape(str(s.get("source", "?")))}</td>
        </tr>"""
        for s in spans
        if isinstance(s, dict)
    )

    color = "#E53935" if score >= 70 else "#FFB300" if score >= 40 else "#43A047"

    preview = html.escape((text or "")[:200])
    if len(text or "") > 200:
        preview += "…"

    spans_block = (
        "<table><tr><th>Class</th><th>Match</th><th>Risk</th><th>Source</th></tr>"
        + spans_html
        + "</table>"
        if spans
        else '<p style="color:#666">No specific PII spans extracted — contextual detection only.</p>'
    )

    html_doc = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Paste Report {ts}</title>
<style>
  body{{font-family:sans-serif;padding:24px;
       background:#f9fafb;color:#333}}
  .header{{background:{color};color:white;
           padding:16px 20px;border-radius:8px;
           margin-bottom:20px}}
  table{{width:100%;border-collapse:collapse}}
  th{{background:#f0f0f0;padding:8px;
      text-align:left;font-size:12px}}
  td{{padding:8px;border-bottom:1px solid #eee;
      font-size:12px}}
  code{{background:#f5f5f5;padding:2px 4px;
        border-radius:3px;font-size:11px}}
</style></head><body>
<div class="header">
  <h2 style="margin:0">Sentinel Paste Report</h2>
  <p style="margin:4px 0 0;opacity:.85">
    {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    · Score: {score}/100 · Action: {action.upper()}
  </p>
</div>
<p><strong>Text preview:</strong>
   <code>{preview}</code>
</p>
<h3>Detected spans ({len(spans)})</h3>
{spans_block}
</body></html>"""

    report_path = report_dir / f"paste_{ts}.html"
    report_path.write_text(html_doc, encoding="utf-8")
    print(f"[report] saved: {report_path}", flush=True)
    return report_path


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
