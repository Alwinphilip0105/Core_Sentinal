"""
Core Sentinel — Performance & UX Audit
Run from core-sentinel-guardrail/
Usage: ..\.venv\Scripts\python.exe performance_audit.py
"""

import sys, time, json, statistics, traceback
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

@dataclass
class PerfResult:
    name: str
    category: str
    duration_ms: float
    passed: bool
    target_ms: float
    actual_value: str = ""
    suggestion: str = ""
    severity: str = "info"  # info / warn / critical

def bench(fn, runs=5):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn()
        times.append((time.perf_counter()-t0)*1000)
    return statistics.median(times), result

def load_scorer():
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from infer import score_clipboard_with_pii
        return score_clipboard_with_pii
    except Exception as e:
        print(f"{RED}Cannot import infer.py: {e}{RESET}")
        sys.exit(1)

def run_perf_tests(scorer) -> List[PerfResult]:
    results = []

    # ── INFERENCE SPEED ──────────────────────────────────────────
    cases = [
        ("Safe text (short)",      "The weather is sunny today", 150),
        ("Safe text (long)",       "Machine learning is a field of AI. " * 30, 300),
        ("Email address",          "Contact john@company.com for info", 150),
        ("Phone number",           "Call 212-555-0134 anytime", 150),
        ("SSN pattern",            "My SSN is 123-45-6789", 100),
        ("Credit card",            "Card: 4532-1234-5678-9012 exp 09/27", 100),
        ("API key",                "API_KEY=sk-abc123XYZsecretkey9999", 50),
        ("JWT token",              "Bearer eyJhbGciOiJIUzI1NiJ9.abc.xyz", 50),
        ("AWS credentials",        "AKIAIOSFODNN7EXAMPLE wJalrXUtnFEMI", 50),
        ("Full PII combo",         "John Smith SSN 123-45-6789 card 4532-1234-5678-9012", 150),
        ("Empty text",             "   ", 20),
        ("Unicode text",           "こんにちは、私の名前は田中です。今日は天気がいいですね。", 200),
        ("Very long text",         "This is safe business content. " * 100, 500),
        ("Repeated scan (cache)",  "The weather is sunny today", 20),
    ]

    print(f"\n{BOLD}Inference speed{RESET}")
    for name, text, target in cases:
        try:
            ms, res = bench(lambda t=text: scorer(t), runs=3)
            action = res.get('action','?')
            score  = res.get('risk_score', 0)
            passed = ms <= target
            sev    = "info" if passed else ("warn" if ms < target*2 else "critical")
            icon   = f"{GREEN}✓{RESET}" if passed else f"{RED}✗{RESET}"
            print(f"  {icon} {name:<35} {ms:6.0f}ms  (target {target}ms)  {action}/{score}")
            results.append(PerfResult(
                name=name, category="Inference speed",
                duration_ms=ms, passed=passed,
                target_ms=target,
                actual_value=f"{ms:.0f}ms / action={action} score={score}",
                suggestion=f"Target {target}ms — consider caching or lighter model" if not passed else "",
                severity=sev
            ))
        except Exception as e:
            print(f"  {RED}✗ {name}: {e}{RESET}")

    # ── STARTUP / IMPORT TIME ─────────────────────────────────────
    print(f"\n{BOLD}Module import times{RESET}")
    modules = [
        ("infer",            "from infer import score_clipboard_with_pii", 2000),
        ("risk_mapping",     "import risk_mapping", 500),
        ("active_window_llm","from active_window_llm import is_llm_window", 300),
        ("feedback_store",   "from feedback_store import FeedbackStore", 300),
    ]
    for name, stmt, target in modules:
        try:
            t0 = time.perf_counter()
            exec(stmt)
            ms = (time.perf_counter()-t0)*1000
            passed = ms <= target
            icon = f"{GREEN}✓{RESET}" if passed else f"{YELLOW}!{RESET}"
            print(f"  {icon} {name:<25} {ms:6.0f}ms  (target {target}ms)")
            results.append(PerfResult(
                name=f"Import {name}", category="Startup time",
                duration_ms=ms, passed=passed, target_ms=target,
                actual_value=f"{ms:.0f}ms",
                suggestion="Heavy import — consider lazy loading" if not passed else "",
                severity="warn" if not passed else "info"
            ))
        except Exception as e:
            print(f"  {RED}✗ {name}: {e}{RESET}")

    # ── REGEX LAYER SPEED ─────────────────────────────────────────
    print(f"\n{BOLD}Regex layer speed{RESET}")
    try:
        from risk_mapping import strong_regex_pii_spans
        regex_cases = [
            ("SSN",         "SSN: 123-45-6789", 5),
            ("Credit card", "4532-1234-5678-9012", 5),
            ("API key",     "sk-abc123XYZsecretkey", 5),
            ("JWT",         "eyJhbGciOiJIUzI1NiJ9.abc.xyz", 5),
            ("Safe text",   "Hello world how are you", 5),
            ("Long safe",   "Hello world. " * 200, 20),
        ]
        for name, text, target in regex_cases:
            ms, spans = bench(
                lambda t=text: strong_regex_pii_spans(t), runs=10)
            passed = ms <= target
            icon = f"{GREEN}✓{RESET}" if passed else f"{RED}✗{RESET}"
            print(f"  {icon} {name:<25} {ms:5.2f}ms  {len(spans)} spans")
            results.append(PerfResult(
                name=f"Regex: {name}", category="Regex speed",
                duration_ms=ms, passed=passed, target_ms=target,
                actual_value=f"{ms:.2f}ms / {len(spans)} spans",
                suggestion="Regex too slow — check for catastrophic backtracking" if not passed else "",
                severity="critical" if not passed else "info"
            ))
    except Exception as e:
        print(f"  {RED}Cannot test regex: {e}{RESET}")

    # ── THROUGHPUT ────────────────────────────────────────────────
    print(f"\n{BOLD}Throughput test{RESET}")
    texts = [
        "The weather is nice today",
        "Contact me at john@example.com",
        "My SSN is 123-45-6789",
        "API_KEY=sk-abc123secret",
        "Call 212-555-0134 for info",
    ] * 4  # 20 pastes
    t0 = time.perf_counter()
    for t in texts:
        scorer(t)
    total = (time.perf_counter()-t0)*1000
    per_paste = total/len(texts)
    passed = per_paste < 200
    icon = f"{GREEN}✓{RESET}" if passed else f"{RED}✗{RESET}"
    print(f"  {icon} {len(texts)} pastes in {total:.0f}ms = {per_paste:.0f}ms/paste")
    results.append(PerfResult(
        name="Throughput (20 pastes)", category="Throughput",
        duration_ms=per_paste, passed=passed, target_ms=200,
        actual_value=f"{per_paste:.0f}ms avg / {total:.0f}ms total",
        suggestion="Too slow for real-time use — add result caching" if not passed else "",
        severity="critical" if not passed else "info"
    ))

    # ── DATA / STORAGE ────────────────────────────────────────────
    print(f"\n{BOLD}Storage health{RESET}")
    checks = [
        ("logs/guardrail.db",              "SQLite events DB",    50*1024*1024),
        ("logs/feedback_store.jsonl",       "Feedback store",      10*1024*1024),
        ("config/user_settings.json",       "User settings",       100*1024),
        ("config/risk_policy.json",         "Risk policy",         100*1024),
        ("models/tinybert_guardrail",       "Model directory",     500*1024*1024),
        ("logs/supabase_sync_state.json",   "Sync state",          10*1024),
    ]
    for path_str, label, max_bytes in checks:
        p = Path(path_str)
        if p.exists():
            size = sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) \
                   if p.is_dir() else p.stat().st_size
            size_mb = size/1024/1024
            passed = size <= max_bytes
            icon = f"{GREEN}✓{RESET}" if passed else f"{YELLOW}!{RESET}"
            print(f"  {icon} {label:<30} {size_mb:.1f} MB")
            results.append(PerfResult(
                name=f"Size: {label}", category="Storage",
                duration_ms=0, passed=passed, target_ms=0,
                actual_value=f"{size_mb:.1f} MB",
                suggestion=f"File growing large — consider rotating logs" if not passed else "",
                severity="warn" if not passed else "info"
            ))
        else:
            print(f"  {YELLOW}? {label:<30} not found{RESET}")

    # ── FEEDBACK STATS ────────────────────────────────────────────
    print(f"\n{BOLD}Feedback & learning stats{RESET}")
    fb = Path('logs/feedback_store.jsonl')
    if fb.exists():
        total_fb = wrong_fb = used_fb = 0
        corrections = {}
        with open(fb) as f:
            for line in f:
                try:
                    e = json.loads(line.strip())
                    total_fb += 1
                    if e.get('feedback_type') == 'wrong':
                        wrong_fb += 1
                        k = f"{e.get('predicted','?')}→{e.get('correct','?')}"
                        corrections[k] = corrections.get(k,0)+1
                    if e.get('used_for_training'):
                        used_fb += 1
                except: pass
        pending = wrong_fb - used_fb
        print(f"  Total feedback entries: {total_fb}")
        print(f"  Wrong corrections:      {wrong_fb}")
        print(f"  Used for training:      {used_fb}")
        print(f"  Pending for retrain:    {pending}")
        if corrections:
            print(f"  Top corrections:")
            for k,v in sorted(corrections.items(),
                               key=lambda x:-x[1])[:5]:
                print(f"    {k}: {v}x")
        results.append(PerfResult(
            name="Pending corrections", category="Learning",
            duration_ms=0, passed=pending<30, target_ms=30,
            actual_value=f"{pending} pending",
            suggestion="Run train.py to retrain model" if pending>=30 else "",
            severity="warn" if pending>=30 else "info"
        ))
    else:
        print(f"  {YELLOW}No feedback data yet{RESET}")

    # ── SUPABASE SYNC ─────────────────────────────────────────────
    print(f"\n{BOLD}Supabase sync status{RESET}")
    sync_state = Path('logs/supabase_sync_state.json')
    if sync_state.exists():
        try:
            state = json.loads(sync_state.read_text())
            last_id   = state.get('last_id', 0)
            last_sync = state.get('last_sync', 'never')
            print(f"  Last synced ID:  {last_id}")
            print(f"  Last sync time: {last_sync}")
            # check how stale
            try:
                from datetime import datetime, timezone
                ls = datetime.fromisoformat(
                    last_sync.replace('Z',''))
                age_mins = (datetime.utcnow()-ls
                            ).total_seconds()/60
                stale = age_mins > 10
                icon = f"{YELLOW}!{RESET}" if stale else f"{GREEN}✓{RESET}"
                print(f"  {icon} Sync age: {age_mins:.0f} minutes")
                results.append(PerfResult(
                    name="Supabase sync freshness",
                    category="Cloud sync",
                    duration_ms=age_mins*60*1000,
                    passed=not stale,
                    target_ms=10*60*1000,
                    actual_value=f"{age_mins:.0f} mins ago",
                    suggestion="Sync is stale — check daemon thread" if stale else "",
                    severity="warn" if stale else "info"
                ))
            except: pass
        except Exception as e:
            print(f"  {RED}Error reading sync state: {e}{RESET}")
    else:
        print(f"  {YELLOW}No sync state file — daemon may not be running{RESET}")

    return results

def generate_html(results: List[PerfResult]) -> str:
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # UX recommendations based on results
    ux_issues = []

    slow_inference = [r for r in results
                      if r.category == "Inference speed"
                      and not r.passed]
    if slow_inference:
        worst = max(slow_inference, key=lambda r: r.duration_ms)
        ux_issues.append({
            "severity": "critical",
            "title": f"Slow inference: {worst.name} takes {worst.duration_ms:.0f}ms",
            "impact": "User presses Ctrl+V and waits over 1 second with no feedback. Feels broken.",
            "fix": "Add result caching for recently-seen text. Show 'Scanning...' spinner immediately on Ctrl+V press.",
            "effort": "Low"
        })

    if any(r.name == "Repeated scan (cache)" and r.duration_ms > 30
           for r in results):
        ux_issues.append({
            "severity": "high",
            "title": "No caching — repeated text re-scored every time",
            "impact": "Copying the same text multiple times causes unnecessary 60-1600ms delays.",
            "fix": "LRU cache keyed by MD5 hash, 60s TTL, max 50 entries.",
            "effort": "Low"
        })

    ux_issues.extend([
        {
            "severity": "high",
            "title": "No immediate feedback on Ctrl+V press",
            "impact": "Silent gap between keypress and result destroys perceived responsiveness.",
            "fix": "Show amber pulsing traffic light + 'Scanning...' text within 5ms of Ctrl+V.",
            "effort": "Low"
        },
        {
            "severity": "medium",
            "title": "Panel opens but score ring animates from 0",
            "impact": "Score ring fills slowly which is distracting when urgent.",
            "fix": "Animate score ring over 300ms with easeOut — fast enough to feel snappy.",
            "effort": "Low"
        },
        {
            "severity": "medium",
            "title": "Toast notifications stack over each other",
            "impact": "Multiple pastes in quick succession cause toasts to overlap.",
            "fix": "Stagger toasts 8px upward, max 3 visible, dismiss oldest when 4th arrives.",
            "effort": "Low"
        },
        {
            "severity": "medium",
            "title": "Panel slides in from right but takes 200ms",
            "impact": "High risk paste — user waits 200ms to see what was detected.",
            "fix": "Reduce slide-in to 120ms with easeOut. Speed = urgency.",
            "effort": "Low"
        },
        {
            "severity": "low",
            "title": "Traffic light too small to read at a glance",
            "impact": "User has to focus on the overlay to see the risk level.",
            "fix": "Active/glowing dot should be at least 12px with 6px outer glow ring.",
            "effort": "Low"
        },
        {
            "severity": "low",
            "title": "Overlay opacity 0.55 when idle makes it hard to notice",
            "impact": "When clipboard has high-risk content, dim overlay is easy to miss.",
            "fix": "When clipboard preview is red, raise opacity to 0.85 even in idle state.",
            "effort": "Low"
        },
        {
            "severity": "low",
            "title": "No sound/haptic feedback for critical blocks",
            "impact": "Silent block — user may not realise paste was stopped.",
            "fix": "Play a short system sound (winsound.Beep) for critical blocks only.",
            "effort": "Low"
        }
    ])

    sev_order = {"critical":0,"high":1,"medium":2,"low":3,"info":4}
    ux_issues.sort(key=lambda x: sev_order.get(x["severity"],4))

    sev_color = {
        "critical": ("#FFEBEE","#B71C1C","#E53935"),
        "high":     ("#FFF8E1","#E65100","#FFB300"),
        "medium":   ("#E3F2FD","#0D47A1","#1565C0"),
        "low":      ("#E8F5E9","#1B5E20","#43A047"),
    }

    ux_html = ""
    for issue in ux_issues:
        bg, tc, bc = sev_color.get(
            issue["severity"], ("#F5F5F5","#333","#999"))
        ux_html += f"""
        <div style="border-left:4px solid {bc};
                    background:{bg};border-radius:6px;
                    padding:12px 14px;margin-bottom:10px">
          <div style="display:flex;align-items:center;
                      gap:8px;margin-bottom:4px">
            <span style="background:{bc};color:white;
                         padding:2px 8px;border-radius:10px;
                         font-size:11px;font-weight:700">
              {issue['severity'].upper()}
            </span>
            <strong style="font-size:13px;color:{tc}">
              {issue['title']}
            </strong>
            <span style="margin-left:auto;font-size:11px;
                         color:#888">effort: {issue['effort']}</span>
          </div>
          <div style="font-size:12px;color:#555;margin-bottom:4px">
            <strong>Impact:</strong> {issue['impact']}
          </div>
          <div style="font-size:12px;color:{tc}">
            <strong>Fix:</strong> {issue['fix']}
          </div>
        </div>"""

    # perf table rows
    perf_rows = ""
    for r in results:
        if r.duration_ms == 0 and r.category not in [
                "Inference speed","Throughput","Startup time",
                "Regex speed","Cloud sync"]:
            continue
        passed = r.passed
        bg = "#F9FBF9" if passed else "#FFF5F5"
        icon = "✓" if passed else "✗"
        ic = "#2E7D32" if passed else "#C62828"
        perf_rows += f"""
        <tr style="background:{bg}">
          <td style="text-align:center;color:{ic};font-size:14px">{icon}</td>
          <td style="font-size:12px">{r.name}</td>
          <td style="font-size:11px;color:#888">{r.category}</td>
          <td style="font-size:12px;font-weight:600;
                     color:{'#C62828' if not passed else '#2E7D32'}">
            {r.actual_value}
          </td>
          <td style="font-size:11px;color:#666">{r.suggestion}</td>
        </tr>"""

    passed_count = sum(1 for r in results if r.passed)
    total = len(results)

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Core Sentinel — Performance & UX Audit</title>
<style>
  body{{font-family:-apple-system,BlinkMacSystemFont,
       'Segoe UI',sans-serif;background:#F9FAFB;
       margin:0;color:#1a1a1a}}
  .hdr{{background:linear-gradient(135deg,#1C2B4B,#0D47A1);
        padding:28px 36px;color:white}}
  .hdr h1{{margin:0;font-size:22px;font-weight:700}}
  .hdr p{{margin:4px 0 0;opacity:.7;font-size:13px}}
  .body{{max-width:1100px;margin:24px auto;padding:0 24px}}
  .stat-row{{display:grid;grid-template-columns:repeat(4,1fr);
             gap:12px;margin-bottom:24px}}
  .stat{{background:white;border-radius:10px;padding:16px;
         border:1px solid #E5E7EB;text-align:center}}
  .stat-num{{font-size:28px;font-weight:700;margin-bottom:2px}}
  .stat-lbl{{font-size:11px;color:#888;text-transform:uppercase;
             letter-spacing:.05em}}
  .card{{background:white;border-radius:10px;padding:20px 24px;
         border:1px solid #E5E7EB;margin-bottom:20px}}
  .card h2{{margin:0 0 16px;font-size:16px;color:#1C2B4B;
            display:flex;align-items:center;gap:8px}}
  table{{width:100%;border-collapse:collapse}}
  th{{background:#F8F9FA;padding:8px 10px;text-align:left;
      font-size:11px;color:#666;font-weight:600;
      text-transform:uppercase;letter-spacing:.04em}}
  td{{padding:8px 10px;border-bottom:1px solid #F0F0F0;
      vertical-align:top}}
  .perf-bar{{height:8px;border-radius:4px;
             background:#E5E7EB;overflow:hidden;
             margin-top:4px}}
  .perf-fill{{height:100%;border-radius:4px}}
</style>
</head><body>
<div class="hdr">
  <h1>🛡 Core Sentinel — Performance & UX Audit</h1>
  <p>Generated {now} · {total} checks · 
     {passed_count} passed · {total-passed_count} issues</p>
</div>
<div class="body">

  <div class="stat-row">
    <div class="stat">
      <div class="stat-num" style="color:#2E7D32">
        {passed_count}</div>
      <div class="stat-lbl">Checks passed</div>
    </div>
    <div class="stat">
      <div class="stat-num" style="color:#C62828">
        {total-passed_count}</div>
      <div class="stat-lbl">Issues found</div>
    </div>
    <div class="stat">
      <div class="stat-num" style="color:#1565C0">
        {len([i for i in ux_issues if i['severity'] in ('critical','high')])}</div>
      <div class="stat-lbl">Critical UX issues</div>
    </div>
    <div class="stat">
      <div class="stat-num" style="color:#6A1B9A">
        {len(ux_issues)}</div>
      <div class="stat-lbl">UX improvements</div>
    </div>
  </div>

  <div class="card">
    <h2>⚡ UX improvements — ordered by impact</h2>
    {ux_html}
  </div>

  <div class="card">
    <h2>📊 Performance measurements</h2>
    <table>
      <thead>
        <tr>
          <th style="width:30px"></th>
          <th>Test</th>
          <th>Category</th>
          <th>Result</th>
          <th>Suggestion</th>
        </tr>
      </thead>
      <tbody>{perf_rows}</tbody>
    </table>
  </div>

  <div class="card">
    <h2>🎨 UI quality assessment</h2>
    <table>
      <thead>
        <tr>
          <th>Component</th>
          <th>Current state</th>
          <th>Rating</th>
          <th>Improvement</th>
        </tr>
      </thead>
      <tbody>
        <tr style="background:#F9FBF9">
          <td style="font-weight:600;font-size:12px">Traffic light</td>
          <td style="font-size:12px">Attached to pill, 3 dots, color changes per risk</td>
          <td><span style="background:#E8F5E9;color:#1B5E20;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:700">Good</span></td>
          <td style="font-size:11px;color:#555">Make active dot larger (12px) with stronger glow</td>
        </tr>
        <tr>
          <td style="font-weight:600;font-size:12px">Pill overlay</td>
          <td style="font-size:12px">Expands on hover, shows LLM name + action + streak</td>
          <td><span style="background:#E8F5E9;color:#1B5E20;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:700">Good</span></td>
          <td style="font-size:11px;color:#555">Opacity 0.55 idle is too dim — raise to 0.75</td>
        </tr>
        <tr style="background:#F9FBF9">
          <td style="font-weight:600;font-size:12px">Score ring</td>
          <td style="font-size:12px">Circular arc fills 0-100, color by risk level</td>
          <td><span style="background:#E8F5E9;color:#1B5E20;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:700">Good</span></td>
          <td style="font-size:11px;color:#555">Add easeOut animation on fill for polish</td>
        </tr>
        <tr>
          <td style="font-weight:600;font-size:12px">Side panel</td>
          <td style="font-size:12px">Slides in from right, shows issue cards, Fix/Skip buttons</td>
          <td><span style="background:#FFF8E1;color:#E65100;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:700">Fair</span></td>
          <td style="font-size:11px;color:#555">200ms slide is too slow — reduce to 120ms</td>
        </tr>
        <tr style="background:#F9FBF9">
          <td style="font-weight:600;font-size:12px">Toast notifications</td>
          <td style="font-size:12px">Dark card, colored icon, progress bar, auto-dismiss</td>
          <td><span style="background:#E8F5E9;color:#1B5E20;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:700">Good</span></td>
          <td style="font-size:11px;color:#555">Add stacking — multiple toasts push upward</td>
        </tr>
        <tr>
          <td style="font-weight:600;font-size:12px">Character feedback</td>
          <td style="font-size:12px">Emoji + speech bubble + streak counter near overlay</td>
          <td><span style="background:#E8F5E9;color:#1B5E20;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:700">Good</span></td>
          <td style="font-size:11px;color:#555">Show every 3rd safe paste, always on milestones</td>
        </tr>
        <tr style="background:#F9FBF9">
          <td style="font-weight:600;font-size:12px">Analysing state</td>
          <td style="font-size:12px">No feedback during 60-1600ms scoring gap</td>
          <td><span style="background:#FFEBEE;color:#B71C1C;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:700">Poor</span></td>
          <td style="font-size:11px;color:#555">Show spinner + Scanning text within 5ms of Ctrl+V</td>
        </tr>
        <tr>
          <td style="font-weight:600;font-size:12px">Feedback buttons</td>
          <td style="font-size:12px">Correct/Wrong on panel, correction picker inline</td>
          <td><span style="background:#E8F5E9;color:#1B5E20;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:700">Good</span></td>
          <td style="font-size:11px;color:#555">Also show on overlay for 8s after detection</td>
        </tr>
        <tr style="background:#F9FBF9">
          <td style="font-weight:600;font-size:12px">Toolbar</td>
          <td style="font-size:12px">6 buttons slide out on hover from left of pill</td>
          <td><span style="background:#FFF8E1;color:#E65100;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:700">Fair</span></td>
          <td style="font-size:11px;color:#555">Add keyboard shortcut labels next to each button</td>
        </tr>
        <tr>
          <td style="font-weight:600;font-size:12px">History chart</td>
          <td style="font-size:12px">7 bars below pill on hover, color by action</td>
          <td><span style="background:#E8F5E9;color:#1B5E20;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:700">Good</span></td>
          <td style="font-size:11px;color:#555">Add hover tooltip showing exact score per bar</td>
        </tr>
      </tbody>
    </table>
  </div>

</div></body></html>"""

def main():
    print(f"\n{BOLD}{CYAN}Core Sentinel — Performance & UX Audit{RESET}")
    print(f"{'─'*55}")
    print(f"Loading scorer...", end=' ', flush=True)
    scorer = load_scorer()
    print(f"{GREEN}OK{RESET}")

    results = run_perf_tests(scorer)

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed

    print(f"\n{'─'*55}")
    print(f"{BOLD}Summary: {GREEN}{passed} passed{RESET} / "
          f"{RED}{failed} issues{RESET}{RESET}")

    # save report
    report = Path("logs/perf_audit.html")
    report.parent.mkdir(exist_ok=True)
    report.write_text(
        generate_html(results), encoding='utf-8')
    print(f"\n{GREEN}Report saved:{RESET} {report.resolve()}")
    print("Open in browser for full UX recommendations.\n")

if __name__ == "__main__":
    main()
