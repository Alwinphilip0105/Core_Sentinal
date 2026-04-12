"""
Core Sentinel — Automated Test Suite
Tests all detection flows and generates an HTML report.
Run from: core-sentinel-guardrail/
Usage: ..\.venv\Scripts\python.exe test_sentinel.py
"""

import sys
import time
import json
import hashlib
import traceback
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

# ── colour codes for terminal ──────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

@dataclass
class TestResult:
    name:        str
    category:    str
    input_text:  str
    expected_action: str
    expected_score_min: int
    expected_score_max: int
    actual_action:  str  = ""
    actual_score:   int  = 0
    actual_spans:   list = field(default_factory=list)
    passed:         bool = False
    error:          str  = ""
    duration_ms:    float = 0.0
    suggestion:     str  = ""

# ── test cases ──────────────────────────────────────────────────────────────
TEST_CASES = [

    # ── CRITICAL ────────────────────────────────────────────────────────────
    dict(name="API key (sk- prefix)", category="Critical",
         text="API_KEY=sk-abc123XYZsecretkey9999",
         action="block", min_score=85, max_score=100,
         suggestion="Regex pattern for sk- prefix should always catch this"),

    dict(name="AWS credentials", category="Critical",
         text="AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE\nAWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
         action="block", min_score=85, max_score=100,
         suggestion="AKIA prefix + high entropy secret should both fire"),

    dict(name="DB password", category="Critical",
         text="DB_PASSWORD=MyP@ssw0rd123!\nDB_HOST=prod.cluster.mongodb.net",
         action="block", min_score=85, max_score=100,
         suggestion="password= pattern should be caught by regex"),

    dict(name="JWT Bearer token", category="Critical",
         text="Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
         action="block", min_score=85, max_score=100,
         suggestion="eyJ prefix + 3 base64 parts should always block"),

    dict(name="GitHub token", category="Critical",
         text="GITHUB_TOKEN=ghp_abc123XYZgithubpersonaltoken9999",
         action="block", min_score=85, max_score=100,
         suggestion="ghp_ prefix is a strong signal"),

    dict(name="SSH private key", category="Critical",
         text="-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA2a2rwplBQLzaaaaaaaaa\n-----END RSA PRIVATE KEY-----",
         action="block", min_score=85, max_score=100,
         suggestion="BEGIN RSA PRIVATE KEY header is unambiguous"),

    dict(name="MongoDB connection string", category="Critical",
         text="mongodb://admin:Sup3rS3cr3t@cluster.mongodb.net/production",
         action="block", min_score=85, max_score=100,
         suggestion="mongodb:// protocol with credentials should block"),

    dict(name="Stripe live key", category="Critical",
         text="STRIPE_SECRET_KEY=sk_live_abc123XYZsecretkey",
         action="block", min_score=85, max_score=100,
         suggestion="sk_live_ prefix should be caught"),

    # ── HIGH ────────────────────────────────────────────────────────────────
    dict(name="SSN alone", category="High",
         text="My SSN is 123-45-6789",
         action="block", min_score=70, max_score=100,
         suggestion="SSN regex \\d{3}-\\d{2}-\\d{4} should always fire"),

    dict(name="Credit card Visa", category="High",
         text="Please charge card 4532-1234-5678-9012 expiry 09/27",
         action="block", min_score=70, max_score=100,
         suggestion="Luhn-valid Visa number should always block"),

    dict(name="Credit card + CVV", category="High",
         text="Card: 5425-2334-3010-9903 CVV: 874 expiry 11/26",
         action="block", min_score=85, max_score=100,
         suggestion="Card + CVV combination should be critical"),

    dict(name="SSN + DOB combo", category="High",
         text="Full name: James Wilson\nSSN: 456-78-9012\nDate of birth: 07/22/1988",
         action="block", min_score=80, max_score=100,
         suggestion="Name + SSN + DOB combination escalates to critical"),

    dict(name="Passport number", category="High",
         text="Passport: US B98765432 expires 2028",
         action="block", min_score=65, max_score=100,
         suggestion="Passport pattern [A-Z]{1,2}\\d{6,9} should fire"),

    dict(name="Medical record", category="High",
         text="Patient MRN: 1234567\nDiagnosis: F32.1 major depressive disorder\nInsurance: Aetna HIC-123456789",
         action="block", min_score=70, max_score=100,
         suggestion="MRN + ICD code combination should block"),

    dict(name="IBAN", category="High",
         text="Please transfer to IBAN: GB29NWBK60161331926819",
         action="block", min_score=70, max_score=100,
         suggestion="IBAN regex [A-Z]{2}\\d{2}[A-Z0-9]{4,30} should fire"),

    dict(name="Bank account + routing", category="High",
         text="Account: 12345678901\nRouting: 021000021",
         action="block", min_score=65, max_score=100,
         suggestion="Bank account + routing number combo should block"),

    dict(name="Full PII combo", category="High",
         text="John Smith DOB 04/15/1990\nemail: john@gmail.com\nphone: 212-555-0134\nSSN: 123-45-6789\ncard: 4532-9876-5432-1098",
         action="block", min_score=90, max_score=100,
         suggestion="Multiple PII types should push to critical"),

    # ── MEDIUM ──────────────────────────────────────────────────────────────
    dict(name="Email address", category="Medium",
         text="Please contact sarah.johnson@gmail.com for details",
         action="warn", min_score=40, max_score=75,
         suggestion="Single email should warn not block"),

    dict(name="Phone number US", category="Medium",
         text="Call me on 646-555-0187 after 3pm",
         action="warn", min_score=40, max_score=75,
         suggestion="US phone number should warn"),

    dict(name="Email + phone", category="Medium",
         text="Contact Mike at mike.brown@outlook.com or 415-555-0192",
         action="warn", min_score=50, max_score=80,
         suggestion="Email + phone combo should warn with higher score"),

    dict(name="Street address", category="Medium",
         text="My address is 456 Oak Avenue Brooklyn NY 11201",
         action="warn", min_score=40, max_score=75,
         suggestion="Street address should warn"),

    dict(name="IPv4 address", category="Medium",
         text="Server IP is 192.168.1.100 port 8080",
         action="warn", min_score=30, max_score=70,
         suggestion="IP address should warn"),

    dict(name="International phone", category="Medium",
         text="Call +44 7911 123456 for the London office",
         action="warn", min_score=35, max_score=70,
         suggestion="International phone should warn"),

    # ── LOW / SAFE ───────────────────────────────────────────────────────────
    dict(name="Generic safe text", category="Safe",
         text="The weather in New York today is partly cloudy with a high of 72 degrees",
         action="silent", min_score=0, max_score=35,
         suggestion="Generic text should always be silent"),

    dict(name="Code question", category="Safe",
         text="Please help me write a Python function that sorts a list of integers",
         action="silent", min_score=0, max_score=35,
         suggestion="Code questions should always be silent"),

    dict(name="Business report", category="Safe",
         text="The quarterly report shows strong revenue growth across all divisions this year",
         action="silent", min_score=0, max_score=35,
         suggestion="Business text should be silent"),

    dict(name="Meeting note", category="Safe",
         text="Team meeting scheduled for Tuesday at 3pm in conference room B",
         action="silent", min_score=0, max_score=35,
         suggestion="Calendar text should be silent"),

    dict(name="Technical docs", category="Safe",
         text="The API endpoint returns a JSON response with status code 200",
         action="silent", min_score=0, max_score=35,
         suggestion="API documentation should be silent"),

    # ── EDGE CASES ───────────────────────────────────────────────────────────
    dict(name="Multilingual with PII", category="Edge",
         text="Hola me llamo Carlos\nmi tarjeta es 4532-1234-5678-9012\ntelefono +34 612 345 678",
         action="block", min_score=70, max_score=100,
         suggestion="Non-English text with PII should still be caught by regex"),

    dict(name="Fake/example SSN", category="Edge",
         text="For example, SSN format is 123-45-6789 (do not use real SSNs)",
         action="block", min_score=60, max_score=100,
         suggestion="Even example SSNs should be caught — model cannot verify intent"),

    dict(name="Partial card number", category="Edge",
         text="Last 4 digits of card: 9012",
         action="silent", min_score=0, max_score=50,
         suggestion="Partial card number alone should not block"),

    dict(name="Empty text", category="Edge",
         text="   ",
         action="silent", min_score=0, max_score=10,
         suggestion="Empty/whitespace should always be silent"),

    dict(name="Very long safe text", category="Edge",
         text="This is a very long document about machine learning concepts. " * 20,
         action="silent", min_score=0, max_score=40,
         suggestion="Long safe text should not false-positive"),

    dict(name="Corporate internal", category="Edge",
         text="Employee ID: EMP-123456\nSalary: $125,000\nProject: PRJ-ENG-2024\nPerformance: 4.8/5.0 CONFIDENTIAL",
         action="warn", min_score=40, max_score=85,
         suggestion="Corporate PII should at least warn"),

    # ── 30 ADDITIONAL SAFE CASES (real-world content) ──────────
    dict(name="News article", category="Safe",
         text="The Federal Reserve announced it would hold interest rates steady at its meeting on Wednesday citing continued economic uncertainty.",
         action="silent", min_score=0, max_score=35,
         suggestion="News text should never trigger detection"),

    dict(name="Recipe text", category="Safe",
         text="Preheat oven to 375F. Mix 2 cups flour with 1 tsp baking soda. Add 2 eggs and 1 cup butter. Bake for 25 minutes until golden.",
         action="silent", min_score=0, max_score=35,
         suggestion="Recipe text should be silent"),

    dict(name="Sports scores", category="Safe",
         text="The Lakers defeated the Celtics 112-98 last night. LeBron James scored 34 points with 9 assists and 7 rebounds.",
         action="silent", min_score=0, max_score=35,
         suggestion="Sports text should be silent"),

    dict(name="Code snippet Python", category="Safe",
         text="def calculate_average(numbers):\n    return sum(numbers) / len(numbers)\n\nresult = calculate_average([1, 2, 3, 4, 5])",
         action="silent", min_score=0, max_score=35,
         suggestion="Code snippets should be silent"),

    dict(name="Code snippet SQL", category="Safe",
         text="SELECT user_id, COUNT(*) as total FROM orders WHERE created_at > '2024-01-01' GROUP BY user_id ORDER BY total DESC LIMIT 10;",
         action="silent", min_score=0, max_score=35,
         suggestion="SQL queries should be silent"),

    dict(name="Product description", category="Safe",
         text="The iPhone 15 Pro features a 48MP camera system with optical zoom and a titanium frame. Available in natural titanium and black titanium.",
         action="silent", min_score=0, max_score=35,
         suggestion="Product text should be silent"),

    dict(name="Academic abstract", category="Safe",
         text="This paper presents a novel approach to natural language processing using transformer architectures. We evaluate on standard benchmarks and achieve state-of-the-art results.",
         action="silent", min_score=0, max_score=35,
         suggestion="Academic text should be silent"),

    dict(name="Meeting agenda", category="Safe",
         text="Q4 Planning Meeting Agenda: 1. Review Q3 results 2. Set Q4 OKRs 3. Budget allocation 4. Headcount planning 5. AOB",
         action="silent", min_score=0, max_score=35,
         suggestion="Meeting agenda should be silent"),

    dict(name="Error log message", category="Safe",
         text="ERROR 2024-01-15 14:23:01 DatabaseConnection timeout after 30000ms. Retrying connection attempt 3 of 5.",
         action="silent", min_score=0, max_score=35,
         suggestion="Technical error logs should be silent"),

    dict(name="Wikipedia style", category="Safe",
         text="The Apollo program was a NASA spaceflight program that landed the first humans on the Moon between 1969 and 1972 using Saturn V rockets.",
         action="silent", min_score=0, max_score=35,
         suggestion="Encyclopedia text should be silent"),

    dict(name="Slack-style message", category="Safe",
         text="Hey team, the deployment is scheduled for Friday 6pm EST. Please make sure all PRs are merged by Thursday EOD. Thanks!",
         action="silent", min_score=0, max_score=35,
         suggestion="Generic work messages should be silent"),

    dict(name="GitHub commit message", category="Safe",
         text="feat: add pagination to user list endpoint\n\n- Add page and limit query params\n- Default to page 1 with 20 items\n- Add total count to response",
         action="silent", min_score=0, max_score=35,
         suggestion="Git commit messages should be silent"),

    dict(name="Stack Overflow question", category="Safe",
         text="How do I reverse a string in Python? I tried using reversed() but it returns an iterator. What is the most efficient way to do this?",
         action="silent", min_score=0, max_score=35,
         suggestion="Technical questions should be silent"),

    dict(name="Business email body", category="Safe",
         text="Please find attached the quarterly report for your review. Let me know if you have any questions or need additional information before the board meeting.",
         action="silent", min_score=0, max_score=35,
         suggestion="Generic business emails should be silent"),

    dict(name="Dictionary definition", category="Safe",
         text="Serendipity: the occurrence and development of events by chance in a happy or beneficial way. Example: a fortunate stroke of serendipity.",
         action="silent", min_score=0, max_score=35,
         suggestion="Dictionary content should be silent"),

    # ── 5 ADVERSARIAL EVASION CASES ────────────────────────────
    dict(name="SSN with asterisks", category="Adversarial",
         text="My SSN is 123*45*6789 please process",
         action="block", min_score=60, max_score=100,
         suggestion="Asterisk-separated SSN must still be caught"),

    dict(name="SSN spelled out", category="Adversarial",
         text="social security number is one two three dash four five dash six seven eight nine",
         action="warn", min_score=30, max_score=80,
         suggestion="Spelled-out SSN — at minimum warn, model should detect context"),

    dict(name="Card with spaces", category="Adversarial",
         text="charge my card 4 5 3 2 1 2 3 4 5 6 7 8 9 0 1 2",
         action="block", min_score=50, max_score=100,
         suggestion="Space-separated card digits should be caught"),

    dict(name="Partial card hint", category="Adversarial",
         text="card ending in 9012 expiry 09/27 CVV 452",
         action="warn", min_score=30, max_score=70,
         suggestion="Partial card + CVV + expiry combo should warn"),

    dict(name="Obfuscated API key", category="Adversarial",
         text="my key is sk dash abc123 XYZ secret key 9999",
         action="warn", min_score=25, max_score=70,
         suggestion="Obfuscated API key — hard to catch, at least warn"),

    # ── 5 NON-US PII FORMATS ───────────────────────────────────
    dict(name="UK NHS number", category="Non-US",
         text="Patient NHS number 485 777 3456 admitted for treatment",
         action="block", min_score=60, max_score=100,
         suggestion="UK NHS numbers should be caught as medical PII"),

    dict(name="UK National Insurance", category="Non-US",
         text="My NI number is AB 12 34 56 C for the tax return",
         action="block", min_score=60, max_score=100,
         suggestion="UK NI number is equivalent to SSN"),

    dict(name="Indian Aadhaar", category="Non-US",
         text="Aadhaar number 1234 5678 9012 required for verification",
         action="block", min_score=60, max_score=100,
         suggestion="Indian national ID should be caught"),

    dict(name="Canadian SIN", category="Non-US",
         text="My Social Insurance Number is 123-456-789 for the T4",
         action="block", min_score=65, max_score=100,
         suggestion="Canadian SIN is equivalent to SSN format"),

    dict(name="EU phone number", category="Non-US",
         text="Contact me at +49 30 12345678 or +33 1 23 45 67 89",
         action="warn", min_score=35, max_score=75,
         suggestion="European phone numbers should warn"),

    # ── 5 IMPLICIT / CONTEXTUAL PII ────────────────────────────
    dict(name="Implicit DOB", category="Implicit",
         text="I was born the year the Berlin Wall fell and grew up in the city that hosted the 1996 Olympics",
         action="warn", min_score=20, max_score=60,
         suggestion="Implicit DOB + location — hard edge case, at minimum low warn"),

    dict(name="Indirect financial", category="Implicit",
         text="My annual package is in the low six figures and I get 15 percent equity vesting over four years",
         action="silent", min_score=0, max_score=40,
         suggestion="Implicit financial beyond TinyBERT scope — needs GPT-4"),

    dict(name="Medical context", category="Implicit",
         text="I have been managing my chronic condition with the medication my endocrinologist prescribed since my diagnosis three years ago",
         action="warn", min_score=20, max_score=50,
         suggestion="Partially detected — acceptable for TinyBERT"),

    dict(name="Location pattern", category="Implicit",
         text="I live two blocks from the Empire State Building and commute to the financial district every day",
         action="silent", min_score=0, max_score=35,
         suggestion="Implicit location beyond TinyBERT scope"),

    dict(name="Identity context", category="Implicit",
         text="As the only female software engineer on the third floor of building B you would recognize me easily",
         action="silent", min_score=0, max_score=35,
         suggestion="Implicit identity beyond TinyBERT scope"),
]

# ── runner ──────────────────────────────────────────────────────────────────

def load_scorer():
    """Import the scoring function from infer.py."""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from infer import score_clipboard_with_pii
        return score_clipboard_with_pii
    except ImportError as e:
        print(f"{RED}ERROR: Cannot import infer.py - {e}{RESET}")
        print("Make sure you run this from core-sentinel-guardrail/")
        sys.exit(1)

def run_test(scorer, tc: dict) -> TestResult:
    result = TestResult(
        name=tc['name'],
        category=tc['category'],
        input_text=tc['text'],
        expected_action=tc['action'],
        expected_score_min=tc['min_score'],
        expected_score_max=tc['max_score'],
        suggestion=tc.get('suggestion', '')
    )
    try:
        t0 = time.perf_counter()
        out = scorer(tc['text'])
        result.duration_ms = (time.perf_counter() - t0) * 1000

        result.actual_action = out.get('action', 'silent')
        result.actual_score  = out.get('risk_score', 0)
        result.actual_spans  = out.get('spans', [])

        # score in range
        score_ok = (result.expected_score_min
                    <= result.actual_score
                    <= result.expected_score_max)
        # action match
        action_ok = result.actual_action == result.expected_action

        result.passed = score_ok and action_ok

        if not action_ok:
            result.error = (f"Action: expected {result.expected_action!r} "
                            f"got {result.actual_action!r}")
        elif not score_ok:
            result.error = (f"Score {result.actual_score} outside "
                            f"[{result.expected_score_min}"
                            f"-{result.expected_score_max}]")

    except Exception:
        result.error = traceback.format_exc()
    return result

def print_live(r: TestResult, idx: int, total: int):
    icon = f"{GREEN}[+]{RESET}" if r.passed else f"{RED}[x]{RESET}"
    score = r.actual_score
    sc = (GREEN if r.passed else RED) + str(score) + RESET
    print(f"  {icon} [{idx:2d}/{total}] {r.name:<40} "
          f"score={sc:>3}  action={r.actual_action:<7} "
          f"{r.duration_ms:5.0f}ms"
          + (f"  {YELLOW}<- {r.error}{RESET}" if r.error else ""))

# ── HTML report ─────────────────────────────────────────────────────────────

def generate_html(results: list[TestResult], 
                  total_ms: float) -> str:
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    pct    = round(passed / len(results) * 100) if results else 0

    cats = {}
    for r in results:
        cats.setdefault(r.category, []).append(r)

    def badge(action):
        colors = {
            'block':  ('#FFEBEE','#B71C1C'),
            'warn':   ('#FFF8E1','#E65100'),
            'silent': ('#E8F5E9','#1B5E20'),
        }
        bg, fg = colors.get(action, ('#F5F5F5','#333'))
        return (f'<span style="background:{bg};color:{fg};'
                f'padding:2px 8px;border-radius:10px;'
                f'font-size:11px;font-weight:700">{action}</span>')

    def row(r: TestResult, i: int) -> str:
        bg = '#F9FBF9' if r.passed else '#FFF5F5'
        icon = '✓' if r.passed else '✗'
        icon_color = '#2E7D32' if r.passed else '#C62828'
        score_color = ('#2E7D32' if r.actual_score <= 40
                       else '#E65100' if r.actual_score <= 70
                       else '#C62828')
        spans_html = ''
        for s in r.actual_spans[:3]:
            cls = s.get('class','?')
            match = s.get('match','')[:30]
            spans_html += (f'<div style="font-size:10px;'
                           f'color:#666;margin-top:2px">'
                           f'• {cls}: <code>{match}</code></div>')
        err_html = ''
        if r.error and not r.passed:
            err_html = (f'<div style="color:#C62828;'
                        f'font-size:11px;margin-top:4px">'
                        f'⚠ {r.error}</div>')
        sug_html = ''
        if not r.passed and r.suggestion:
            sug_html = (f'<div style="color:#1565C0;'
                        f'font-size:11px;margin-top:4px">'
                        f'💡 {r.suggestion}</div>')
        return f"""
        <tr style="background:{bg}">
          <td style="text-align:center;font-size:16px;
                     color:{icon_color}">{icon}</td>
          <td style="font-size:12px;font-weight:600">{r.name}</td>
          <td style="font-size:11px;color:#555;max-width:200px;
                     word-break:break-word">
            <code>{r.input_text[:80]}{'…' if len(r.input_text)>80 else ''}</code>
          </td>
          <td style="text-align:center">{badge(r.expected_action)}</td>
          <td style="text-align:center">{badge(r.actual_action)}</td>
          <td style="text-align:center;font-weight:700;
                     color:{score_color}">{r.actual_score}</td>
          <td style="font-size:11px">
            {spans_html}{err_html}{sug_html}
          </td>
          <td style="text-align:right;font-size:11px;
                     color:#999">{r.duration_ms:.0f}ms</td>
        </tr>"""

    cat_sections = ''
    for cat, cat_results in cats.items():
        cp = sum(1 for r in cat_results if r.passed)
        cat_sections += f"""
        <h3 style="margin:28px 0 8px;font-size:15px;
                   color:#333;border-bottom:2px solid #eee;
                   padding-bottom:6px">
          {cat} 
          <span style="font-size:12px;font-weight:400;
                       color:#888;margin-left:8px">
            {cp}/{len(cat_results)} passed
          </span>
        </h3>
        <table style="width:100%;border-collapse:collapse;
                      margin-bottom:8px">
          <thead>
            <tr style="background:#F5F5F5">
              <th style="padding:8px;width:30px"></th>
              <th style="padding:8px;text-align:left;
                         font-size:11px">Test</th>
              <th style="padding:8px;text-align:left;
                         font-size:11px">Input</th>
              <th style="padding:8px;font-size:11px">Expected</th>
              <th style="padding:8px;font-size:11px">Actual</th>
              <th style="padding:8px;font-size:11px">Score</th>
              <th style="padding:8px;text-align:left;
                         font-size:11px">Details / Suggestions</th>
              <th style="padding:8px;font-size:11px">Time</th>
            </tr>
          </thead>
          <tbody>
            {''.join(row(r, i) for i, r in enumerate(cat_results, 1))}
          </tbody>
        </table>"""

    # improvement areas
    failed_results = [r for r in results if not r.passed]
    improvements = ''
    if failed_results:
        improvements = '<ul style="margin:0;padding-left:20px">'
        for r in failed_results:
            improvements += (
                f'<li style="margin-bottom:8px">'
                f'<strong>{r.name}</strong> ({r.category}): '
                f'{r.error}. '
                f'<span style="color:#1565C0">{r.suggestion}</span>'
                f'</li>')
        improvements += '</ul>'
    else:
        improvements = ('<p style="color:#2E7D32;font-weight:600">'
                        '🎉 All tests passed! No improvements needed.</p>')

    avg_ms = sum(r.duration_ms for r in results) / len(results)
    slow = [r for r in results if r.duration_ms > 500]

    perf_html = f"""
    <div style="background:#F3F4F6;border-radius:8px;
                padding:12px 16px;margin-bottom:16px">
      <strong>Performance:</strong>
      Avg {avg_ms:.0f}ms per inference · 
      Total suite {total_ms/1000:.1f}s · 
      {len(slow)} tests over 500ms
      {('<br><span style="color:#E65100">Slow tests: ' +
        ', '.join(r.name for r in slow) + '</span>') if slow else ''}
    </div>"""

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Core Sentinel — Test Report</title>
<style>
  body{{font-family:-apple-system,BlinkMacSystemFont,
       'Segoe UI',sans-serif;margin:0;background:#F9FAFB;
       color:#1a1a1a}}
  .header{{background:linear-gradient(135deg,#1C2B4B,#0D47A1);
           padding:28px 36px;color:white}}
  .header h1{{margin:0;font-size:24px;font-weight:700}}
  .header p{{margin:4px 0 0;opacity:.7;font-size:13px}}
  .body{{max-width:1200px;margin:24px auto;padding:0 24px}}
  .stat-row{{display:grid;grid-template-columns:repeat(5,1fr);
             gap:12px;margin-bottom:24px}}
  .stat{{background:white;border-radius:10px;padding:16px;
         border:1px solid #E5E7EB;text-align:center}}
  .stat-num{{font-size:28px;font-weight:700;margin-bottom:2px}}
  .stat-lbl{{font-size:11px;color:#888;text-transform:uppercase;
             letter-spacing:.05em}}
  .card{{background:white;border-radius:10px;padding:20px 24px;
         border:1px solid #E5E7EB;margin-bottom:20px}}
  .card h2{{margin:0 0 16px;font-size:16px;color:#1C2B4B}}
  td,th{{padding:8px 10px;border-bottom:1px solid #F0F0F0}}
  code{{background:#F5F5F5;padding:1px 5px;border-radius:3px;
        font-size:10px}}
</style>
</head>
<body>
<div class="header">
  <h1>🛡 Core Sentinel — Automated Test Report</h1>
  <p>Generated {now} · {len(results)} tests · 
     {total_ms/1000:.1f}s total</p>
</div>
<div class="body">
  <div class="stat-row">
    <div class="stat">
      <div class="stat-num" style="color:#2E7D32">{passed}</div>
      <div class="stat-lbl">Passed</div>
    </div>
    <div class="stat">
      <div class="stat-num" style="color:#C62828">{failed}</div>
      <div class="stat-lbl">Failed</div>
    </div>
    <div class="stat">
      <div class="stat-num" 
           style="color:{'#2E7D32' if pct>=80 else '#E65100' if pct>=60 else '#C62828'}">
        {pct}%
      </div>
      <div class="stat-lbl">Pass rate</div>
    </div>
    <div class="stat">
      <div class="stat-num" style="color:#1565C0">{avg_ms:.0f}ms</div>
      <div class="stat-lbl">Avg inference</div>
    </div>
    <div class="stat">
      <div class="stat-num" style="color:#6A1B9A">{len(results)}</div>
      <div class="stat-lbl">Total tests</div>
    </div>
  </div>

  <div class="card">
    <h2>🔧 Areas to improve</h2>
    {improvements}
  </div>

  {perf_html}

  <div class="card">
    <h2>📋 All test results</h2>
    {cat_sections}
  </div>
</div>
</body>
</html>"""

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}{CYAN}Core Sentinel - Automated Test Suite{RESET}")
    print("-" * 60)
    print(f"Loading scorer from infer.py...", end=' ', flush=True)

    scorer = load_scorer()
    print(f"{GREEN}OK{RESET}")
    print(f"Running {len(TEST_CASES)} tests...\n")

    results = []
    cats = sorted(set(t['category'] for t in TEST_CASES))
    suite_start = time.perf_counter()

    for cat in cats:
        cat_tests = [t for t in TEST_CASES if t['category']==cat]
        print(f"{BOLD}{cat}{RESET} ({len(cat_tests)} tests)")
        for i, tc in enumerate(cat_tests, 1):
            r = run_test(scorer, tc)
            results.append(r)
            print_live(r, i, len(cat_tests))
        print()

    total_ms = (time.perf_counter() - suite_start) * 1000

    # summary
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    pct    = round(passed / len(results) * 100)

    print("-" * 60)
    print(f"{BOLD}Results: "
          f"{GREEN}{passed} passed{RESET} / "
          f"{RED}{failed} failed{RESET} / "
          f"{pct}% pass rate / "
          f"{total_ms/1000:.1f}s total{RESET}")

    if failed:
        print(f"\n{YELLOW}{BOLD}Failed tests:{RESET}")
        for r in results:
            if not r.passed:
                print(f"  {RED}[x]{RESET} {r.name}: {r.error}")
                print(f"    {CYAN}-> {r.suggestion}{RESET}")

    # write HTML report
    report_path = Path("logs/test_report.html")
    report_path.parent.mkdir(exist_ok=True)
    html = generate_html(results, total_ms)
    report_path.write_text(html, encoding='utf-8')
    print(f"\n{GREEN}Report saved:{RESET} {report_path.resolve()}")
    print(f"Open it in your browser to see full details.\n")

    # write JSON summary
    summary_path = Path("logs/test_summary.json")
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "pass_rate_pct": pct,
        "total_duration_ms": round(total_ms),
        "results": [
            {"name": r.name,
             "category": r.category,
             "passed": r.passed,
             "expected_action": r.expected_action,
             "actual_action": r.actual_action,
             "actual_score": r.actual_score,
             "error": r.error,
             "suggestion": r.suggestion,
             "duration_ms": round(r.duration_ms)}
            for r in results
        ]
    }
    summary_path.write_text(
        json.dumps(summary, indent=2), encoding='utf-8')
    print(f"{GREEN}JSON summary:{RESET} {summary_path.resolve()}\n")

    sys.exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
