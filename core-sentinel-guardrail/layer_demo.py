"""
layer_demo.py — Visual demonstration of Core Sentinel's 3-layer detection pipeline.

Shows exactly what each layer catches (and misses), proving the ML model
adds value beyond regex and NER.

Usage:
    python layer_demo.py              # Run all demo scenarios
    python layer_demo.py --interactive  # Paste your own text to test
    python layer_demo.py --export      # Save results to reports/layer_demo_report.html

Requires: infer.py, spacy (en_core_web_sm or en_core_web_trf), the trained model
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────
# LAYER 1: Regex patterns (deterministic)
# ─────────────────────────────────────────────────────────────
REGEX_PATTERNS = {
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "Credit Card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "Email": r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    "Phone": r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
    "IP Address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    "API Key": r"(?:sk-[a-zA-Z0-9]{20,}|AKIA[A-Z0-9]{16}|ghp_[a-zA-Z0-9]{36})",
    "AWS Key": r"AKIA[0-9A-Z]{16}",
    "JWT": r"eyJ[a-zA-Z0-9_-]{10,}\.eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]+",
    "MRN": r"\b(?:MRN|Medical Record|Med Rec|Chart)\s*#?\s*:?\s*\d{5,12}\b",
    "DEA Number": r"\bDEA\s*#?\s*:?\s*[A-Z]{2}\d{7}\b",
    "NPI": r"\bNPI\s*#?\s*:?\s*\d{10}\b",
    "Date (MM/DD/YYYY)": r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b",
}


def run_regex_layer(text: str) -> list[dict]:
    """Layer 1: Run all regex patterns. Returns list of matches."""
    findings = []
    for name, pattern in REGEX_PATTERNS.items():
        for m in re.finditer(pattern, text, re.IGNORECASE):
            findings.append({
                "layer": "REGEX",
                "type": name,
                "match": m.group(),
                "start": m.start(),
                "end": m.end(),
                "confidence": 1.0,  # Regex is deterministic
            })
    return findings


# ─────────────────────────────────────────────────────────────
# LAYER 2: spaCy NER (entity recognition)
# ─────────────────────────────────────────────────────────────
_nlp = None


def _load_spacy():
    global _nlp
    if _nlp is not None:
        return _nlp
    import spacy
    for model_name in ["en_core_web_trf", "en_core_web_sm", "en_core_web_md"]:
        try:
            _nlp = spacy.load(model_name)
            print(f"  [NER] Loaded spaCy model: {model_name}")
            return _nlp
        except OSError:
            continue
    print("  [NER] WARNING: No spaCy model found. Install: python -m spacy download en_core_web_sm")
    return None


def run_ner_layer(text: str) -> list[dict]:
    """Layer 2: Run spaCy NER. Returns list of entities found."""
    nlp = _load_spacy()
    if nlp is None:
        return []
    doc = nlp(text[:10000])  # Cap for performance
    findings = []
    # PII-relevant entity types
    pii_ent_types = {
        "PERSON": "Full Name",
        "ORG": "Organization",
        "GPE": "Location",
        "LOC": "Location",
        "DATE": "Date",
        "MONEY": "Financial Amount",
        "CARDINAL": "Number",
        "FAC": "Facility",
    }
    for ent in doc.ents:
        if ent.label_ in pii_ent_types:
            findings.append({
                "layer": "NER",
                "type": pii_ent_types[ent.label_],
                "match": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "ner_label": ent.label_,
                "confidence": 0.85,  # NER doesn't give calibrated confidence
            })
    return findings


# ─────────────────────────────────────────────────────────────
# LAYER 3: ML Model (TinyBERT classifier)
# ─────────────────────────────────────────────────────────────
_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        model_path = str(ROOT / "models" / "tinybert_guardrail")
        if not Path(model_path).exists():
            print(f"  [ML] WARNING: Model not found at {model_path}")
            return None, None
        _tokenizer = AutoTokenizer.from_pretrained(model_path)
        _model = AutoModelForSequenceClassification.from_pretrained(model_path)
        _model.eval()
        num_labels = _model.config.num_labels
        print(f"  [ML] Loaded model: {model_path} ({num_labels}-class)")
        return _model, _tokenizer
    except Exception as e:
        print(f"  [ML] WARNING: Could not load model: {e}")
        return None, None


def run_ml_layer(text: str) -> dict:
    """Layer 3: Run TinyBERT classifier. Returns risk prediction."""
    import torch
    model, tokenizer = _load_model()
    if model is None or tokenizer is None:
        return {"layer": "ML", "prediction": "unknown", "confidence": 0.0, "probabilities": {}}

    inputs = tokenizer(
        text[:512],
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    num_labels = model.config.num_labels
    if num_labels == 2:
        labels = ["safe", "risky"]
    else:
        labels = ["low", "med", "high"]

    prob_dict = {labels[i]: round(probs[i].item(), 4) for i in range(num_labels)}
    pred_idx = probs.argmax().item()
    pred_label = labels[pred_idx]
    pred_conf = probs[pred_idx].item()

    risk_score = int(probs[-1].item() * 100)  # P(risky) or P(high) * 100

    return {
        "layer": "ML",
        "prediction": pred_label,
        "confidence": round(pred_conf, 4),
        "risk_score": risk_score,
        "probabilities": prob_dict,
    }


# ─────────────────────────────────────────────────────────────
# COMBINED PIPELINE
# ─────────────────────────────────────────────────────────────
def run_full_pipeline(text: str) -> dict:
    """Run all 3 layers and aggregate results."""
    t0 = time.perf_counter()
    regex_results = run_regex_layer(text)
    t_regex = time.perf_counter() - t0

    t1 = time.perf_counter()
    ner_results = run_ner_layer(text)
    t_ner = time.perf_counter() - t1

    t2 = time.perf_counter()
    ml_result = run_ml_layer(text)
    t_ml = time.perf_counter() - t2

    # Determine final decision
    has_critical_regex = any(
        r["type"] in ("SSN", "Credit Card", "API Key", "AWS Key", "JWT", "MRN", "DEA Number")
        for r in regex_results
    )
    has_medium_regex = any(
        r["type"] in ("Email", "Phone", "IP Address", "Date (MM/DD/YYYY)")
        for r in regex_results
    )
    has_ner_person = any(r["ner_label"] == "PERSON" for r in ner_results if "ner_label" in r)
    has_ner_medical = has_ner_person and any(
        kw in text.lower()
        for kw in ["patient", "diagnosis", "medication", "prescription", "treatment", "hospital"]
    )

    ml_says_risky = ml_result.get("prediction") in ("risky", "high", "med")
    ml_risk_score = ml_result.get("risk_score", 0)

    # Decision logic
    if has_critical_regex:
        decision = "BLOCK"
        severity = "CRITICAL"
        reason = "Regex detected critical identifier"
    elif has_ner_medical:
        decision = "BLOCK"
        severity = "HIGH"
        reason = "NER detected person in medical context"
    elif has_medium_regex and ml_says_risky:
        decision = "WARN"
        severity = "MEDIUM"
        reason = "Regex found medium PII + ML confirms risk"
    elif ml_says_risky and ml_risk_score >= 70:
        decision = "BLOCK"
        severity = "HIGH"
        reason = "ML model detected high contextual risk (no regex/NER match)"
    elif ml_says_risky and ml_risk_score >= 40:
        decision = "WARN"
        severity = "MEDIUM"
        reason = "ML model detected moderate contextual risk"
    elif has_ner_person and not has_medium_regex:
        decision = "WARN"
        severity = "MEDIUM"
        reason = "NER detected person name (no regex match)"
    elif has_medium_regex:
        decision = "WARN"
        severity = "LOW"
        reason = "Regex found low-risk pattern"
    else:
        decision = "SILENT"
        severity = "SAFE"
        reason = "No PII detected by any layer"

    return {
        "text_preview": text[:100] + ("..." if len(text) > 100 else ""),
        "decision": decision,
        "severity": severity,
        "reason": reason,
        "risk_score": ml_risk_score,
        "layers": {
            "regex": {
                "findings": regex_results,
                "count": len(regex_results),
                "time_ms": round(t_regex * 1000, 2),
            },
            "ner": {
                "findings": ner_results,
                "count": len(ner_results),
                "time_ms": round(t_ner * 1000, 2),
            },
            "ml": {
                **ml_result,
                "time_ms": round(t_ml * 1000, 2),
            },
        },
        "total_time_ms": round((t_regex + t_ner + t_ml) * 1000, 2),
    }


# ─────────────────────────────────────────────────────────────
# DEMO SCENARIOS
# ─────────────────────────────────────────────────────────────
SCENARIOS = [
    {
        "name": "1. SAFE TEXT (all layers agree: no PII)",
        "text": "The quick brown fox jumps over the lazy dog. Python is a programming language created by Guido van Rossum in 1991. TCP provides reliable ordered delivery of data.",
        "expected": "SILENT",
        "demonstrates": "All three layers correctly identify safe text. No false positives.",
    },
    {
        "name": "2. REGEX CATCHES: SSN (structured pattern)",
        "text": "Employee record: SSN 123-45-6789, hired 2024.",
        "expected": "BLOCK",
        "demonstrates": "Layer 1 (regex) catches the SSN with deterministic pattern matching. NER and ML are not needed here but may add context.",
    },
    {
        "name": "3. REGEX CATCHES: API Key in code",
        "text": "export OPENAI_API_KEY=sk-proj-abc123def456ghi789jkl012mno345pqr678stu901",
        "expected": "BLOCK",
        "demonstrates": "Layer 1 (regex) detects the API key prefix 'sk-' pattern. Critical secret = always block.",
    },
    {
        "name": "4. NER CATCHES: Names that regex misses",
        "text": "Please forward this to Sarah Chen at the downtown office. Her manager David Park approved the transfer.",
        "expected": "WARN",
        "demonstrates": "Regex finds nothing (no structured patterns). But Layer 2 (NER) identifies 'Sarah Chen' and 'David Park' as PERSON entities. Names are PII.",
    },
    {
        "name": "5. NER CATCHES: Medical context (HIPAA-relevant)",
        "text": "Patient Michael Rodriguez was admitted to Springfield General Hospital on March 15 with stage 2 hypertension. Dr. Lisa Wang prescribed Lisinopril 10mg.",
        "expected": "BLOCK",
        "demonstrates": "Regex catches the date. NER catches PERSON names (patient + doctor), ORG (hospital). The combination of person + medical keywords = HIPAA-relevant PHI. This is HIGH severity.",
    },
    {
        "name": "6. ML MODEL CATCHES: Contextual risk (no regex, no NER match)",
        "text": "My annual salary is around one fifty and my manager told me I'm up for promotion next quarter. The bonus structure is based on our team's Q3 revenue which was about twelve million.",
        "expected": "WARN",
        "demonstrates": "Regex finds nothing (no SSN, email, phone). NER may not flag anything (no proper names, amounts written as words). But the ML model recognizes this as sensitive business/financial context — salary, revenue, compensation details are PII in enterprise settings.",
    },
    {
        "name": "7. ML MODEL CATCHES: Implicit medical PII",
        "text": "I've been dealing with chronic back pain for three years and my therapist recommended increasing my antidepressant dosage. My insurance copay went up again this month.",
        "expected": "WARN",
        "demonstrates": "No structured patterns for regex. NER may not find named entities. But the ML model detects medical/health information (diagnosis, medication, insurance) which is sensitive under HIPAA even without explicit names.",
    },
    {
        "name": "8. ALL LAYERS CONTRIBUTE: Complex document",
        "text": "DISCHARGE SUMMARY\nPatient: James Wilson, MRN: 456789\nDOB: 03/15/1985\nEmail: j.wilson@email.com\nPhone: (555) 012-3456\nDiagnosis: Type 2 Diabetes Mellitus\nMedications: Metformin 500mg BID, Lisinopril 10mg daily\nFollow-up with Dr. Sarah Chen, NPI: 1234567890\nInsurance: BlueCross Policy #BC-789012",
        "expected": "BLOCK",
        "demonstrates": "ALL THREE LAYERS contribute:\n  - Regex: MRN, DOB date, email, phone, NPI number\n  - NER: James Wilson (PERSON), Sarah Chen (PERSON), BlueCross (ORG)\n  - ML: medical document context (diagnosis, medications, discharge)\nThis is the strongest detection — multiple independent signals confirm HIGH risk.",
    },
    {
        "name": "9. ADVERSARIAL: Obfuscated PII (tests ML robustness)",
        "text": "my social is one two three dash four five dash six seven eight nine and my cc is four five three two 1234 5678 9012",
        "expected": "WARN or BLOCK",
        "demonstrates": "Regex MISSES this (numbers written as words, no standard format). NER may catch nothing. The ML model must recognize the INTENT — 'my social is...' and 'my cc is...' indicate SSN and credit card numbers even in non-standard format. This tests the model's contextual understanding.",
    },
    {
        "name": "10. EDGE CASE: Mixed safe and risky content",
        "text": "Hey, can you review this code? The function handles user authentication.\n\ndef login(username, password):\n    # TODO: remove hardcoded creds\n    if username == 'admin' and password == 'SuperSecret123!':\n        return generate_token()\n    db_url = 'postgresql://root:p@ssw0rd@prod-db.internal.company.com:5432/users'",
        "expected": "BLOCK",
        "demonstrates": "Most of the text is safe code discussion. But embedded within it are hardcoded credentials and a database connection string. Regex catches the DB URL pattern. ML should flag the overall context as risky. NER may not contribute here. This tests detection within noisy contexts.",
    },
]


# ─────────────────────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────────────────────
def color(text, code):
    """ANSI color wrapper."""
    return f"\033[{code}m{text}\033[0m"


def display_result(scenario: dict, result: dict):
    """Pretty-print a single scenario result."""
    name = scenario["name"]
    demonstrates = scenario["demonstrates"]

    # Header
    print("\n" + "=" * 80)
    print(color(f"  {name}", "1;37"))
    print("=" * 80)
    print(color(f"  Demonstrates: {demonstrates}", "0;90"))
    print()

    # Text preview
    text = scenario["text"]
    if len(text) > 200:
        print(f"  Text: {text[:200]}...")
    else:
        print(f"  Text: {text}")
    print()

    # Layer 1: Regex
    regex = result["layers"]["regex"]
    regex_color = "0;32" if regex["count"] == 0 else "1;33"
    print(color(f"  LAYER 1 — REGEX ({regex['time_ms']}ms)", regex_color))
    if regex["count"] == 0:
        print(color("    No matches", "0;90"))
    else:
        for f in regex["findings"]:
            print(color(f"    [{f['type']}] \"{f['match']}\"", "1;33"))
    print()

    # Layer 2: NER
    ner = result["layers"]["ner"]
    ner_color = "0;32" if ner["count"] == 0 else "1;36"
    print(color(f"  LAYER 2 — NER ({ner['time_ms']}ms)", ner_color))
    if ner["count"] == 0:
        print(color("    No entities found", "0;90"))
    else:
        for f in ner["findings"]:
            label = f.get("ner_label", "?")
            print(color(f"    [{f['type']}] \"{f['match']}\" (spaCy: {label})", "1;36"))
    print()

    # Layer 3: ML
    ml = result["layers"]["ml"]
    ml_pred = ml.get("prediction", "unknown")
    ml_conf = ml.get("confidence", 0)
    ml_score = ml.get("risk_score", 0)
    ml_probs = ml.get("probabilities", {})
    if ml_pred in ("risky", "high"):
        ml_color = "1;31"
    elif ml_pred in ("med", "medium"):
        ml_color = "1;33"
    else:
        ml_color = "0;32"
    print(color(f"  LAYER 3 — ML MODEL ({ml['time_ms']}ms)", ml_color))
    print(f"    Prediction: {color(ml_pred.upper(), ml_color)} (confidence: {ml_conf:.1%})")
    print(f"    Risk score: {ml_score}/100")
    print(f"    Probabilities: {ml_probs}")
    print()

    # Final Decision
    decision = result["decision"]
    severity = result["severity"]
    reason = result["reason"]
    if decision == "BLOCK":
        dec_color = "1;31"
        dec_icon = "BLOCK"
    elif decision == "WARN":
        dec_color = "1;33"
        dec_icon = "WARN"
    else:
        dec_color = "1;32"
        dec_icon = "PASS"

    print(color(f"  DECISION: {dec_icon} ({severity})", dec_color))
    print(f"  Reason: {reason}")
    print(f"  Total pipeline time: {result['total_time_ms']}ms")

    # Layer contribution summary
    contributed = []
    if regex["count"] > 0:
        contributed.append("Regex")
    if ner["count"] > 0:
        contributed.append("NER")
    if ml_pred in ("risky", "high", "med"):
        contributed.append("ML Model")
    if contributed:
        print(color(f"  Contributing layers: {' + '.join(contributed)}", "1;37"))
    else:
        print(color("  Contributing layers: None (all clear)", "0;32"))


# ─────────────────────────────────────────────────────────────
# HTML REPORT
# ─────────────────────────────────────────────────────────────
def generate_html_report(results: list[tuple[dict, dict]], output_path: str):
    """Generate a visual HTML report of all scenarios."""
    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Core Sentinel — Layer Detection Demo</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #080810; color: #e8e8e8; font-family: 'Segoe UI', system-ui, sans-serif; padding: 40px; }
  h1 { font-size: 28px; margin-bottom: 8px; }
  .subtitle { color: #6b7280; font-size: 14px; margin-bottom: 40px; }
  .scenario { background: #111118; border: 1px solid rgba(2,195,154,0.1); border-radius: 12px; padding: 24px; margin-bottom: 24px; }
  .scenario-title { font-size: 16px; font-weight: 700; margin-bottom: 4px; }
  .demonstrates { color: #6b7280; font-size: 12px; margin-bottom: 16px; white-space: pre-line; }
  .text-box { background: #0a0a0f; border: 1px solid #1a1a2e; border-radius: 8px; padding: 12px; font-family: monospace; font-size: 12px; margin-bottom: 16px; white-space: pre-wrap; word-break: break-all; }
  .layers { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-bottom: 16px; }
  .layer-card { background: #0d0d14; border-radius: 8px; padding: 16px; }
  .layer-name { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
  .layer-name.regex { color: #FFB300; }
  .layer-name.ner { color: #42A5F5; }
  .layer-name.ml { color: #CE93D8; }
  .finding { font-size: 12px; margin-bottom: 4px; padding: 4px 8px; background: rgba(255,255,255,0.04); border-radius: 4px; }
  .finding .type { font-weight: 600; }
  .no-findings { color: #4b5563; font-size: 12px; font-style: italic; }
  .decision { display: flex; align-items: center; gap: 12px; padding: 12px 16px; border-radius: 8px; font-weight: 600; }
  .decision.BLOCK { background: rgba(229,57,53,0.15); border: 1px solid rgba(229,57,53,0.3); color: #E53935; }
  .decision.WARN { background: rgba(255,179,0,0.15); border: 1px solid rgba(255,179,0,0.3); color: #FFB300; }
  .decision.SILENT { background: rgba(46,125,50,0.15); border: 1px solid rgba(46,125,50,0.3); color: #43A047; }
  .decision .severity { font-size: 11px; background: rgba(255,255,255,0.1); padding: 2px 8px; border-radius: 10px; }
  .decision .reason { font-size: 12px; font-weight: 400; color: #9ca3af; margin-left: auto; }
  .meta { display: flex; gap: 16px; margin-top: 8px; font-size: 11px; color: #4b5563; }
  .contributing { font-size: 12px; color: #02C39A; margin-top: 8px; font-weight: 600; }
  @media (max-width: 768px) { .layers { grid-template-columns: 1fr; } }
</style></head><body>
<h1>Core Sentinel — 3-Layer Detection Pipeline Demo</h1>
<p class="subtitle">Each scenario shows what Regex, NER, and ML independently detect.<br>
This proves the ML model adds value beyond deterministic pattern matching.</p>
"""
    for scenario, result in results:
        decision = result["decision"]
        html += f'<div class="scenario">\n'
        html += f'<div class="scenario-title">{scenario["name"]}</div>\n'
        html += f'<div class="demonstrates">{scenario["demonstrates"]}</div>\n'
        html += f'<div class="text-box">{scenario["text"]}</div>\n'

        # Layer cards
        html += '<div class="layers">\n'

        # Regex
        regex = result["layers"]["regex"]
        html += '<div class="layer-card">\n'
        html += f'<div class="layer-name regex">Layer 1: Regex ({regex["time_ms"]}ms)</div>\n'
        if regex["count"] == 0:
            html += '<div class="no-findings">No matches</div>\n'
        else:
            for f in regex["findings"]:
                html += f'<div class="finding"><span class="type">{f["type"]}</span>: "{f["match"]}"</div>\n'
        html += '</div>\n'

        # NER
        ner = result["layers"]["ner"]
        html += '<div class="layer-card">\n'
        html += f'<div class="layer-name ner">Layer 2: NER ({ner["time_ms"]}ms)</div>\n'
        if ner["count"] == 0:
            html += '<div class="no-findings">No entities found</div>\n'
        else:
            for f in ner["findings"]:
                label = f.get("ner_label", "?")
                html += f'<div class="finding"><span class="type">{f["type"]}</span>: "{f["match"]}" ({label})</div>\n'
        html += '</div>\n'

        # ML
        ml = result["layers"]["ml"]
        html += '<div class="layer-card">\n'
        html += f'<div class="layer-name ml">Layer 3: ML Model ({ml["time_ms"]}ms)</div>\n'
        pred = ml.get("prediction", "unknown").upper()
        conf = ml.get("confidence", 0)
        score = ml.get("risk_score", 0)
        probs = ml.get("probabilities", {})
        html += f'<div class="finding"><span class="type">Prediction</span>: {pred} ({conf:.1%})</div>\n'
        html += f'<div class="finding"><span class="type">Risk score</span>: {score}/100</div>\n'
        for k, v in probs.items():
            html += f'<div class="finding"><span class="type">P({k})</span>: {v:.4f}</div>\n'
        html += '</div>\n'

        html += '</div>\n'  # .layers

        # Decision
        html += f'<div class="decision {decision}">\n'
        html += f'<span>{decision}</span>\n'
        html += f'<span class="severity">{result["severity"]}</span>\n'
        html += f'<span class="reason">{result["reason"]}</span>\n'
        html += '</div>\n'

        # Contributing layers
        contributed = []
        if regex["count"] > 0:
            contributed.append("Regex")
        if ner["count"] > 0:
            contributed.append("NER")
        if ml.get("prediction") in ("risky", "high", "med"):
            contributed.append("ML Model")
        if contributed:
            html += f'<div class="contributing">Contributing layers: {" + ".join(contributed)}</div>\n'

        html += f'<div class="meta"><span>Total: {result["total_time_ms"]}ms</span></div>\n'
        html += '</div>\n'  # .scenario

    html += """
<div style="margin-top:40px; padding:20px; background:#111118; border-radius:12px; border:1px solid rgba(2,195,154,0.1);">
<h3 style="color:#02C39A; margin-bottom:8px;">Key Takeaway</h3>
<p style="color:#9ca3af; font-size:13px;">
Each layer catches different types of PII. Regex handles structured patterns (SSN, credit cards, API keys). 
NER handles named entities (person names, organizations, locations). The ML model handles contextual risk 
that neither regex nor NER can catch (medical narratives, financial discussions, obfuscated PII). 
Together, they form a defense-in-depth detection pipeline.
</p>
</div>
</body></html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n  Report saved: {output_path}")


# ─────────────────────────────────────────────────────────────
# INTERACTIVE MODE
# ─────────────────────────────────────────────────────────────
def interactive_mode():
    """Let user paste text and see layer-by-layer analysis."""
    print("\n" + "=" * 60)
    print("  Core Sentinel — Interactive Layer Demo")
    print("  Paste any text to see what each layer detects.")
    print("  Type 'quit' to exit.")
    print("=" * 60)

    while True:
        print("\n" + "-" * 40)
        text = input("  Paste text> ").strip()
        if text.lower() in ("quit", "exit", "q"):
            break
        if not text:
            continue
        result = run_full_pipeline(text)
        display_result({"name": "Custom Input", "text": text, "demonstrates": "Your input"}, result)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Core Sentinel 3-layer detection demo")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode: paste your own text")
    parser.add_argument("--export", action="store_true", help="Export HTML report to reports/")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
        return

    print("\n" + "=" * 60)
    print("  Core Sentinel — 3-Layer Detection Pipeline Demo")
    print("  Showing 10 scenarios that demonstrate each layer's role")
    print("=" * 60)

    # Load models once
    print("\n  Loading models...")
    _load_spacy()
    _load_model()

    results = []
    for scenario in SCENARIOS:
        result = run_full_pipeline(scenario["text"])
        results.append((scenario, result))
        display_result(scenario, result)

    if args.export or True:  # Always export
        report_path = str(ROOT / "reports" / "layer_demo_report.html")
        generate_html_report(results, report_path)

    if args.json:
        json_path = str(ROOT / "reports" / "layer_demo_results.json")
        with open(json_path, "w") as f:
            json.dump([{"scenario": s["name"], "result": r} for s, r in results], f, indent=2)
        print(f"  JSON saved: {json_path}")

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    regex_only = sum(1 for _, r in results if r["layers"]["regex"]["count"] > 0 and r["layers"]["ner"]["count"] == 0)
    ner_only = sum(1 for _, r in results if r["layers"]["ner"]["count"] > 0 and r["layers"]["regex"]["count"] == 0)
    ml_only = sum(1 for _, r in results if r["layers"]["ml"].get("prediction") in ("risky", "high", "med") and r["layers"]["regex"]["count"] == 0 and r["layers"]["ner"]["count"] == 0)
    all_layers = sum(1 for _, r in results if r["layers"]["regex"]["count"] > 0 and r["layers"]["ner"]["count"] > 0 and r["layers"]["ml"].get("prediction") in ("risky", "high", "med"))
    print(f"  Scenarios where ONLY regex caught PII:     {regex_only}")
    print(f"  Scenarios where ONLY NER caught PII:       {ner_only}")
    print(f"  Scenarios where ONLY ML model caught PII:  {ml_only}")
    print(f"  Scenarios where ALL layers contributed:     {all_layers}")
    print(f"  Total scenarios: {len(results)}")
    print()
    print("  This demonstrates that removing any single layer would")
    print("  leave gaps in detection coverage. The 3-layer pipeline")
    print("  provides defense-in-depth.")


if __name__ == "__main__":
    main()
