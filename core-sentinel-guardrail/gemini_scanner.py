"""
Use Gemini Flash to detect PII in extracted document text (optional; requires GEMINI_API_KEY).
"""

from __future__ import annotations

import json
import os

from google import genai
from google.genai import types

_GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")

PII_PROMPT = """
You are a PII detection expert. Analyze the following
text and identify ALL personally identifiable information.

For each PII item found, return a JSON array with objects:
{
  "text": "the exact PII text found",
  "type": "one of: NAME, EMAIL, PHONE, SSN, CREDIT_CARD,
           ADDRESS, DOB, PASSPORT, NATIONAL_ID, IP_ADDRESS,
           USERNAME, PASSWORD, API_KEY, BANK_ACCOUNT,
           HEALTH_INFO, OTHER_PII",
  "risk": "high, med, or low",
  "start_pos": approximate character position,
  "confidence": 0.0 to 1.0,
  "reason": "brief explanation"
}

Rules:
- high risk: SSN, passwords, API keys, credit cards,
             bank accounts, passports
- med risk: names, emails, phones, addresses, DOB
- low risk: usernames, general references
- Return ONLY the JSON array, no other text
- If no PII found, return []

Text to analyze:
"""


def scan_with_gemini(text: str, model_name: str = "gemini-2.0-flash") -> list:
    """
    Returns list of PII findings from Gemini.
    Falls back to empty list if API unavailable.
    """
    if not _GEMINI_KEY:
        print("[gemini] GEMINI_API_KEY not set — skipping")
        return []

    try:
        client = genai.Client(api_key=_GEMINI_KEY)
        cfg = types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
        )

        chunks = _chunk_text(text, max_chars=30000)
        all_findings = []

        for i, chunk in enumerate(chunks):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=PII_PROMPT + chunk,
                    config=cfg,
                )
                raw = (response.text or "").strip()
                raw = raw.replace("```json", "").replace("```", "").strip()
                findings = json.loads(raw)
                if isinstance(findings, list):
                    all_findings.extend(findings)
            except json.JSONDecodeError:
                print(f"[gemini] JSON parse error on chunk {i}")
                continue
            except Exception as e:
                print(f"[gemini] chunk {i} error: {e}")
                continue

        return all_findings

    except Exception as e:
        print(f"[gemini] scan failed: {e}")
        return []


def _chunk_text(text: str, max_chars: int = 30000) -> list:
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end < len(text):
            break_pos = text.rfind("\n\n", start, end)
            if break_pos > start:
                end = break_pos
        chunks.append(text[start:end])
        start = end
    return chunks


def merge_local_and_gemini(local_result: dict, gemini_findings: list) -> dict:
    """
    Merge TinyBERT local results with Gemini findings.
    Gemini findings are more detailed, local is faster.
    Combined result is most accurate.
    """
    merged = dict(local_result)

    existing_texts = {s.get("match", "").lower() for s in local_result.get("spans", [])}

    for finding in gemini_findings:
        text = finding.get("text", "")
        if text.lower() not in existing_texts:
            merged.setdefault("spans", []).append(
                {
                    "class": finding.get("type", "OTHER_PII"),
                    "match": text,
                    "risk": finding.get("risk", "med"),
                    "source": "gemini",
                    "confidence": finding.get("confidence", 0.8),
                    "reason": finding.get("reason", ""),
                }
            )

    high_from_gemini = any(f.get("risk") == "high" for f in gemini_findings)
    if high_from_gemini and merged.get("risk") != "high":
        merged["risk"] = "high"
        merged["action"] = "block"

    merged["gemini_used"] = True
    merged["gemini_count"] = len(gemini_findings)

    return merged


if __name__ == "__main__":
    test = "My name is John Smith, SSN 123-45-6789, email john@test.com"
    findings = scan_with_gemini(test)
    print(json.dumps(findings, indent=2))
