"""
PII label → risk mapping and regex override layer for Core Sentinel guardrail.

Maps token-level PII labels to LOW / MED / HIGH and aggregates span risks.
Strong PII regex matches (SSN, credit card, email, IP, etc.) upgrade risk to HIGH.
"""

import re
from enum import Enum

# --- Risk level constants ---
LOW = "low"
MED = "med"
HIGH = "high"


class RiskLevel(str, Enum):
    LOW = "low"
    MED = "med"
    HIGH = "high"


# PII label → risk (user-specified policy)
# High: CONTACT, ID, FINANCIAL, HEALTH, AUTH, OTHER_PII
# Medium: NAME, LOCATION
# Low: O / non-PII
LABEL_TO_RISK_MAP = {
    "O": LOW,
    "NAME": MED,
    "CONTACT": HIGH,
    "LOCATION": MED,
    "ID": HIGH,
    "FINANCIAL": HIGH,
    "HEALTH": HIGH,
    "AUTH": HIGH,
    "OTHER_PII": HIGH,
}


def label_to_risk(label: str) -> str:
    """
    Map a single PII label to risk level.

    - High: CONTACT, ID, FINANCIAL, HEALTH, AUTH, OTHER_PII
    - Medium: NAME, LOCATION
    - Low: O or unknown
    """
    if not label:
        return LOW
    normalized = str(label).strip().upper()
    return LABEL_TO_RISK_MAP.get(normalized, LOW)


def aggregate_span_risks(span_labels: list[str]) -> str:
    """
    Aggregate per-span risks into one overall risk.

    Priority: if any span is HIGH → HIGH; else if any span is MED → MED; else LOW.
    """
    if not span_labels:
        return LOW
    risks = [label_to_risk(l) for l in span_labels]
    if HIGH in risks:
        return HIGH
    if MED in risks:
        return MED
    return LOW


# --- Strong PII regex overrides (compiled once) ---
# SSN-like: 123-45-6789 or 123456789 (9 digits, optional dashes)
_SSN = re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b")
# Credit card: 4 groups of 4 digits, optional spaces/dashes (simplified; no Luhn here)
_CC = re.compile(r"\b(?:\d[-\s]*){12,19}\d\b")
# IBAN: 2 letters + 2 digits + up to 30 alphanumeric (basic)
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}\s?[A-Z0-9]{4}\s?[A-Z0-9]{4}\s?[A-Z0-9]{4,14}\b", re.IGNORECASE)
# Bank account / routing: common US routing 9 digits
_ROUTING = re.compile(r"\b\d{9}\b")
# US phone: 917-555-0132, (917) 555-0132, +1 917.555.0132, etc.
_PHONE_US = re.compile(
    r"\b(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b"
)
# Email
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# IPv4
_IPV4 = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|1?\d\d?)\b")
# IPv6 (simplified: 8 groups of hex separated by :)
_IPV6 = re.compile(r"\b(?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}\b")

# (pattern, user-facing trigger name) for message generation
_STRONG_PII_PATTERNS = [_SSN, _CC, _IBAN, _ROUTING, _PHONE_US, _EMAIL, _IPV4, _IPV6]
_STRONG_PII_NAMES = [
    "SSN pattern",
    "credit card pattern",
    "IBAN",
    "bank routing number",
    "phone number",
    "email address",
    "IP address (IPv4)",
    "IP address (IPv6)",
]

assert len(_STRONG_PII_PATTERNS) == len(_STRONG_PII_NAMES)

# --- Company KB regex layer (lazy; from kb_loader) ---
_kb_patterns: list | None = None


def _get_kb_patterns():
    global _kb_patterns
    if _kb_patterns is None:
        from kb_loader import get_active_kb, get_kb_compiled_patterns

        kb = get_active_kb()
        _kb_patterns = get_kb_compiled_patterns(kb)
    return _kb_patterns


def kb_rule_spans(text: str) -> list[dict]:
    """
    Character-level spans from the knowledge-base regex / forbidden-term layer.
    Each item includes source \"kb_rule\" and per-pattern risk (low/med/high).
    """
    if not text:
        return []
    out: list[dict] = []
    for name, pattern, risk in _get_kb_patterns():
        for m in pattern.finditer(text):
            out.append(
                {
                    "start": m.start(),
                    "end": m.end(),
                    "class": name,
                    "match": m.group()[:50],
                    "source": "kb_rule",
                    "risk": risk,
                }
            )
    return out


def iter_regex_pii_matches(text: str):
    """
    Yield (start, end, class_name, matched_text) for every strong-PII regex match.
    Overlapping matches from different patterns are all returned.
    """
    if not text:
        return
    for pat, name in zip(_STRONG_PII_PATTERNS, _STRONG_PII_NAMES):
        for m in pat.finditer(text):
            yield m.start(), m.end(), name, m.group()


def get_pii_override_triggers(text: str) -> list[str]:
    """
    Return human-readable names of strong PII patterns that match the text.
    Used for user-facing messages (e.g. "SSN pattern", "email address").
    Includes KB rules with risk \"high\" (same escalation path as strong regex).
    """
    if not text:
        return []
    triggers = []
    for pat, name in zip(_STRONG_PII_PATTERNS, _STRONG_PII_NAMES):
        if pat.search(text):
            triggers.append(name)
    for name, cre, risk in _get_kb_patterns():
        if str(risk).lower() != HIGH:
            continue
        if cre.search(text):
            triggers.append(name)
    return triggers


def apply_pii_overrides(text: str, base_risk: str) -> str:
    """
    If raw text matches any strong PII pattern, upgrade risk to HIGH.
    KB rules with risk \"high\" are treated the same (can force block path via HIGH).
    Never downgrades; only upgrades.
    """
    if not text or base_risk == HIGH:
        return base_risk
    for pat in _STRONG_PII_PATTERNS:
        if pat.search(text):
            return HIGH
    for _name, cre, risk in _get_kb_patterns():
        if str(risk).lower() != HIGH:
            continue
        if cre.search(text):
            return HIGH
    return base_risk
