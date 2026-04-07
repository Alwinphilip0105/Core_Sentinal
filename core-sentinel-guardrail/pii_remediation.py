"""
Simple regex-based PII remediation: mask, hash, or encrypt placeholders.
Demo only; not production crypto.
"""

import base64
import hashlib
import re

# Patterns (aligned with risk_mapping for consistency)
_SSN = re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b")
_CC = re.compile(r"\b(?:\d[-\s]*){12,19}\d\b")
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_IPV4 = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|1?\d\d?)\b"
)
_IBAN = re.compile(
    r"\b[A-Z]{2}\d{2}\s?[A-Z0-9]{4}\s?[A-Z0-9]{4}\s?[A-Z0-9]{4,14}\b", re.IGNORECASE
)
_ROUTING = re.compile(r"\b\d{9}\b")

# Order matters for replacement (longer/more specific first)
_PATTERNS = [
    ("ssn", _SSN),
    ("cc", _CC),
    ("iban", _IBAN),
    ("routing", _ROUTING),
    ("email", _EMAIL),
    ("ipv4", _IPV4),
]


def _last_four(s: str) -> str:
    digits = "".join(c for c in s if c.isdigit())
    return digits[-4:] if len(digits) >= 4 else digits or "****"


def _mask_repl(m: re.Match, name: str) -> str:
    raw = m.group(0)
    if name == "ssn":
        return f"***-**-{_last_four(raw)}"
    if name in ("cc", "iban", "routing"):
        return f"**** **** **** {_last_four(raw)}"
    if name == "email":
        return "[email_redacted]"
    if name == "ipv4":
        return "[ip_redacted]"
    return "[redacted]"


def mask_pii(text: str) -> str:
    """Replace PII with masked versions (e.g. ***-**-6789, [email_redacted])."""
    out = text
    for name, pat in _PATTERNS:
        out = pat.sub(lambda m, n=name: _mask_repl(m, n), out)
    return out


def hash_pii(text: str) -> str:
    """Replace each PII substring with HASH::<sha256_hex>."""
    out = text
    for _, pat in _PATTERNS:
        out = pat.sub(
            lambda m: "HASH::" + hashlib.sha256(m.group(0).encode("utf-8")).hexdigest()[:16],
            out,
        )
    return out


def encrypt_pii(text: str) -> str:
    """Replace each PII substring with ENC::<base64> (demo only)."""
    out = text
    for _, pat in _PATTERNS:
        out = pat.sub(
            lambda m: "ENC::" + base64.b64encode(m.group(0).encode("utf-8")).decode("ascii"),
            out,
        )
    return out


def redact_all_literal(text: str) -> str:
    """Replace every PII match with the literal token [REDACTED] (patterns applied in order)."""
    out = text or ""
    for _, pat in _PATTERNS:
        out = pat.sub("[REDACTED]", out)
    return out


def remediate_text(text: str, action: str) -> str:
    """Dispatch to mask_pii, hash_pii, or encrypt_pii. action in ('redact','mask','hash','encrypt')."""
    a = (action or "").lower().strip()
    if a in ("redact", "mask"):
        return mask_pii(text)
    if a == "hash":
        return hash_pii(text)
    if a == "encrypt":
        return encrypt_pii(text)
    return text
