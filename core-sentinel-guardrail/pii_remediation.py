"""
PII remediation: mask, hash, encrypt (Fernet), and span-based redaction for clipboard UX.
Use GUARDRAIL_ENCRYPT_KEY for encrypt_text / encrypt_pii; default key is for demos only.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re

from cryptography.fernet import Fernet

# Patterns (aligned with risk_mapping for consistency)
_SSN = re.compile(
    r"(?ix)"
    r"(?:"
    r"\b\d{3}[-\s*]?\d{2}[-\s*]?\d{4}\b"
    r"|"
    r"(?:ssn|social\s*security(?:\s*number)?|tax\s*id)\s*[:#-]?\s*"
    r"\d{3}[-\s*]?\d{2}[-\s*]?\d{3,4}\b"
    r")"
)
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


def _make_fernet_key(password: str = "guardrail-default") -> bytes:
    raw = hashlib.sha256(password.encode()).digest()
    return base64.urlsafe_b64encode(raw)


def encrypt_text(text: str, password: str | None = None) -> str:
    """Symmetric encrypt full string (Fernet). Key from password or GUARDRAIL_ENCRYPT_KEY."""
    pwd = password or os.environ.get("GUARDRAIL_ENCRYPT_KEY", "guardrail-default")
    f = Fernet(_make_fernet_key(pwd))
    return f.encrypt(text.encode()).decode()


def decrypt_text(token: str, password: str | None = None) -> str:
    """Decrypt a string produced by encrypt_text."""
    pwd = password or os.environ.get("GUARDRAIL_ENCRYPT_KEY", "guardrail-default")
    f = Fernet(_make_fernet_key(pwd))
    return f.decrypt(token.encode()).decode()


def _last_four(s: str) -> str:
    digits = "".join(c for c in s if c.isdigit())
    return digits[-4:] if len(digits) >= 4 else digits or "****"


def _mask_email_token(raw: str) -> str:
    """Local-part asterisk mask; keep @domain visible (same idea as span partial mode)."""
    if "@" not in raw:
        return "*" * len(raw) if raw else ""
    user, _, rest = raw.partition("@")
    domain = rest
    if len(user) <= 2:
        masked_user = "*" * len(user)
    else:
        masked_user = user[0] + "*" * (len(user) - 2) + user[-1]
    return f"{masked_user}@{domain}"


def _mask_ipv4_token(raw: str) -> str:
    """Hide first three octets; keep last (common for LAN / host id hints without full exposure)."""
    parts = raw.split(".")
    if len(parts) == 4 and all(p.isdigit() for p in parts):
        return f"***.***.***.{parts[3]}"
    return "*" * len(raw) if raw else ""


def _mask_repl(m: re.Match, name: str) -> str:
    raw = m.group(0)
    if name == "ssn":
        return f"***-**-{_last_four(raw)}"
    if name in ("cc", "iban", "routing"):
        return f"**** **** **** {_last_four(raw)}"
    if name == "email":
        return _mask_email_token(raw)
    if name == "ipv4":
        return _mask_ipv4_token(raw)
    return "*" * len(raw) if raw else ""


def mask_pii(text: str) -> str:
    """Replace PII with masked values (partial digits, masked email local-part, masked IP octets, * elsewhere)."""
    out = text
    for name, pat in _PATTERNS:
        out = pat.sub(lambda m, n=name: _mask_repl(m, n), out)
    return out


def _resolve_span_bounds(text: str, span: dict) -> tuple[int, int, str] | None:
    """
    Resolve (start, end, segment) for a span. Prefer valid start/end; otherwise
    locate ``match`` in ``text`` (case-sensitive, then case-insensitive).
    If there is no ``match`` string, use offsets only when they slice non-empty text.
    """
    n = len(text)
    match_raw = span.get("match", "")
    match = str(match_raw).strip() if match_raw is not None else ""
    start = span.get("start")
    end = span.get("end")
    try:
        si = int(start) if start is not None else None
        ei = int(end) if end is not None else None
    except (TypeError, ValueError):
        si, ei = None, None

    if match:
        valid = (
            si is not None
            and ei is not None
            and 0 <= si < ei <= n
            and text[si:ei].strip() != ""
        )
        if valid:
            seg = text[si:ei]
            if seg.strip() == match.strip():
                return si, ei, seg
        pos = text.find(match)
        if pos == -1:
            pos = text.lower().find(match.lower())
        if pos == -1:
            return None
        end_pos = pos + len(match)
        return pos, end_pos, text[pos:end_pos]

    if si is None or ei is None:
        return None
    if not (0 <= si < ei <= n) or not text[si:ei].strip():
        return None
    return si, ei, text[si:ei]


def _dedupe_non_overlapping(
    items: list[tuple[int, int, str]],
) -> list[tuple[int, int, str]]:
    """Keep non-overlapping ranges; ``items`` sorted by start descending."""
    deduped: list[tuple[int, int, str]] = []
    for r in items:
        overlaps = any(
            not (r[1] <= d[0] or r[0] >= d[1])
            for d in deduped
        )
        if not overlaps:
            deduped.append(r)
    return deduped


def mask_pii_spans(text: str, spans: list, mode: str = "partial") -> str:
    """
    Replace span regions with masked text (offsets preserved by applying right-to-left).

    Resolves each span with :func:`_resolve_span_bounds` so bad ``start``/``end``
    are corrected via ``match`` search when possible.

    mode options:
    - "partial"  : show first + last chars / type-aware partial masking
    - "full"     : replace entirely with [CLASS]
    - "stars"    : replace all chars with *
    - "type_only": show type label only e.g. [EMAIL]
    """
    if not text:
        return text or ""
    if not spans:
        return text
    mode = (mode or "partial").strip().lower()
    if mode not in ("partial", "full", "stars", "type_only"):
        mode = "partial"

    replacements: list[tuple[int, int, str]] = []
    for span in spans:
        if not isinstance(span, dict):
            continue
        resolved = _resolve_span_bounds(text, span)
        if resolved is None:
            continue
        start, end, segment = resolved
        cls = str(span.get("class", "PII")).replace("]", "}")
        masked = _partial_mask(segment, cls, mode)
        replacements.append((start, end, masked))

    if not replacements:
        return text

    replacements.sort(key=lambda x: x[0], reverse=True)
    deduped = _dedupe_non_overlapping(replacements)

    result = text
    for start, end, replacement in deduped:
        result = result[:start] + replacement + result[end:]
    return result


def _partial_mask(value: str, cls: str, mode: str = "partial") -> str:
    """Smart masking based on PII type and mode."""
    c = str(cls or "PII")
    c_up = c.upper()
    c_lo = c.lower()

    if mode == "full":
        return f"[{c}]"

    if mode == "stars":
        return "*" * len(value)

    if mode == "type_only":
        return f"[{c}]"

    # PARTIAL mode — type-aware masking
    v = value.strip()
    length = len(v)

    if length == 0:
        return value

    # EMAIL: show domain, mask user
    if c_lo == "email" or c_up == "EMAIL" or c_up == "CONTACT" or "email" in c_lo:
        if "@" in v:
            parts = v.split("@", 1)
            user = parts[0]
            domain = parts[1] if len(parts) > 1 else ""
            if len(user) <= 2:
                masked_user = "*" * len(user)
            else:
                masked_user = user[0] + "*" * (len(user) - 2) + user[-1]
            return f"{masked_user}@{domain}"

    # PHONE (CONTACT without @ treated as phone when digits present)
    if (
        c_lo == "phone"
        or c_up == "PHONE"
        or (c_up == "CONTACT" and "@" not in v)
    ) and any(ch.isdigit() for ch in v):
        digits_only = "".join(ch for ch in v if ch.isdigit())
        if len(digits_only) >= 4:
            visible = digits_only[-4:]
            return f"***-***-{visible}"
        return "*" * length

    # SSN
    if (
        "ssn" in c_lo
        or "ssn pattern" in c_lo
        or c_up in ("SSN", "ID")
        or c == "SSN pattern"
    ):
        digits = "".join(ch for ch in v if ch.isdigit())
        if len(digits) >= 4:
            return f"***-**-{digits[-4:]}"
        return "***-**-****"

    # CREDIT CARD / FINANCIAL
    if (
        c_up in ("FINANCIAL", "CREDIT_CARD")
        or c_lo in ("credit_card", "cc")
        or "card" in c_lo
    ):
        digits = "".join(ch for ch in v if ch.isdigit())
        if len(digits) >= 4:
            last4 = digits[-4:]
            return f"****-****-****-{last4}"
        return "****-****-****-****"

    # API KEY / AUTH
    if c_up in ("AUTH", "API_KEY", "PASSWORD") or c_lo in ("auth", "password", "api_key"):
        if length <= 8:
            return v[0] + "*" * (length - 1) if length else ""
        return v[:4] + "*" * (length - 8) + v[-4:]

    # NAME
    if c_up == "NAME" or c_lo == "name":
        words = v.split()
        masked_words: list[str] = []
        for word in words:
            if len(word) <= 1:
                masked_words.append(word)
            else:
                masked_words.append(word[0] + "*" * (len(word) - 1))
        return " ".join(masked_words)

    # DEFAULT partial mask
    if length <= 4:
        return v[0] + "*" * (length - 1)
    if length <= 8:
        return v[:2] + "*" * (length - 3) + v[-1]
    return v[:2] + "*" * (length - 4) + v[-2:]


def redact_for_clipboard(text: str, spans: list, mode: str = "partial") -> str:
    """Return text with PII masked (per mode), ready to copy to clipboard."""
    if not spans:
        return text
    return mask_pii_spans(text, spans, mode=mode)


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
    """Replace each PII substring with ENC::<fernet_token> (Fernet symmetric encryption)."""
    out = text
    for _, pat in _PATTERNS:
        out = pat.sub(
            lambda m: "ENC::" + encrypt_text(m.group(0)),
            out,
        )
    return out


def redact_all_literal(text: str) -> str:
    """Replace every regex PII match with the same type-aware masks as :func:`mask_pii`."""
    return mask_pii(text or "")


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


if __name__ == "__main__":
    original = "My SSN is 123-45-6789"
    encrypted = encrypt_text(original)
    decrypted = decrypt_text(encrypted)
    assert decrypted == original
    print("encrypt/decrypt: OK")
    masked = mask_pii_spans(
        original,
        [
            {"start": 10, "end": 21, "class": "SSN pattern"},
        ],
        mode="partial",
    )
    print("masked:", masked)
