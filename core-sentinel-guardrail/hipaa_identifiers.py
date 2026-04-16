"""
HIPAA Safe Harbor identifiers and current coverage checklist.

This module is intentionally data-only so detection code can reference the
checklist without embedding the full Safe Harbor inventory inline.
"""

HIPAA_SAFE_HARBOR_IDENTIFIERS = [
    {
        "id": 1,
        "identifier": "Names",
        "coverage": "existing",
        "check": False,
        "notes": "Model and regex-adjacent heuristics may detect personal names.",
    },
    {
        "id": 2,
        "identifier": "Geographic subdivisions smaller than a state",
        "coverage": "existing",
        "check": False,
        "notes": "Addresses and location-like text are partially covered.",
    },
    {
        "id": 3,
        "identifier": "All elements of dates except year directly related to an individual",
        "coverage": "partial",
        "check": False,
        "notes": "Not comprehensively enumerated by regex in this checklist.",
    },
    {
        "id": 4,
        "identifier": "Telephone numbers",
        "coverage": "existing",
        "check": False,
        "notes": "Covered by phone regex rules.",
    },
    {
        "id": 5,
        "identifier": "Fax numbers",
        "coverage": "partial",
        "check": False,
        "notes": "May be inferred from phone-like patterns but not explicitly tagged.",
    },
    {
        "id": 6,
        "identifier": "Email addresses",
        "coverage": "existing",
        "check": False,
        "notes": "Covered by email regex rules.",
    },
    {
        "id": 7,
        "identifier": "Social Security numbers",
        "coverage": "existing",
        "check": False,
        "notes": "Covered by SSN regex rules.",
    },
    {
        "id": 8,
        "identifier": "Medical record numbers",
        "coverage": "new_regex",
        "check": False,
        "notes": "Covered by HIPAA MRN regex with explicit MRN/Medical Record prefix.",
    },
    {
        "id": 9,
        "identifier": "Health plan beneficiary numbers",
        "coverage": "new_regex",
        "check": False,
        "notes": "Covered by Member/Beneficiary/Policy-prefixed identifier regex.",
    },
    {
        "id": 10,
        "identifier": "Account numbers",
        "coverage": "partial",
        "check": False,
        "notes": "Some financial/account-like identifiers are already covered elsewhere.",
    },
    {
        "id": 11,
        "identifier": "Certificate or license numbers",
        "coverage": "new_regex",
        "check": False,
        "notes": "DEA number regex covers a common healthcare license identifier.",
    },
    {
        "id": 12,
        "identifier": "Vehicle identifiers and serial numbers, including license plate numbers",
        "coverage": "missing",
        "check": False,
        "notes": "Not explicitly covered by this change.",
    },
    {
        "id": 13,
        "identifier": "Device identifiers and serial numbers",
        "coverage": "partial",
        "check": False,
        "notes": "Some device-like tokens may be caught by broader heuristics.",
    },
    {
        "id": 14,
        "identifier": "Web URLs",
        "coverage": "missing",
        "check": False,
        "notes": "Not explicitly covered by this change.",
    },
    {
        "id": 15,
        "identifier": "Internet Protocol (IP) address numbers",
        "coverage": "existing",
        "check": False,
        "notes": "Covered by IPv4 and IPv6 regex rules.",
    },
    {
        "id": 16,
        "identifier": "Biometric identifiers, including finger and voice prints",
        "coverage": "missing",
        "check": False,
        "notes": "Not explicitly covered by this change.",
    },
    {
        "id": 17,
        "identifier": "Full-face photographic images and comparable images",
        "coverage": "missing",
        "check": False,
        "notes": "Out of scope for text regex detection.",
    },
    {
        "id": 18,
        "identifier": "Any other unique identifying number, characteristic, or code",
        "coverage": "partial",
        "check": False,
        "notes": "NPI and health-plan style identifiers broaden coverage here.",
    },
]
