from infer import score_clipboard_with_pii


def main() -> None:
    safe = [
        ("SAFE", "Sprint planning notes for the backend refactor and release timeline."),
        ("SAFE", "def normalize(text): return text.strip().lower()"),
        ("SAFE", "Weekly KPI summary: conversion up by 3 percent quarter over quarter."),
        ("SAFE", "SQL snippet: SELECT id, status FROM tasks WHERE done=0;"),
        ("SAFE", "Please review the architecture diagram and deployment checklist."),
        ("SAFE", "Incident postmortem action items and owner assignments for next sprint."),
        ("SAFE", "Roadmap draft includes mobile parity and API pagination improvements."),
        ("SAFE", "Unit test output: 124 passed, 0 failed, coverage 91 percent."),
        ("SAFE", "Customer feedback theme: onboarding clarity and dashboard discoverability."),
        ("SAFE", "Meeting at 3pm to discuss backlog prioritization and testing plan."),
    ]
    med = [
        ("MED", "Contact me at olivia.lee@contoso.com for the updated report."),
        ("MED", "Call +1 415-555-0139 when the build finishes."),
        ("MED", "Ship docs to 742 Evergreen Terrace Springfield."),
        ("MED", "Primary contact: samir.patel@acme.io and backup at 212-555-0188."),
        ("MED", "Server endpoint 192.168.1.50 needs maintenance window approval."),
        ("MED", "Reach finance desk on +44 20 7946 0958 for invoice follow-up."),
        ("MED", "Please send status update to support@company.org before noon."),
        ("MED", "Forward package to 15 Park Avenue, Austin TX 78701."),
        ("MED", "Notify recruiter at maria.jones@talenthub.com about interview timing."),
        ("MED", "Escalation phone is 646-555-0177 for production incidents."),
        ("MED", "Contact card: devops@infra.net, office line 917-555-0191."),
        ("MED", "Temporary office address is 221B Baker Street London."),
        ("MED", "Partner email is legal.ops@enterprise.com for contract edits."),
        ("MED", "Proxy host 10.10.8.22 should be allowlisted for staging access."),
        ("MED", "Update CRM owner contact at nina.wu@saleshub.io today."),
    ]
    high = [
        ("HIGH", "Employee SSN 123-45-6789 must be redacted."),
        ("HIGH", "Card 4532 1488 0343 6467 failed verification check."),
        ("HIGH", "Transfer funds to IBAN GB29NWBK60161331926819 by EOD."),
        ("HIGH", "Patient MRN: 1234567 diagnosis notes included below."),
        ("HIGH", "Passport number US B98765432 expires in 2028."),
        ("HIGH", "Bank routing 021000021 with account 12345678901 attached."),
        ("HIGH", "Social insurance number 123-456-789 provided for payroll."),
        ("HIGH", "Aadhaar 1234 5678 9012 appears in uploaded form."),
        ("HIGH", "National insurance number AB 12 34 56 C in tax draft."),
        ("HIGH", "Card ending details and CVV 874 should never be pasted."),
        ("HIGH", "SSN formatted as 123 45 6789 appears in this snippet."),
        ("HIGH", "Medical summary includes patient name and treatment ID."),
        ("HIGH", "Wire instructions include routing and beneficiary account values."),
        ("HIGH", "Payment note lists full card number 5425-2334-3010-9903."),
        ("HIGH", "Identity packet includes DOB plus SSN reference."),
    ]
    critical = [
        (
            "CRITICAL",
            "AWS_ACCESS_KEY_ID=AKIA_REDACTED_EXAMPLE AWS_SECRET_ACCESS_KEY=REDACTED_AWS_SECRET_PLACEHOLDER",
        ),
        (
            "CRITICAL",
            "Authorization: Bearer REDACTED.JWT.PAYLOAD",
        ),
        ("CRITICAL", "DB_PASSWORD=Sup3rSecret! DB_HOST=prod.cluster.mongodb.net"),
        ("CRITICAL", "GITHUB_TOKEN=ghp_REDACTED_TOKEN_PLACEHOLDER"),
        (
            "CRITICAL",
            "-----BEGIN RSA PRIVATE KEY----- MIIEpAIBAAKCAQEA2a2rwplBQLz -----END RSA PRIVATE KEY-----",
        ),
        ("CRITICAL", "mongodb://admin:TopSecret123@cluster.mongodb.net/production"),
        ("CRITICAL", "STRIPE_SECRET_KEY=sk_live_REDACTED_SECRET"),
        ("CRITICAL", "api_key = sk-test-REDACTED-PLACEHOLDER"),
        ("CRITICAL", "slack_token = REDACTED_SLACK_TOKEN"),
        ("CRITICAL", "Password credential: passwd=UltraHiddenValue123"),
    ]

    cases = safe + med + high + critical
    expected = {"SAFE": "silent", "MED": "warn", "HIGH": "block", "CRITICAL": "block"}

    passed = 0
    fails = []
    print("Custom 50-phrase matrix (expected action by class)\n")
    for i, (cls, text) in enumerate(cases, 1):
        r = score_clipboard_with_pii(text)
        got = str(r.get("action"))
        exp = expected[cls]
        ok = got == exp
        if ok:
            passed += 1
        print(
            f"{i:02d}. {cls:<8} exp={exp:<6} got={got:<6} "
            f"risk={r.get('risk')} score={r.get('risk_score')}"
        )
        if not ok:
            fails.append(
                (
                    i,
                    cls,
                    exp,
                    got,
                    text,
                    r.get("risk"),
                    r.get("risk_score"),
                    r.get("triggers"),
                )
            )

    print("\nSummary:")
    total = len(cases)
    print(f"  total={total} passed={passed} failed={total - passed} pass_rate={passed / total * 100:.1f}%")
    if fails:
        print("\nMismatches:")
        for f in fails:
            print(
                f"  #{f[0]:02d} {f[1]} exp={f[2]} got={f[3]} risk={f[5]} "
                f"score={f[6]} triggers={f[7]} text={f[4][:90]}"
            )


if __name__ == "__main__":
    main()

