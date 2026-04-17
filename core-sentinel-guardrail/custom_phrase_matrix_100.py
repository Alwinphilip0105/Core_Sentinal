from infer import score_clipboard_with_pii


def build_cases() -> list[tuple[str, str]]:
    cases: list[tuple[str, str]] = []

    for i in range(1, 26):
        if i % 5 == 1:
            text = f"Sprint update {i}: backend refactor tasks and timeline review."
        elif i % 5 == 2:
            text = f"Python snippet {i}: def normalize_{i}(text): return text.strip().lower()"
        elif i % 5 == 3:
            text = f"Dashboard KPI note {i}: churn down, conversion up, no user identifiers included."
        elif i % 5 == 4:
            text = f"SQL sample {i}: SELECT id, status FROM jobs WHERE active = 1;"
        else:
            text = f"Release checklist {i}: QA signoff, docs update, and deployment window."
        cases.append(("SAFE", text))

    for i in range(1, 26):
        if i % 5 == 1:
            text = f"Contact analyst{i}@contoso.com for revised forecast."
        elif i % 5 == 2:
            text = f"Call +1 415-555-{1000 + i:04d} once the pipeline finishes."
        elif i % 5 == 3:
            text = f"Ship package to {100 + i} Park Avenue, Austin TX 7870{i % 10}."
        elif i % 5 == 4:
            text = f"Escalation endpoint 192.168.1.{10 + i} should be monitored."
        else:
            text = f"Primary contact legal.ops{i}@enterprise.com and backup +44 20 7946 09{i:02d}."
        cases.append(("MED", text))

    for i in range(1, 26):
        if i % 5 == 1:
            text = f"Employee SSN 123-45-{6700 + i:04d} requires redaction."
        elif i % 5 == 2:
            text = f"Card 4532 1488 0343 {6400 + i:04d} failed verification."
        elif i % 5 == 3:
            text = "Transfer to IBAN GB29NWBK60161331926819 with beneficiary routing details."
        elif i % 5 == 4:
            text = f"Aadhaar number 1234 5678 {9000 + i:04d} appears in uploaded form."
        else:
            text = (
                f"Identity packet {i} includes DOB and Social insurance number 123-456-78{i % 10}."
            )
        cases.append(("HIGH", text))

    for i in range(1, 26):
        if i % 5 == 1:
            text = (
                "AWS_ACCESS_KEY_ID=AKIA_REDACTED_EXAMPLE "
                "AWS_SECRET_ACCESS_KEY=REDACTED_AWS_SECRET_PLACEHOLDER"
            )
        elif i % 5 == 2:
            text = (
                "Authorization: Bearer "
                "REDACTED.JWT.PAYLOAD"
            )
        elif i % 5 == 3:
            text = f"DB_PASSWORD=UltraSecret{i}! DB_HOST=prod.cluster.mongodb.net"
        elif i % 5 == 4:
            text = f"GITHUB_TOKEN=ghp_REDACTED_TOKEN_{i:02d}"
        else:
            text = f"STRIPE_SECRET_KEY=sk_live_REDACTED_SECRET_{i:02d}"
        cases.append(("CRITICAL", text))

    return cases


def main() -> None:
    cases = build_cases()
    expected = {"SAFE": "silent", "MED": "warn", "HIGH": "block", "CRITICAL": "block"}
    passed = 0
    mismatches: list[tuple[int, str, str, str, str, str | None, int | None, list | None]] = []

    print("Custom 100-phrase matrix (expected action by class)\n")
    for i, (cls, text) in enumerate(cases, 1):
        r = score_clipboard_with_pii(text)
        got = str(r.get("action"))
        exp = expected[cls]
        ok = got == exp
        if ok:
            passed += 1
        print(
            f"{i:03d}. {cls:<8} exp={exp:<6} got={got:<6} "
            f"risk={r.get('risk')} score={r.get('risk_score')}"
        )
        if not ok:
            mismatches.append(
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

    total = len(cases)
    print("\nSummary:")
    print(f"  total={total} passed={passed} failed={total - passed} pass_rate={passed / total * 100:.1f}%")
    if mismatches:
        print("\nMismatches:")
        for m in mismatches:
            print(
                f"  #{m[0]:03d} {m[1]} exp={m[2]} got={m[3]} "
                f"risk={m[5]} score={m[6]} triggers={m[7]} text={m[4][:100]}"
            )


if __name__ == "__main__":
    main()

