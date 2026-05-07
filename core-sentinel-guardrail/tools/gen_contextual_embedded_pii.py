#!/usr/bin/env python3
"""
Generate contextual embedded PII training lines (synthetic placeholders only).

Augments Enron-style email tone with customer-service and legal snippets where PII is
referenced in prose rather than as lone tokens.

Run from core-sentinel-guardrail/:
  python tools/gen_contextual_embedded_pii.py

Writes data/extra_pools/contextual_embedded_pii.jsonl (committed when rebuilt).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

EXAMPLES: list[tuple[str, str]] = [
    # --- Business email (med / high contextual) ---
    (
        "Hi Avery — following up on the Riverside renewal. Finance asked me to confirm the "
        "subscriber ID printed on last month's PDF (looks like HB-77492-Q) matches what you "
        "have on file. Ping me before COB Thursday.",
        "med",
    ),
    (
        "Teams, please route invoice INV-2024-99821 to procurement; vendor contact is listed "
        "under our vendor portal login — do not paste credentials in chat. Payroll batch "
        "closes Friday 5pm Central.",
        "med",
    ),
    (
        "Connor — Legal wants the diligence zip password-shared out of band. Include the draft "
        "APA exhibit listing customer account ranges (masked except last four) only on the "
        "secure share link.",
        "high",
    ),
    (
        "Morning shift lead: escalated ticket EZ-445901 — customer insists the refund hit the "
        "wrong card ending 4421; please verify BIN and last-four against CRM note from 03/18.",
        "med",
    ),
    (
        "Sarah, attach the SOC2 summary for Harbor Dental but strip employee directories; "
        "recipient list should stay to external counsel only.",
        "med",
    ),
    (
        "Per ops: warehouse dock badge reader failed at Gate 7; contractor badge RF-88301 was "
        "cloned twice yesterday — facilities opened an incident INC-77432.",
        "med",
    ),
    (
        "Reminder: quote response for Midtown Clinic must reference MRN prefixes only (no "
        "full identifiers). Lab interface spec is under \\shared\\clinical\\interfaces\\.",
        "high",
    ),
    (
        "Jordan — client wants W-9 on file before wire; EIN goes in encrypted fields only per "
        "Treasury desk policy from last quarter.",
        "med",
    ),
    (
        "We accidentally CC'd the pilot cohort list in the marketing thread; please recall "
        "message id <MSG-20260214-aa12> and send a corrected version without personal emails.",
        "high",
    ),
    (
        "AP team: payment file for batch P-90912 failed tokenization; re-run after key rotation "
        "per security — no plaintext card data in Slack.",
        "high",
    ),
    # --- Customer service / chat transcript style ---
    (
        "Agent: Thanks for contacting NovaShop. Cust: Hi, order #889221 says delivered but "
        "nothing arrived. Agent: What's the ship-to postal code on the confirmation? Cust: "
        "60614. Agent: Seeing a carrier scan at locker LK-993; want a resend?",
        "med",
    ),
    (
        "Cust: Can't log into my brokerage — says 2FA device mismatch. Support: We'll send a "
        "one-time link to the address ending @westmail.test after you confirm DOB verbally "
        "(not typed here). Cust: Okay.",
        "med",
    ),
    (
        "Agent: Warranty claim WC-774100 approved; technician will bring replacement SN "
        "prefix NV-884x. Cust: Apartment buzzer broken — call my mobile on file.",
        "med",
    ),
    (
        "Live chat escalation: abusive language + repeated requests for another user's loyalty "
        "ID. Close ticket CS-993812; flag fraud queue with session hash only.",
        "med",
    ),
    (
        "Voice summary: Caller verified last four SSN ***********4412 for address change — "
        "entered DOB mismatch once; CSR applied manual review code MR-551.",
        "high",
    ),
    # --- Legal correspondence fragments ---
    (
        "Pursuant to your records request dated 02/05/2026, please preserve documents regarding "
        "Employee ID sequences 884100–884190 and any workstation assignment logs through Q1.",
        "med",
    ),
    (
        "Counsel requests redaction of exhibits A–C containing unredacted driver's license "
        "numbers prior to disclosure to opposing party under protective order stipulation.",
        "high",
    ),
    (
        "Subpoena response draft: omit patient names; cite internal matter HL-77432 only. "
        "Coordinate with HIPAA counsel before production.",
        "high",
    ),
    (
        "Settlement wire instructions are updated; IBAN suffix must match engagement letter EL-"
        "2025-884. Do not circulate Swift details by email.",
        "high",
    ),
    (
        "Deposition postponed; witness subpoena envelope returned — verify service address "
        "matches county assessor parcel APN 441-992-881.",
        "med",
    ),
    (
        "Plaintiff's counsel attached medical billing codes without member ID masking; opposing "
        "counsel emailed objection within two hours.",
        "med",
    ),
    # --- Exec / HR / onboarding context ---
    (
        "Onboarding buddy: remind new hire to upload passport copy to Workday sandbox task "
        "ONB-9931 — SSO not active until badge prints at HQ reception.",
        "med",
    ),
    (
        "HRBP: Salary band discussion for Req R-88421 should stay in Talent Room; spreadsheet "
        "with employee numbers was shared in error — delete local copies.",
        "high",
    ),
    (
        "IT: laptop shipment for remote employee — ship to verified address on file in HRIS "
        "only; courier label must not show internal cost center codes.",
        "med",
    ),
    (
        "Benefits: dependent verification due for enrollee badges ending in •••884; upload "
        "birth certificate PDF through encrypted portal.",
        "med",
    ),
    (
        "Investigation memo: screenshots show Slack DM containing customer IBAN pasted in "
        "cleartext; preserve thread TS-993812 and revoke tokens for workspace WS-northwind.",
        "high",
    ),
    # --- Engineering / SaaS incidental PII ---
    (
        "SRE:Canary failed — sample log line includes user email duplicated in breadcrumb field; "
        "scrub before posting to observability notebook NB-884.",
        "med",
    ),
    (
        "Pager dump shows stack trace referencing internal user LDAP cn=jdoe884,ou=Finance — "
        "sanitize repro steps before attaching to public issue #4412.",
        "med",
    ),
    (
        "Incident IR-993: exported PDF from analytics includes device serial list — rotate "
        "share link and notify DPO mailbox only.",
        "high",
    ),
    (
        "Support bundle from customer mistakenly contains plaintext Wi-Fi PSK and home VoIP "
        "CID; delete attachment and regenerate with masking profile P2.",
        "high",
    ),
    (
        "Cron job mailed CSV attachment with partner API keys embedded in headers — revoked "
        "keys K_partner_993x and invalidated download URLs.",
        "high",
    ),
    # --- Benign parallels (hard negatives-ish, low risk) ---
    (
        "All-hands recap: roadmap slides are safe to share externally after marketing review; "
        "no customer stories or numbered incidents included.",
        "low",
    ),
    (
        "Facilities reminder: badges must be visibly worn Mon–Fri; visitor escort policy "
        "unchanged. Report broken readers to the helpdesk form only.",
        "low",
    ),
    (
        "Quarterly phishing drill results: click rate dropped; report suspicious messages via "
        "the Phish Alert button rather than screenshots in group chat.",
        "low",
    ),
    (
        "Parking garage level B will close for lighting maintenance Saturday midnight–4am — "
        "no access changes to elevators.",
        "low",
    ),
    (
        "Reminder: cafeteria menu updates post on Friday; allergens list is informational only.",
        "low",
    ),
    # --- More business / thread-style embedded references ---
    (
        "Please do not forward chain; original sender included a CSV with partner employee IDs "
        "in column D — delete thread from mailboxes and purge from litigation hold folder LH-221.",
        "high",
    ),
    (
        "Regional sales: attach territory map only; customer ship-to list with full addresses "
        "must stay in CRM export behind VPN group VPN-ops-only.",
        "med",
    ),
    (
        "Audit sample wants five random policyholders with claim numbers redacted to CLM-****; "
        "actuary needs DOB year only, not day.",
        "med",
    ),
    (
        "Slack thread #proj-nova leaked a photo of a driver's license on a monitor — remove "
        "message and open privacy ticket PRIV-8831.",
        "high",
    ),
    (
        "Travel desk: itinerary PDF shows passport number in header; reissue using template T-"
        "travel-v3 with ID fields blanked.",
        "high",
    ),
    (
        "Partner API doc example used a fake JWT in code block; reviewer noted it matches a "
        "revoked dev token pattern — replace with obvious placeholder string.",
        "med",
    ),
    (
        "Collections email template still had previous debtor name in greeting — compliance "
        "wants versioning disabled until QA signs off.",
        "med",
    ),
    (
        "Call center QA: evaluator flagged Agent 12 reading full card number aloud; coaching "
        "session mandatory, reference QA-202603-14.",
        "high",
    ),
    (
        "Biometrics pilot: kiosk stored face templates without consent wording v2; halt rollout "
        "to stores ST-884 and ST-885 pending legal.",
        "high",
    ),
    (
        "Newsletter draft hyperlinked to unsecured Google Sheet with donor emails column "
        "unhidden; unsubscribe link test only on staging list.",
        "high",
    ),
    (
        "Contractor onboarding: badge photo filename included national ID digits — rename BR-88421 "
        "batch and sanitize EXIF.",
        "high",
    ),
    (
        "Helpdesk note: VIP caller referenced \"the same fax as last lawsuit\" — do not annotate "
        "fax number in ticketing system; escalate to paralegal pool.",
        "med",
    ),
    (
        "Data science: Jupyter output shows kernel variable with API secret string; purge "
        "outputs before checking notebook into repo branch ds/forecast-v2.",
        "high",
    ),
]


def main() -> None:
    from data import MAX_TEXT_LEN, MIN_TEXT_LEN

    out = _ROOT / "data" / "extra_pools" / "contextual_embedded_pii.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    n_bad = 0
    rows: list[dict] = []
    for text, risk in EXAMPLES:
        t = text.strip()
        if len(t) < MIN_TEXT_LEN or len(t) > MAX_TEXT_LEN:
            n_bad += 1
            continue
        rows.append(
            {
                "text": t,
                "risk": risk,
                "label": "CONTEXT_EMBED",
                "source": "synthetic_contextual_embedded_pii",
            }
        )
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Written {len(rows)} lines to {out} (skipped length: {n_bad})")


if __name__ == "__main__":
    main()
