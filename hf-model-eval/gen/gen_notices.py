#!/usr/bin/env python3
"""Synthetic IRS notice classification set for area C (small instruction models -> strict JSON).

Writes to data/notices/:
    notice01.txt .. notice25.txt     plain-text IRS-style notice letters (120-320 words each)
    ground_truth.json                keyed "notice01".. with EXACTLY the eval/fields.json notice_fields
                                     (notice_type, tax_year, notice_date, amount_due, response_deadline,
                                     action_required, urgency) plus an unscored "rubric_reason" string.

Everything is synthetic: invented taxpayers, invented businesses, synthetic EINs, SSNs printed masked as
XXX-XX-NNNN (last 4 visible) the way redacted client copies look. The IRS letterhead, notice numbers and
public IRS phone lines are used for realism only. Notice n is seeded with random.Random(n), so reruns are
byte-identical. Run:  python gen/gen_notices.py

Canonical ground-truth formats (eval/fields.json): money "4812.37" ("0.00" when the notice asks for no
payment), dates "YYYY-MM-DD", tax_year "2025", codes UPPERCASE, response_deadline "NONE" when the notice
sets no deadline for the taxpayer.

Deadline styles (about half each, see SPECS):
    explicit  - the letter states the date ("respond by March 16, 2026"); ground truth = that date
    relative  - the letter states a window ("within 30 days from the date of this notice");
                ground truth = notice_date + N calendar days (datetime arithmetic, NO business-day
                adjustment, exactly as the eval prompt instructs the model to compute it)
    none      - the notice sets no deadline for the taxpayer -> "NONE"

RUBRIC (applied in code via RUBRIC[notice_type]; identical to NOTICE_RUBRIC in eval/prompts.py):
    urgency   CRITICAL  LT11, CP90, CP504, CP3219A              (levy / seizure / Tax Court clock)
              HIGH      CP2000, CP2501, CP14, CP503, CP161, CP22A, CP23, CP75
              MEDIUM    4883C, 5071C, CP501, CP11, CP259, CP71C, CP05
              LOW       CP49, CP12, CP21B
    action    PAY                        CP14, CP501, CP503, CP504, CP161, CP22A, CP23, CP71C, CP11
              RESPOND_WITH_DOCUMENTATION CP2000, CP2501, CP75
              REQUEST_HEARING            LT11, CP90 (CDP hearing, Form 12153), CP3219A (Tax Court petition)
              VERIFY_IDENTITY            4883C, 5071C
              FILE_RETURN                CP259
              NO_ACTION                  CP49, CP12, CP21B, CP05
    amount_due is the amount the notice asks the taxpayer to pay ("0.00" for refunds, holds, identity
    letters, CP2501 initial contact, CP259 missing return).  Refund amounts are never amount_due.
    Deliberate hard cases: CP05 mentions the IRS's own 60-day review window but sets no taxpayer deadline
    (NONE); CP71C has an amount due but "pay immediately" and no new due date (NONE); CP12 is a refund
    (0.00, NO_ACTION) whose only deadline is the 60-day window to dispute the change.
"""
import json
import os
import random
import re
import sys
from datetime import date, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "data", "notices")
GT_PATH = os.path.join(OUT_DIR, "ground_truth.json")
with open(os.path.join(ROOT, "eval", "fields.json")) as fh:
    _F = json.load(fh)
NOTICE_FIELDS = _F["notice_fields"]
ENUMS = _F["notice_enums"]

# --------------------------------------------------------------------------- rubric (see docstring)
URGENCY = {
    "CRITICAL": ["LT11", "CP90", "CP504", "CP3219A"],
    "HIGH": ["CP2000", "CP2501", "CP14", "CP503", "CP161", "CP22A", "CP23", "CP75"],
    "MEDIUM": ["4883C", "5071C", "CP501", "CP11", "CP259", "CP71C", "CP05"],
    "LOW": ["CP49", "CP12", "CP21B"],
}
ACTION = {
    "PAY": ["CP14", "CP501", "CP503", "CP504", "CP161", "CP22A", "CP23", "CP71C", "CP11"],
    "RESPOND_WITH_DOCUMENTATION": ["CP2000", "CP2501", "CP75"],
    "REQUEST_HEARING": ["LT11", "CP90", "CP3219A"],
    "VERIFY_IDENTITY": ["4883C", "5071C"],
    "FILE_RETURN": ["CP259"],
    "NO_ACTION": ["CP49", "CP12", "CP21B", "CP05"],
}
RUBRIC = {}
for _u, _types in URGENCY.items():
    for _t in _types:
        RUBRIC[_t] = {"urgency": _u}
for _a, _types in ACTION.items():
    for _t in _types:
        RUBRIC[_t]["action_required"] = _a
assert all(set(v) == {"urgency", "action_required"} for v in RUBRIC.values()), "rubric incomplete"
assert set(RUBRIC) <= set(ENUMS["notice_type"]), "rubric type not in fields.json enum"

# Relative response windows in calendar days by notice type (ground truth = notice_date + days).
WINDOW_DAYS = {
    "CP2000": 30, "CP2501": 30, "CP14": 21, "CP501": 21, "CP503": 10, "CP504": 30, "LT11": 30,
    "CP12": 60, "CP11": 21, "CP22A": 21, "CP23": 21, "CP161": 21, "CP75": 30, "4883C": 30,
    "5071C": 30, "CP3219A": 90, "CP90": 30, "CP259": 30,
}
# (doc number, notice type, deadline style). 22 distinct types; 3 repeats (CP2000, CP14, LT11).
SPECS = [
    (1, "CP2000", "explicit"), (2, "CP2501", "relative"), (3, "CP14", "relative"), (4, "CP501", "explicit"),
    (5, "CP503", "relative"), (6, "CP504", "explicit"), (7, "LT11", "relative"), (8, "CP49", "none"),
    (9, "CP12", "relative"), (10, "CP11", "explicit"), (11, "CP22A", "explicit"), (12, "CP23", "relative"),
    (13, "CP161", "explicit"), (14, "CP75", "relative"), (15, "4883C", "relative"), (16, "5071C", "explicit"),
    (17, "CP3219A", "relative"), (18, "CP90", "explicit"), (19, "CP259", "relative"), (20, "CP71C", "none"),
    (21, "CP21B", "none"), (22, "CP05", "none"), (23, "CP2000", "relative"), (24, "CP14", "explicit"),
    (25, "LT11", "explicit"),
]
LETTER_TYPES = {"4883C", "5071C"}          # IRS "Letter" (LTR) rather than "Notice CP..."
BUSINESS_TYPES = {"CP161", "CP259"}        # addressed to an entity; header shows a full synthetic EIN
PHONE = {"CP2000": "800-829-8310", "CP2501": "800-829-8310", "CP3219A": "800-829-8310",
         "CP14": "800-829-0922", "CP501": "800-829-0922", "CP503": "800-829-0922", "CP11": "800-829-0922",
         "CP12": "800-829-0922", "CP22A": "800-829-0922", "CP23": "800-829-0922", "CP49": "800-829-0922",
         "CP21B": "800-829-0922", "CP71C": "800-829-0922", "CP504": "800-829-3903", "LT11": "800-829-3903",
         "CP90": "800-829-3903", "4883C": "800-830-5084", "5071C": "800-830-5084",
         "CP161": "800-829-0115", "CP259": "800-829-0115", "CP75": "800-829-1040", "CP05": "800-829-1040"}
CENTERS = ["Internal Revenue Service, Austin, TX 73301-0023", "Internal Revenue Service, Ogden, UT 84201-0038",
           "Internal Revenue Service, Kansas City, MO 64999-0025", "Internal Revenue Service, Fresno, CA 93888-0025",
           "Internal Revenue Service, Philadelphia, PA 19255-0025", "Internal Revenue Service, Cincinnati, OH 45999-0039"]

# --------------------------------------------------------------------------- synthetic identities
FIRST = ["MARGARET", "DANIEL", "PRIYA", "THOMAS", "ELENA", "MARCUS", "SOPHIA", "RAYMOND", "NADIA", "JULIAN",
         "HELEN", "VICTOR", "BEATRIZ", "OWEN", "LORRAINE", "KWAME", "INGRID", "TOBIAS"]
LAST = ["OKAFOR", "WHITFIELD", "RAMIREZ", "LINDQVIST", "CASTELLANO", "NAKAMURA", "BRENNAN", "DESAI", "HOLLOWAY",
        "FITZGERALD", "MORAVEC", "ABERNATHY", "VANCE", "OYELARAN", "SZABO", "PELLETIER"]
STREETS = ["WILLOW CREST DR", "SYCAMORE LN", "HARBOR POINT RD", "MAPLE RIDGE CT", "FOXGLOVE AVE",
           "CANYON VIEW BLVD", "OLD MILL RD", "LAKESHORE DR", "PRAIRIE WIND WAY", "BIRCHWOOD TER", "QUARRY HILL RD"]
CITIES = [("SPRINGFIELD", "MO", "65807"), ("TAMPA", "FL", "33602"), ("DAYTON", "OH", "45414"), ("BOISE", "ID", "83702"),
          ("ALBUQUERQUE", "NM", "87102"), ("RALEIGH", "NC", "27601"), ("TUCSON", "AZ", "85701"),
          ("MADISON", "WI", "53703"), ("RENO", "NV", "89501"), ("CHATTANOOGA", "TN", "37402"),
          ("PEORIA", "IL", "61602"), ("SPOKANE", "WA", "99201")]
ENTITIES = ["NORTHGATE LOGISTICS LLC", "CEDAR RIDGE DENTAL PARTNERS LLC", "PINECREST AUTOMOTIVE INC",
            "KESTREL HVAC SERVICES INC", "BLUE HERON PROPERTIES LP", "MARLOWE DESIGN STUDIO INC",
            "SUMMIT CREEK HOLDINGS LP", "HALVORSEN PRICE CONSULTING LLC"]
PAYERS = [("VELOCITY MARKETING PARTNERS LLC", "1099-NEC", "nonemployee compensation"),
          ("FIRST MERIDIAN BANK", "1099-INT", "interest income"),
          ("HARBORVIEW SECURITIES LLC", "1099-B", "proceeds from broker transactions"),
          ("CASCADE COMMUNITY CREDIT UNION", "1099-INT", "interest income"),
          ("SILVERLINE CONSULTING GROUP INC", "1099-NEC", "nonemployee compensation"),
          ("REDSTONE PAYMENTS INC", "1099-K", "payment card and third party network transactions"),
          ("GRANITE PEAK RETIREMENT TRUST", "1099-R", "distributions from pensions, annuities, retirement plans")]


def person(rng):
    first, last = rng.choice(FIRST), rng.choice(LAST)
    mi = rng.choice(["", "", rng.choice("ABCDEJKLMRST")])
    city, state, z = rng.choice(CITIES)
    return {"name": " ".join(x for x in (first, mi, last) if x),
            "street": f"{rng.randint(100, 9899)} {rng.choice(STREETS)}",
            "city_line": f"{city} {state} {z}", "ssn4": f"{rng.randint(0, 9999):04d}"}


def entity(rng):
    city, state, z = rng.choice(CITIES)
    return {"name": rng.choice(ENTITIES), "street": f"{rng.randint(100, 9899)} {rng.choice(STREETS)}",
            "city_line": f"{city} {state} {z}",
            "ein": f"{rng.choice([82, 83, 84, 85, 86, 87, 88, 92, 93, 94, 95])}-{rng.randint(1000000, 9999999)}"}


# --------------------------------------------------------------------------- formatting
def money(cents):
    """Printed form with $ and thousands separators: $4,812.37"""
    return f"${cents // 100:,}.{cents % 100:02d}"


def canon(cents):
    """Ground-truth canonical money: 4812.37"""
    return f"{cents // 100}.{cents % 100:02d}"


def long_date(d):
    return f"{d.strftime('%B')} {d.day}, {d.year}"


def cents(rng, lo, hi):
    return rng.randint(lo * 100, hi * 100)


def dl(ctx, rng, verb_by="by"):
    """Deadline phrase. explicit -> 'by March 16, 2026'; relative -> a varied 'within N days ...' phrase."""
    if ctx["style"] == "explicit":
        return f"{verb_by} {long_date(ctx['deadline'])}"
    n = ctx["days"]
    doc = "letter" if ctx["ntype"] in LETTER_TYPES else "notice"
    return rng.choice([
        f"within {n} days from the date of this {doc}",
        f"within {n} days of the date of this {doc}",
        f"no later than {n} days from the date of this {doc}",
        f"by the due date, which is {n} days from the {doc} date",
    ])


# --------------------------------------------------------------------------- header / footer
def header(ctx):
    t, letter = ctx["ntype"], ctx["ntype"] in LETTER_TYPES
    rows = [("Letter number" if letter else "Notice", f"LTR {t}" if letter else t),
            ("Tax year", ctx["tax_year"]),
            ("Letter date" if letter else "Notice date", long_date(ctx["notice_date"]))]
    if t in BUSINESS_TYPES:
        rows.append(("Employer ID number", ctx["tp"]["ein"]))
    else:
        rows.append(("Social Security number", f"XXX-XX-{ctx['tp']['ssn4']}"))
    rows.append(("To contact us", f"Phone {PHONE[t]}"))
    if ctx.get("caller_id"):
        rows.append(("Your caller ID", ctx["caller_id"]))
    rows.append(("Page", f"1 of {ctx['pages']}"))
    lines = ["Department of the Treasury - Internal Revenue Service", ctx["center"], ""]
    lines += [f"{k:<24}{v}" for k, v in rows]
    lines += ["", ctx["tp"]["name"], ctx["tp"]["street"], ctx["tp"]["city_line"], ""]
    return lines


def footer(ctx):
    t = ctx["ntype"]
    url = f"www.irs.gov/{t.lower()}" if t not in LETTER_TYPES else "www.irs.gov/identitytheft"
    return ["", f"Visit {url} for more information about this {'letter' if t in LETTER_TYPES else 'notice'}. "
                f"Keep this {'letter' if t in LETTER_TYPES else 'notice'} with your {ctx['tax_year']} tax records."]


def billing(rows):
    out = ["", "Billing summary"]
    out += [f"  {label:<42}{money(c):>14}" for label, c in rows]
    return out


# --------------------------------------------------------------------------- templates (body only)
def t_cp2000(ctx, rng):
    ty, payer, form, desc = ctx["tax_year"], *ctx["payer"]
    amt, tax, pen, intr = ctx["unreported"], ctx["tax"], ctx["pen"], ctx["intr"]
    return [
        f"Proposed changes to your {ty} Form 1040",
        f"Proposed amount due: {money(ctx['amount'])}",
        "",
        f"The income and payment information we have on file from third parties doesn't match the information you "
        f"reported on your {ty} Form 1040. If our information is correct, you will owe {money(ctx['amount'])}, "
        f"including penalty and interest, which you need to pay {dl(ctx, rng)}.",
        "", "Summary of proposed changes",
        f"  {payer} (Form {form}, {desc}) reported to IRS: {money(amt)}; shown on your return: $0.00",
        f"  Increase in tax: {money(tax)}",
        f"  Accuracy-related penalty (IRC section 6662): {money(pen)}",
        f"  Interest computed to the date of this notice: {money(intr)}",
        f"  Proposed amount due: {money(ctx['amount'])}",
        "",
        "This is not a bill. It is a proposal based on information reported to us.",
        "", "What you need to do",
        f"Review the proposed changes and compare them with your records. Complete the enclosed Response form and "
        f"return it to us {dl(ctx, rng)}. If you agree, sign the form and include your payment of "
        f"{money(ctx['amount'])}. If you disagree, check the box that applies, attach a signed statement explaining "
        f"why, and include copies of supporting documents such as corrected Forms 1099 or cost basis records.",
        "",
        f"If you don't respond {dl(ctx, rng)}, we will send you a Statutory Notice of Deficiency.",
        "", f"If you have questions, call us at {PHONE['CP2000']}. Have this notice and your {ty} return available "
            f"when you call.",
    ]


def t_cp2501(ctx, rng):
    ty, payer, form, desc = ctx["tax_year"], *ctx["payer"]
    return [
        f"Information doesn't match your {ty} tax return",
        "",
        f"We received information from third parties about income or payments that doesn't match what you reported "
        f"on your {ty} Form 1040. We haven't proposed any changes to your tax yet. This notice asks you to explain "
        f"the difference before we do.",
        "", "Information reported to us that isn't on your return",
        f"  {payer}, Form {form} ({desc}): {money(ctx['unreported'])}",
        "", "What you need to do",
        f"Respond {dl(ctx, rng)}. Complete the enclosed Response form, explain why the information above differs "
        f"from your return, and attach copies of documents that support your explanation, such as a corrected "
        f"Form {form}, your brokerage statements, or your general ledger. If you agree that the income was left off "
        f"your return, tell us so on the Response form and we will send you a notice showing the additional tax, "
        f"penalty, and interest you owe.",
        "",
        "If you don't respond, we may propose changes to your return based on the information we received and send "
        "you a notice showing the amount you owe.",
        "",
        "You don't need to send a payment with your response, and you don't need to file an amended return. Don't "
        "send original documents; we can't return them.",
        "", f"If you have questions, call us at {PHONE['CP2501']} and have this notice available.",
    ]


def t_cp14(ctx, rng):
    ty = ctx["tax_year"]
    return [
        f"You have unpaid taxes for {ty}",
        f"Amount due: {money(ctx['amount'])}",
        "",
        f"Our records show you have unpaid taxes on your {ty} Form 1040. Pay the amount due {dl(ctx, rng)} to avoid "
        f"additional penalty and interest charges.",
        *billing([("Tax you owed", ctx["tax"]), ("Failure-to-pay penalty", ctx["pen"]),
                  ("Interest charges", ctx["intr"]), ("Amount due", ctx["amount"])]),
        "", "What you need to do",
        f"Pay {money(ctx['amount'])} {dl(ctx, rng)}. Pay online at www.irs.gov/payments using Direct Pay from your "
        f"bank account, or send a check or money order payable to the United States Treasury with the payment stub "
        f"from this notice. Write your Social Security number, the tax year ({ty}), and the notice number (CP14) on "
        f"your payment.",
        "",
        "If you can't pay the full amount now, you may qualify for a payment plan. Apply online at www.irs.gov/opa "
        "or call us. We will continue to charge interest and the failure-to-pay penalty until the balance is paid "
        "in full.",
        "", f"If you already paid this amount, or if you think we made a mistake, call us at {PHONE['CP14']} with "
            f"your payment information.",
    ]


def t_cp501(ctx, rng):
    ty = ctx["tax_year"]
    return [
        f"Reminder: you have unpaid taxes for {ty}",
        f"Amount due: {money(ctx['amount'])}",
        "",
        f"You have an unpaid balance on your {ty} Form 1040. We previously sent you a notice about this balance and "
        f"haven't received payment. This is a reminder of the amount you owe, including penalty and interest "
        f"computed to the date of this notice.",
        *billing([("Amount you owed", ctx["tax"]), ("Failure-to-pay penalty", ctx["pen"]),
                  ("Interest charges", ctx["intr"]), ("Amount due", ctx["amount"])]),
        "", "What you need to do",
        f"Pay {money(ctx['amount'])} {dl(ctx, rng)}. Pay online at www.irs.gov/payments, or mail a check or money "
        f"order payable to the United States Treasury with the payment stub. Include your Social Security number, "
        f"tax year {ty}, and notice number CP501 on your payment.",
        "",
        f"If you don't pay {dl(ctx, rng)}, we will send you a further notice and penalty and interest will continue "
        f"to increase. If you can't pay in full, you can request an installment agreement online at www.irs.gov/opa "
        f"or by filing Form 9465.",
        "", f"If you already paid, allow 21 days for your payment to post. If you disagree with the amount, call us "
            f"at {PHONE['CP501']} and have this notice and your payment records available.",
    ]


def t_cp503(ctx, rng):
    ty = ctx["tax_year"]
    return [
        f"Second reminder: Immediate action required - you have unpaid taxes for {ty}",
        f"Amount due: {money(ctx['amount'])}",
        "",
        f"We haven't received a response to the notices we sent you about your unpaid {ty} Form 1040 balance. You "
        f"must pay the amount due {dl(ctx, rng)}. Penalty and interest continue to accrue until you pay in full.",
        *billing([("Amount you owed", ctx["tax"]), ("Failure-to-pay penalty", ctx["pen"]),
                  ("Interest charges", ctx["intr"]), ("Amount due", ctx["amount"])]),
        "", "What you need to do",
        f"Pay {money(ctx['amount'])} {dl(ctx, rng)}. Pay online at www.irs.gov/payments, or send a check or money "
        f"order payable to the United States Treasury with the payment stub from this notice.",
        "",
        f"If you don't pay or contact us {dl(ctx, rng)}, we may file a Notice of Federal Tax Lien, which is a public "
        f"claim against your property and can affect your credit. If you can't pay in full, call us immediately to "
        f"discuss a payment plan or other options.",
        "", f"If you already paid this balance or you disagree with it, call us at {PHONE['CP503']} with a copy of "
            f"this notice and your payment records.",
    ]


def t_cp504(ctx, rng):
    ty = ctx["tax_year"]
    return [
        "Notice of intent to seize (levy) your state tax refund or other property",
        f"Amount due immediately: {money(ctx['amount'])}",
        "",
        f"You have an unpaid balance for tax year {ty}. We have sent you prior notices and have not received payment. "
        f"If you don't pay the amount due {dl(ctx, rng)}, we may seize (levy) your state income tax refund and apply "
        f"it to the amount you owe. If the refund doesn't cover it, we may seize other property or rights to "
        f"property, including wages, bank accounts, and certain federal payments.",
        *billing([("Amount you owed", ctx["tax"]), ("Failure-to-pay penalty", ctx["pen"]),
                  ("Interest charges", ctx["intr"]), ("Amount due", ctx["amount"])]),
        "", "What you need to do",
        f"Pay {money(ctx['amount'])} {dl(ctx, rng)}. Pay online at www.irs.gov/payments, or mail a check or money "
        f"order payable to the United States Treasury with the enclosed payment stub.",
        "",
        f"If you can't pay the full amount {dl(ctx, rng)}, call us immediately at {PHONE['CP504']} to discuss "
        f"options such as an installment agreement or a temporary delay of collection. If you have already paid, "
        f"send proof of payment to the address at the top of this notice.",
    ]


def t_lt11(ctx, rng):
    ty = ctx["tax_year"]
    return [
        "Final Notice of Intent to Levy and Notice of Your Right to a Hearing",
        f"Amount due: {money(ctx['amount'])}",
        "",
        f"We previously asked you to pay the federal tax you owe for tax year {ty}, but we still haven't received "
        f"your payment. This is your notice of our intent to levy under Internal Revenue Code section 6331 and your "
        f"right to a hearing under section 6330.",
        *billing([("Unpaid tax", ctx["tax"]), ("Penalty", ctx["pen"]), ("Interest", ctx["intr"]),
                  ("Amount due", ctx["amount"])]),
        "", "What you need to do",
        f"Pay the amount due {dl(ctx, rng)}. If you don't, we may levy your wages, bank accounts, Social Security "
        f"benefits, and other property or rights to property, and we may file a Notice of Federal Tax Lien.",
        "",
        f"If you disagree with the amount or want to propose a collection alternative, you have the right to a "
        f"Collection Due Process hearing with the IRS Independent Office of Appeals. To request it, complete the "
        f"enclosed Form 12153, Request for a Collection Due Process or Equivalent Hearing, and send it to us "
        f"{dl(ctx, rng)}. If you request a hearing on time, we generally won't levy while your case is pending.",
        "",
        f"If you can't pay in full, call us at {PHONE['LT11']} to discuss an installment agreement, an offer in "
        f"compromise, or a temporary delay of collection.",
    ]


def t_cp49(ctx, rng):
    ty, prior = ctx["tax_year"], ctx["prior_year"]
    rem = ctx["remaining"]
    return [
        f"We applied your {ty} refund to taxes you owe",
        f"Refund applied: {money(ctx['applied'])}",
        "",
        f"We applied {money(ctx['applied'])} of the overpayment on your {ty} Form 1040 to the balance you owed for "
        f"tax year {prior}, which fully paid that balance, including penalty and interest. "
        + (f"The remaining {money(rem)} of your overpayment will be refunded to you within 2 to 3 weeks."
           if rem else "No part of your overpayment remains to be refunded."),
        "", "Summary",
        f"  Overpayment on your {ty} return: {money(ctx['applied'] + rem)}",
        f"  Amount applied to your {prior} balance: {money(ctx['applied'])}",
        f"  Refund to be issued: {money(rem)}",
        "", "What you need to do",
        f"You don't need to do anything, and you don't owe any money as a result of this notice. Keep it with your "
        f"{ty} and {prior} tax records.",
        "",
        f"If you don't receive the remaining refund within 4 weeks, or if you think we applied this overpayment in "
        f"error or disagree with the {prior} balance, call us at {PHONE['CP49']} with a copy of this notice and your "
        f"{prior} tax records.",
    ]


def t_cp12(ctx, rng):
    ty = ctx["tax_year"]
    return [
        f"Changes to your {ty} Form 1040 - Refund increased",
        f"Refund due to you: {money(ctx['corrected'])}",
        "",
        f"We found a miscalculation on your {ty} Form 1040 and corrected it. You claimed a refund of "
        f"{money(ctx['claimed'])}; your corrected refund is {money(ctx['corrected'])}.",
        "", "What we changed",
        f"  {ctx['change']}",
        "", "What you need to do",
        "If you agree with the changes, you don't need to do anything. We will send your refund within 4 to 6 "
        "weeks, provided you don't owe other taxes or debts we're required to collect. You don't need to file an "
        "amended return to accept the changes.",
        "",
        f"If you disagree with the changes, call us at {PHONE['CP12']} or write to us at the address at the top of "
        f"this notice {dl(ctx, rng)}. Include a copy of this notice and any documents that support your position. "
        f"If we don't hear from you by then, we will assume you agree, and the changes will stand.",
    ]


def t_cp11(ctx, rng):
    ty = ctx["tax_year"]
    return [
        f"Changes to your {ty} Form 1040 - Amount due",
        f"Amount due: {money(ctx['amount'])}",
        "",
        f"We made changes to your {ty} return because we found a miscalculation. {ctx['change']} As a result of the "
        f"correction, you owe {money(ctx['amount'])}.",
        *billing([("Increase in tax from the correction", ctx["tax"]), ("Failure-to-pay penalty", ctx["pen"]),
                  ("Interest charges", ctx["intr"]), ("Amount due", ctx["amount"])]),
        "", "What you need to do",
        f"Pay {money(ctx['amount'])} {dl(ctx, rng)}. Pay online at www.irs.gov/payments or send a check or money "
        f"order payable to the United States Treasury with the payment stub from this notice.",
        "",
        f"If you disagree with the changes, call us at {PHONE['CP11']} before the due date and have a copy of your "
        f"return and this notice available. If you don't contact us, we will assume you agree with the changes. "
        f"You don't need to file an amended return.",
    ]


def t_cp22a(ctx, rng):
    ty = ctx["tax_year"]
    return [
        f"Changes to your {ty} Form 1040 based on information you provided - Amount due",
        f"Amount due: {money(ctx['amount'])}",
        "",
        f"We made the changes you requested on your amended return, Form 1040-X, for tax year {ty}. "
        f"{ctx['change']} As a result of these changes, you owe {money(ctx['amount'])}.",
        *billing([("Increase in tax", ctx["tax"]), ("Failure-to-pay penalty", ctx["pen"]),
                  ("Interest charges", ctx["intr"]), ("Amount due", ctx["amount"])]),
        "", "What you need to do",
        f"Pay {money(ctx['amount'])} {dl(ctx, rng)} to avoid additional penalty and interest. Pay online at "
        f"www.irs.gov/payments, or send a check or money order payable to the United States Treasury with the "
        f"payment stub. Write your Social Security number, tax year {ty}, and CP22A on your payment.",
        "",
        f"If you can't pay in full, apply for a payment plan at www.irs.gov/opa. If you think we made the changes "
        f"incorrectly, call us at {PHONE['CP22A']} with a copy of your amended return and this notice.",
    ]


def t_cp23(ctx, rng):
    ty = ctx["tax_year"]
    return [
        f"Changes to your {ty} Form 1040 - Estimated tax payments don't match",
        f"Amount due: {money(ctx['amount'])}",
        "",
        f"The estimated tax payments we credited to your {ty} account ({money(ctx['credited'])}) don't match the "
        f"amount you claimed on your return ({money(ctx['claimed'])}). We adjusted your return to the payments we "
        f"have on record, and as a result you owe {money(ctx['amount'])}.",
        *billing([("Estimated payments claimed on return", ctx["claimed"]),
                  ("Estimated payments credited to your account", ctx["credited"]),
                  ("Failure-to-pay penalty", ctx["pen"]), ("Interest charges", ctx["intr"]),
                  ("Amount due", ctx["amount"])]),
        "", "What you need to do",
        f"Compare the payments we credited with your records, such as cancelled checks, bank statements, or IRS "
        f"Direct Pay confirmations. If you agree, pay {money(ctx['amount'])} {dl(ctx, rng)}. If you made a payment "
        f"we didn't credit, send us a copy of the front and back of the cancelled check or the payment confirmation "
        f"{dl(ctx, rng)} so we can locate it and correct your account.",
        "", f"If you have questions, call us at {PHONE['CP23']} and have this notice and your payment records "
            f"available.",
    ]


def t_cp161(ctx, rng):
    ty, ent = ctx["tax_year"], ctx["tp"]["name"]
    return [
        f"Balance due on your Form 1120-S for tax year {ty}",
        f"Amount due: {money(ctx['amount'])}",
        "",
        f"Our records show a balance due on the Form 1120-S, U.S. Income Tax Return for an S Corporation, that "
        f"{ent} filed for tax year {ty}. The balance is a penalty under IRC section 6699 for filing the return "
        f"{ctx['months_late']} month{'s' if ctx['months_late'] != 1 else ''} after the due date, computed per "
        f"shareholder for each month the return was "
        f"late, plus interest.",
        *billing([("Penalty for filing late", ctx["pen"]), ("Interest charges", ctx["intr"]),
                  ("Amount due", ctx["amount"])]),
        "", "What you need to do",
        f"Pay {money(ctx['amount'])} {dl(ctx, rng)}. Pay online at www.irs.gov/payments using the Electronic Federal "
        f"Tax Payment System (EFTPS), or send a check or money order payable to the United States Treasury with the "
        f"payment stub. Write the employer identification number, tax year {ty}, and Form 1120-S on your payment.",
        "",
        f"If you believe the return was filed on time, or you had reasonable cause for filing late, send us a signed "
        f"statement explaining the circumstances with copies of supporting records. You may also qualify for "
        f"first-time penalty abatement if the corporation has a clean compliance history. Call us at "
        f"{PHONE['CP161']} if you have questions.",
    ]


def t_cp75(ctx, rng):
    ty = ctx["tax_year"]
    return [
        f"We're auditing your {ty} Form 1040 - Earned Income Credit",
        f"Refund amount being held: {money(ctx['held'])}",
        "",
        f"We're auditing the Earned Income Credit (EIC) and the dependents claimed on your {ty} Form 1040. We're "
        f"holding the portion of your refund related to these items, {money(ctx['held'])}, until we complete the "
        f"audit. The rest of your refund has been released.",
        "", "What you need to do",
        f"Send us copies of documents {dl(ctx, rng)} that show each child you claimed lived with you in the United "
        f"States for more than half of {ty}, such as school records, medical records, or a letter from a landlord "
        f"or childcare provider on official letterhead showing the child's name, your address, and the dates "
        f"covered. Also send proof of your relationship to each child, such as a birth certificate. Use the "
        f"enclosed Form 886-H-EIC as a checklist. Send copies, not originals, in the enclosed envelope or fax them "
        f"to 855-555-0147.",
        "",
        f"If you don't send the documents {dl(ctx, rng)}, we will disallow the credit, and you may owe additional "
        f"tax, penalties, and interest. If we find the claim was made recklessly, you may be banned from claiming "
        f"the EIC for 2 years; 10 years if fraudulent.",
        "", f"If you have questions, call us at {PHONE['CP75']} and have this notice available.",
    ]


def t_4883c(ctx, rng):
    ty = ctx["tax_year"]
    return [
        f"We need to verify your identity before we process your {ty} tax return",
        "",
        f"We received a federal income tax return, Form 1040, for tax year {ty} with your name and Social Security "
        f"number. To protect you from identity theft, we need to verify that you filed this return before we "
        f"process it. Until we hear from you, we can't process the return or issue any refund claimed on it.",
        "", "What you need to do",
        f"Call the Taxpayer Protection Program at {PHONE['4883C']} {dl(ctx, rng)}. When you call, have available: "
        f"this letter; a copy of the {ty} tax return you filed, if you filed one; a copy of a prior-year tax return; "
        f"and supporting documents such as Forms W-2 and 1099. We can't verify your identity online for this "
        f"letter; you must call.",
        "",
        f"If you didn't file a {ty} tax return, call the same number so we can remove the return from your account "
        f"and protect you from further misuse of your identity. You may also complete Form 14039, Identity Theft "
        f"Affidavit.",
        "",
        f"If you don't call us {dl(ctx, rng)}, we won't be able to process the return, and any refund claimed will "
        f"not be issued. This letter doesn't ask you to pay anything.",
    ]


def t_5071c(ctx, rng):
    ty = ctx["tax_year"]
    return [
        f"Verify your identity and your {ty} tax return",
        "",
        f"We received a Form 1040 for tax year {ty} with your name and Social Security number, and we need to verify "
        f"your identity before we process it. This is a precaution to protect you from identity theft; it doesn't "
        f"mean someone else filed a return in your name.",
        "", "What you need to do",
        f"Verify your identity online at www.idverify.irs.gov using the access code {ctx['access']} shown here, "
        f"{dl(ctx, rng)}. You will need this letter, the {ty} return you filed, a prior-year return, and a mobile "
        f"phone or a financial account number to confirm your identity. If you can't verify online, call "
        f"{PHONE['5071C']} and have the same documents available.",
        "",
        f"If you didn't file this {ty} return, tell us during the verification so we can remove it from your "
        f"account, and consider filing Form 14039, Identity Theft Affidavit.",
        "",
        f"If you don't verify {dl(ctx, rng)}, we won't process the return or release any refund claimed on it. No "
        f"payment is requested with this letter.",
    ]


def t_cp3219a(ctx, rng):
    ty, payer, form, desc = ctx["tax_year"], *ctx["payer"]
    return [
        "Notice of Deficiency - Proposed increase in tax and notice of your right to challenge in Tax Court",
        f"Total deficiency and penalty: {money(ctx['amount'])}",
        "",
        f"We previously sent you a CP2000 notice proposing changes to your {ty} Form 1040, and we didn't receive a "
        f"response resolving the issue. This is your Notice of Deficiency under IRC section 6212. We have determined a deficiency (increase in tax) of {money(ctx['tax'])} "
        f"for tax year {ty}, plus an accuracy-related penalty of {money(ctx['pen'])} under IRC section 6662, for a "
        f"total of {money(ctx['amount'])}. Interest will be charged in addition.",
        "", "Explanation of changes",
        f"  {payer} reported {money(ctx['unreported'])} of {desc} on Form {form}; this amount isn't on your return.",
        "", "What you need to do",
        f"If you want to challenge this determination in court before paying, you must file a petition with the "
        f"United States Tax Court {dl(ctx, rng)}. The court can't consider a late petition, and we can't extend this "
        f"time. You can file the petition online at www.ustaxcourt.gov or mail it to the court at 400 Second Street "
        f"NW, Washington, DC 20217.",
        "",
        "If you agree, sign and return the enclosed Form 5564, Notice of Deficiency - Waiver, so we can assess the "
        "tax and send you a bill. If you neither petition the Tax Court nor sign the waiver, we will assess the "
        "deficiency after the petition period ends.",
    ]


def t_cp90(ctx, rng):
    ty = ctx["tax_year"]
    return [
        "Final Notice of Intent to Levy and Notice of Your Right to a Hearing",
        f"Amount due: {money(ctx['amount'])}",
        "",
        f"We previously sent you notices asking you to pay your federal tax, and our records show you still owe "
        f"{money(ctx['amount'])} for tax year {ty}. This is your final notice of our intent to levy under IRC "
        f"section 6331 and of your right to a hearing under section 6330. If you don't pay the amount due or make "
        f"other arrangements {dl(ctx, rng)}, we may levy your property, including wages, bank accounts, Social "
        f"Security benefits, and federal payments, and we may file a Notice of Federal Tax Lien.",
        *billing([("Unpaid tax", ctx["tax"]), ("Penalty", ctx["pen"]), ("Interest", ctx["intr"]),
                  ("Amount due", ctx["amount"])]),
        "", "What you need to do",
        f"Pay {money(ctx['amount'])} {dl(ctx, rng)}, or request a Collection Due Process hearing with the IRS "
        f"Independent Office of Appeals by completing the enclosed Form 12153 and sending it to us {dl(ctx, rng)}. "
        f"A timely hearing request generally stops levy action while Appeals considers your case; a late request "
        f"gets only an equivalent hearing and doesn't stop collection.",
        "",
        f"If you can't pay in full, call us at {PHONE['CP90']} to discuss an installment agreement or an offer in "
        f"compromise.",
    ]


def t_cp259(ctx, rng):
    ty, ent = ctx["tax_year"], ctx["tp"]["name"]
    return [
        f"You must file your {ty} Form 1065",
        "",
        f"Our records show that {ent} is required to file Form 1065, U.S. Return of Partnership Income, for tax year "
        f"{ty}, but we have no record of receiving it. The return was due {long_date(ctx['due'])}. No amount is "
        f"due with this notice.",
        "", "What you need to do",
        f"File the {ty} Form 1065 {dl(ctx, rng)}. You can file electronically through an authorized e-file provider "
        f"or mail the return to the address at the top of this notice with a copy of this notice on top.",
        "",
        f"If you already filed, send us a signed copy of the return and proof of filing, such as the e-file "
        f"acceptance acknowledgment, with the enclosed response page {dl(ctx, rng)}. If the partnership isn't "
        f"required to file because it dissolved, had no income or expenses, or elected to be treated as a "
        f"different entity type, complete the enclosed response page and tell us why.",
        "",
        "If you don't respond, we may charge a penalty under IRC section 6698 for each month or part of a month the "
        "return is late, multiplied by the number of partners, and each partner's own return may be delayed or "
        "examined.",
        "", f"If you have questions, call us at {PHONE['CP259']}.",
    ]


def t_cp71c(ctx, rng):
    ty = ctx["tax_year"]
    return [
        f"Annual reminder of your unpaid balance for tax year {ty}",
        f"Amount due: {money(ctx['amount'])}",
        "",
        f"We're required to send you this annual reminder of the unpaid balance on your account for tax year {ty}. "
        f"The amount shown includes penalty and interest computed to the date of this notice. This reminder doesn't "
        f"set a new payment due date; penalty and interest continue to accrue daily until the balance is paid in "
        f"full.",
        *billing([("Unpaid tax", ctx["tax"]), ("Penalty", ctx["pen"]), ("Interest", ctx["intr"]),
                  ("Amount due", ctx["amount"])]),
        "", "What you need to do",
        "Pay the amount due immediately. Pay online at www.irs.gov/payments or send a check or money order payable "
        "to the United States Treasury with the payment stub from this notice. If you have an installment "
        "agreement with us, continue making your scheduled payments; you don't need to send an additional payment, "
        "but paying more reduces the interest you will owe.",
        "",
        f"If you can't pay, call us at {PHONE['CP71C']} to discuss a payment plan, an offer in compromise, or a "
        f"temporary delay of collection. If you already paid this balance, allow 21 days for the payment to post "
        f"before contacting us.",
    ]


def t_cp21b(ctx, rng):
    ty = ctx["tax_year"]
    return [
        f"Changes to your {ty} Form 1040 - Refund due",
        f"Refund: {money(ctx['refund'] + ctx['intr'])}",
        "",
        f"We made the changes you requested on your amended return, Form 1040-X, for tax year {ty}. "
        f"{ctx['change']} As a result of these changes, you're due a refund of {money(ctx['refund'] + ctx['intr'])}, "
        f"which includes {money(ctx['intr'])} of interest we owe you.",
        "", "Summary",
        f"  Decrease in tax from your amended return: {money(ctx['refund'])}",
        f"  Interest we owe you: {money(ctx['intr'])}",
        f"  Refund amount: {money(ctx['refund'] + ctx['intr'])}",
        "", "What you need to do",
        "You don't need to do anything. You should receive your refund within 2 to 3 weeks of the date of this "
        "notice, as long as you don't owe other tax or debts we're required to collect. We don't need any payment "
        "or further information from you.",
        "",
        f"If you don't receive your refund within 6 weeks, call us at {PHONE['CP21B']}. The interest we paid you is "
        f"taxable income; we will send you a Form 1099-INT if it is $10 or more, and you should report it on next "
        f"year's return.",
    ]


def t_cp05(ctx, rng):
    ty = ctx["tax_year"]
    return [
        f"We're reviewing your {ty} tax return",
        "",
        f"We're holding your refund until we finish reviewing your {ty} Form 1040. We're verifying the income, income "
        f"tax withholding, tax credits, and business income you reported. Returns are selected for this review "
        f"based on the information we receive; being selected doesn't mean you made an error.",
        "", "What you need to do",
        "You don't need to do anything at this time, and you don't need to send us any documents or call us. We "
        "will contact you if we need additional information. Please don't file a second return or an amended "
        "return; doing so will delay the review.",
        "", "What happens next",
        f"If our review shows no changes, we will send your refund. If we need information, we will send you a "
        f"letter telling you exactly what to send. We expect to complete the review within 60 days of the date of "
        f"this notice. If you haven't received your refund or a letter after 60 days, you may call us at "
        f"{PHONE['CP05']} with this notice available. No payment is requested with this notice.",
    ]


TEMPLATES = {"CP2000": t_cp2000, "CP2501": t_cp2501, "CP14": t_cp14, "CP501": t_cp501, "CP503": t_cp503,
             "CP504": t_cp504, "LT11": t_lt11, "CP49": t_cp49, "CP12": t_cp12, "CP11": t_cp11, "CP22A": t_cp22a,
             "CP23": t_cp23, "CP161": t_cp161, "CP75": t_cp75, "4883C": t_4883c, "5071C": t_5071c,
             "CP3219A": t_cp3219a, "CP90": t_cp90, "CP259": t_cp259, "CP71C": t_cp71c, "CP21B": t_cp21b,
             "CP05": t_cp05}
CHANGES = [
    "We changed the amount claimed as Child Tax Credit on line 19 because a dependent's Social Security number was "
    "missing or invalid.",
    "We changed the total tax on line 24 because the tax computed from the taxable income on line 15 was "
    "miscalculated.",
    "We changed the amount of federal income tax withheld on line 25a to match the Forms W-2 and 1099 we received.",
    "We changed the standard deduction on line 12 to the amount allowed for your filing status.",
    "We changed the amount of estimated tax payments on line 26 to match the payments credited to your account.",
]


# --------------------------------------------------------------------------- per-notice context
def build_context(n, ntype, style):
    rng = random.Random(n)
    ctx = {"n": n, "id": f"notice{n:02d}", "ntype": ntype, "style": style, "center": rng.choice(CENTERS)}
    ctx["tp"] = entity(rng) if ntype in BUSINESS_TYPES else person(rng)
    # notice date: a 2026 date (CP14 for TY2025 lands May-July; everything else Jan 12 - Oct 20)
    if ntype == "CP14":
        ctx["notice_date"] = date(2026, 5, 4) + timedelta(days=rng.randint(0, 77))
    else:
        ctx["notice_date"] = date(2026, 1, 12) + timedelta(days=rng.randint(0, 281))
    if ntype in ("CP2000", "CP2501", "CP3219A"):
        ty = 2024                                   # AUR matching runs a year behind
    elif ntype in ("CP501", "CP503", "CP504", "LT11", "CP90", "CP71C"):
        ty = rng.choice([2023, 2024])               # aged balances
    else:
        ty = 2025
    ctx["tax_year"] = str(ty)
    ctx["pages"] = rng.choice([1, 2, 2, 3])
    if ntype in ("CP501", "CP503", "CP504", "LT11", "CP90", "CP71C"):
        ctx["caller_id"] = f"{rng.randint(100000, 999999)}"
    # amounts
    ctx["amount"] = 0
    if ntype in ("CP2000", "CP2501", "CP3219A"):
        ctx["payer"] = rng.choice(PAYERS)
        ctx["unreported"] = cents(rng, 1800, 42000)
        ctx["tax"] = int(ctx["unreported"] * rng.choice([0.22, 0.24, 0.32]))
        ctx["pen"] = int(ctx["tax"] * 0.20) if ctx["tax"] > 500000 or rng.random() < 0.5 else 0
        ctx["intr"] = int(ctx["tax"] * rng.uniform(0.06, 0.14))
        if ntype == "CP2000":
            ctx["amount"] = ctx["tax"] + ctx["pen"] + ctx["intr"]
        elif ntype == "CP3219A":
            ctx["amount"] = ctx["tax"] + ctx["pen"]          # interest "charged in addition"
    elif ntype in ("CP14", "CP501", "CP503", "CP504", "LT11", "CP90", "CP71C", "CP11", "CP22A"):
        ctx["tax"] = cents(rng, 900, 38000)
        ctx["pen"] = int(ctx["tax"] * rng.uniform(0.005, 0.12))
        ctx["intr"] = int(ctx["tax"] * rng.uniform(0.01, 0.09))
        ctx["amount"] = ctx["tax"] + ctx["pen"] + ctx["intr"]
        ctx["change"] = rng.choice(CHANGES)
    elif ntype == "CP23":
        ctx["claimed"] = cents(rng, 6000, 30000)
        ctx["credited"] = ctx["claimed"] - cents(rng, 1000, 5500)
        diff = ctx["claimed"] - ctx["credited"]
        ctx["pen"] = int(diff * rng.uniform(0.005, 0.03))
        ctx["intr"] = int(diff * rng.uniform(0.01, 0.05))
        ctx["amount"] = diff + ctx["pen"] + ctx["intr"]
    elif ntype == "CP161":
        ctx["months_late"] = rng.randint(1, 5)
        ctx["pen"] = cents(rng, 500, 3900)
        ctx["intr"] = int(ctx["pen"] * rng.uniform(0.01, 0.04))
        ctx["amount"] = ctx["pen"] + ctx["intr"]
    elif ntype == "CP49":
        ctx["prior_year"] = str(ty - rng.choice([1, 2]))
        ctx["applied"] = cents(rng, 400, 5200)
        ctx["remaining"] = rng.choice([0, cents(rng, 50, 2600)])
    elif ntype == "CP12":
        ctx["claimed"] = cents(rng, 300, 4800)
        ctx["corrected"] = ctx["claimed"] + cents(rng, 120, 2200)
        ctx["change"] = rng.choice(CHANGES)
    elif ntype == "CP75":
        ctx["held"] = cents(rng, 1800, 7400)
    elif ntype == "5071C":
        ctx["access"] = f"{rng.randint(100000, 999999)}"
    elif ntype == "CP259":
        ctx["due"] = date(ty + 1, 3, 16)                    # calendar-year 1065 due date (Mar 15, 2026 is Sunday)
    elif ntype == "CP21B":
        ctx["refund"] = cents(rng, 350, 6900)
        ctx["intr"] = int(ctx["refund"] * rng.uniform(0.01, 0.05))
        ctx["change"] = rng.choice(CHANGES)
    # deadline
    if style == "none":
        ctx["days"], ctx["deadline"] = None, None
    else:
        ctx["days"] = WINDOW_DAYS[ntype]
        ctx["deadline"] = ctx["notice_date"] + timedelta(days=ctx["days"])   # calendar days, no business-day shift
    return ctx, rng


def build_notice(n, ntype, style):
    ctx, rng = build_context(n, ntype, style)
    lines = header(ctx) + TEMPLATES[ntype](ctx, rng) + footer(ctx)
    text = "\n".join(lines).rstrip() + "\n"
    r = RUBRIC[ntype]
    if style == "explicit":
        why = f"deadline stated explicitly as {long_date(ctx['deadline'])}"
    elif style == "relative":
        why = (f"deadline stated as a {ctx['days']}-day window; ground truth = notice date "
               f"{ctx['notice_date'].isoformat()} + {ctx['days']} calendar days = {ctx['deadline'].isoformat()}")
    else:
        why = "notice sets no deadline for the taxpayer -> NONE"
    amt_why = (f"amount due {canon(ctx['amount'])} is what the notice asks the taxpayer to pay" if ctx["amount"]
               else "no payment requested -> 0.00")
    gt = {
        "notice_type": ntype,
        "tax_year": ctx["tax_year"],
        "notice_date": ctx["notice_date"].isoformat(),
        "amount_due": canon(ctx["amount"]),
        "response_deadline": ctx["deadline"].isoformat() if ctx["deadline"] else "NONE",
        "action_required": r["action_required"],
        "urgency": r["urgency"],
        "rubric_reason": f"{ntype} -> {r['action_required']} / {r['urgency']} per rubric; {amt_why}; {why}",
    }
    return ctx, text, gt


# --------------------------------------------------------------------------- checks
def verify(ctx, text, gt):
    wc = len(text.split())
    assert 120 <= wc <= 320, f"{ctx['id']}: {wc} words (need 120-320)"
    assert gt["notice_type"] in ENUMS["notice_type"], gt["notice_type"]
    assert gt["action_required"] in ENUMS["action_required"], gt["action_required"]
    assert gt["urgency"] in ENUMS["urgency"], gt["urgency"]
    assert re.fullmatch(r"\d+\.\d{2}", gt["amount_due"]) and re.fullmatch(r"20\d{2}", gt["tax_year"])
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", gt["notice_date"])
    assert gt["response_deadline"] == "NONE" or re.fullmatch(r"\d{4}-\d{2}-\d{2}", gt["response_deadline"])
    assert long_date(ctx["notice_date"]) in text, "notice date not printed"
    assert "Department of the Treasury - Internal Revenue Service" in text and "What you need to do" in text
    assert ("XXX-XX-" in text) or ctx["ntype"] in BUSINESS_TYPES
    if ctx["style"] == "explicit":
        assert long_date(ctx["deadline"]) in text, "explicit deadline not printed"
    elif ctx["style"] == "relative":
        assert re.search(rf"\b{ctx['days']} days\b", text), "relative window not printed"
        assert long_date(ctx["deadline"]) not in text, "relative-style notice leaks the computed date"
        assert (date.fromisoformat(gt["response_deadline"]) - ctx["notice_date"]).days == ctx["days"]
    else:
        assert gt["response_deadline"] == "NONE"
    if ctx["amount"]:
        assert money(ctx["amount"]) in text, "amount due not printed"
    return wc


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    gts, counts = {}, []
    for n, ntype, style in SPECS:
        ctx, text, gt = build_notice(n, ntype, style)
        counts.append(verify(ctx, text, gt))
        path = os.path.join(OUT_DIR, f"{ctx['id']}.txt")
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        gts[ctx["id"]] = gt
        print(f"{ctx['id']}  {ntype:<8} {style:<8} words={counts[-1]:>3}  date={gt['notice_date']}  "
              f"due={gt['response_deadline']:<10} amt={gt['amount_due']:>9}  {gt['action_required']}/{gt['urgency']}")
    assert all(set(v.keys()) == set(NOTICE_FIELDS) | {"rubric_reason"} for v in gts.values())
    with open(GT_PATH, "w", encoding="utf-8") as fh:
        json.dump(gts, fh, indent=1)
        fh.write("\n")
    types = sorted({v["notice_type"] for v in gts.values()})
    styles = {s: sum(1 for _, _, st in SPECS if st == s) for s in ("explicit", "relative", "none")}
    print(f"\nwrote {len(gts)} notices + {GT_PATH}\n{len(types)} distinct types: {', '.join(types)}\n"
          f"deadline styles: {styles}; words min/max = {min(counts)}/{max(counts)}")


if __name__ == "__main__":
    main()
