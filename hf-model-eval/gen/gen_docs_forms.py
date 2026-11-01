#!/usr/bin/env python3
"""Generate doc01-doc12 (W-2, 1099-NEC/INT/DIV/R, consolidated 1099-B summary)
as one-page letter PDFs + degraded PNG scans, plus ground_truth_forms.json.

Synthetic identities only. Deterministic: every value for doc N comes from
random.Random(N). Rerun -> identical files.

    python3 gen/gen_docs_forms.py
"""
import json
import os
import random
import sys

from reportlab.lib.pagesizes import letter
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

ROOT = "/home/user/FastClose/hf-model-eval"
GEN = os.path.join(ROOT, "gen")
OUT = os.path.join(ROOT, "data", "docs")
sys.path.insert(0, GEN)
from scan_effects import PROFILES, pdf_to_scan  # noqa: E402

FIELDS = json.load(open(os.path.join(ROOT, "eval", "fields.json")))["doc_types"]
PAGE_W, PAGE_H = letter
CAP = "Helvetica"
CAPB = "Helvetica-Bold"
VAL = "Courier"
VALB = "Courier-Bold"
CAP_SIZE = 7
VAL_SIZE = 10

# ----------------------------------------------------------------------------
# synthetic identity pools (all invented)
# ----------------------------------------------------------------------------
PEOPLE = [
    ("MARIA OKONKWO", "1420 PINE HOLLOW RD", "SACRAMENTO", "CA", "95814"),
    ("DANIEL REYES", "88 COPPERFIELD LN", "AUSTIN", "TX", "78745"),
    ("PRIYA SUNDARAM", "307 WILLOW CREST AVE APT 4B", "BROOKLYN", "NY", "11215"),
    ("THOMAS WHITFIELD", "5210 BAYSHORE TER", "TAMPA", "FL", "33611"),
    ("ANGELA KOWALSKI", "762 MAPLE RIDGE DR", "NAPERVILLE", "IL", "60540"),
    ("SAMUEL NDIAYE", "19 HARBOR POINT CT", "STAMFORD", "CT", "06902"),
    ("LAUREN FITZGERALD", "4406 SAGEBRUSH WAY", "PHOENIX", "AZ", "85032"),
    ("OMAR HADDAD", "1133 GRANITE FALLS BLVD", "DENVER", "CO", "80231"),
    ("CHLOE BRENNAN", "27 OLD MILL RD", "PORTSMOUTH", "NH", "03801"),
    ("ELENA VASQUEZ", "930 CYPRESS GROVE ST", "SAN DIEGO", "CA", "92115"),
    ("MARCUS DELANEY", "615 RIVERBEND PKWY", "ATLANTA", "GA", "30312"),
    ("HANNAH LINDQVIST", "2088 SUMMIT VIEW DR", "SEATTLE", "WA", "98115"),
]

EMPLOYERS = [
    ("NORTHGATE LOGISTICS LLC", "2200 INDUSTRIAL PKWY STE 300", "SACRAMENTO", "CA", "95828", "CA", "800-4471-9"),
    ("BRIARWOOD DENTAL GROUP PC", "4501 LAKELINE BLVD", "AUSTIN", "TX", "78717", "TX", "17-2209887-4"),
    ("SILVERLINE SOFTWARE INC", "1 METROTECH CENTER FL 18", "BROOKLYN", "NY", "11201", "NY", "27-5518443"),
    ("HARBORVIEW HOSPITALITY CORP", "7800 WESTSHORE BLVD", "TAMPA", "FL", "33607", "FL", "FL-7733910"),
]

NEC_PAYERS = [
    ("CEDAR RIDGE MARKETING LLC", "901 ELM STREET STE 1200", "DALLAS", "TX", "75202"),
    ("GREGORY LANDIS DBA LANDIS RENTALS", "3318 OAKMONT DR", "COLUMBUS", "OH", "43221"),
]
INT_PAYERS = [
    ("FIRST MERIDIAN BANK NA", "PO BOX 41900", "CHARLOTTE", "NC", "28201"),
    ("LAKESHORE COMMUNITY FCU", "6150 LAKESHORE AVE", "CLEVELAND", "OH", "44102"),
]
DIV_PAYERS = [
    ("WESTBROOK SECURITIES LLC", "200 LIBERTY PLAZA FL 24", "NEW YORK", "NY", "10281"),
    ("PINNACLE ASSET MANAGEMENT INC", "1550 MARKET ST STE 900", "SAN FRANCISCO", "CA", "94103"),
]
R_PAYER = ("GRANITE PEAK RETIREMENT SERVICES", "PO BOX 7770", "SALT LAKE CITY", "UT", "84110")
B_PAYER = ("CASCADE BROKERAGE SERVICES LLC", "500 PIKE STREET STE 2100", "SEATTLE", "WA", "98101")

TAX_YEAR = {n: "2025" for n in range(1, 13)}
TAX_YEAR[3] = "2024"
TAX_YEAR[9] = "2024"
SS_WAGE_BASE = {"2024": 168600.0, "2025": 176100.0}


# ----------------------------------------------------------------------------
# value helpers
# ----------------------------------------------------------------------------
def money(x):
    """printed form: 84,210.55"""
    return f"{x:,.2f}"


def canon(x):
    """ground-truth form: 84210.55"""
    return f"{x:.2f}"


def r2(x):
    return round(x + 1e-9, 2)


def make_ein(rnd):
    prefix = rnd.choice([27, 33, 36, 45, 46, 47, 81, 82, 83, 84, 85, 86, 87, 88, 92, 93, 99])
    return f"{prefix:02d}-{rnd.randint(1000000, 9999999):07d}"


def make_ssn_last4(rnd):
    return f"{rnd.randint(0, 9999):04d}"


def redacted_ssn(last4):
    return f"XXX-XX-{last4}"


def doc_rnd(n):
    return random.Random(n)


def profile_for(n):
    return PROFILES[(n - 1) % 4]


# ----------------------------------------------------------------------------
# drawing helpers
# ----------------------------------------------------------------------------
def wrap(text, font, size, width):
    """Greedy word wrap using real string widths."""
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if stringWidth(trial, font, size) <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def box(c, x, y, w, h, caption, value="", vfont=VALB, vsize=VAL_SIZE, align="left",
        csize=CAP_SIZE, lw=0.6, cfont=CAP):
    """Rectangle with caption (top-left, Helvetica) and value (bottom, Courier).
    Caption wraps to the box width; value sits on the bottom line."""
    c.setLineWidth(lw)
    c.rect(x, y, w, h)
    cy = y + h - csize - 1.5
    if caption:
        c.setFont(cfont, csize)
        for line in wrap(caption, cfont, csize, w - 5):
            c.drawString(x + 2.5, cy, line)
            cy -= csize + 1
    if value:
        c.setFont(vfont, vsize)
        if align == "right":
            c.drawRightString(x + w - 5, y + 4, value)
        else:
            c.drawString(x + 5, y + 4, value)


def mbox(c, x, y, w, h, caption, value):
    """money box: value right-aligned"""
    box(c, x, y, w, h, caption, value, align="right")


def check(c, x, y, checked, size=8):
    c.setLineWidth(0.6)
    c.rect(x, y, size, size)
    if checked:
        c.setFont(VALB, size + 1)
        c.drawString(x + 1.2, y + 1, "X")


def textblock(c, x, y, w, h, caption, lines, vfont=VAL, vsize=VAL_SIZE):
    """Box with caption and several value lines (name/address blocks)."""
    box(c, x, y, w, h, caption)
    n_cap = len(wrap(caption, CAP, CAP_SIZE, w - 5)) if caption else 0
    c.setFont(vfont, vsize)
    ty = y + h - 4 - n_cap * (CAP_SIZE + 1) - vsize - 1
    for ln in lines:
        for sub in wrap(ln, vfont, vsize, w - 10):  # long names wrap instead of overflowing
            c.drawString(x + 5, ty, sub)
            ty -= vsize + 2.5


def hline(c, x1, x2, y, lw=0.6):
    c.setLineWidth(lw)
    c.line(x1, y, x2, y)


def label(c, x, y, text, font=CAP, size=CAP_SIZE, align="left"):
    c.setFont(font, size)
    if align == "right":
        c.drawRightString(x, y, text)
    elif align == "center":
        c.drawCentredString(x, y, text)
    else:
        c.drawString(x, y, text)


# ----------------------------------------------------------------------------
# W-2  (doc01-doc04)
# ----------------------------------------------------------------------------
W2_VARIANTS = {
    # codes listed in box-12 order (12a first); fed=False -> 0.00 federal withholding
    1: dict(codes=["D", "DD"], fed=True),
    2: dict(codes=["DD"], fed=False),
    3: dict(codes=["W"], fed=True),
    4: dict(codes=["D", "C"], fed=True),
}
STATE_RATE = {"CA": (0.045, 0.085), "NY": (0.045, 0.065), "TX": None, "FL": None}


def w2_data(n):
    rnd = doc_rnd(n)
    i = n - 1
    emp = EMPLOYERS[i]
    person = PEOPLE[i]
    year = TAX_YEAR[n]
    var = W2_VARIANTS[n]
    gross = r2(rnd.uniform(28000, 240000))
    ein = make_ein(rnd)
    last4 = make_ssn_last4(rnd)
    control = f"{rnd.randint(1000, 9999)}-{rnd.randint(100000, 999999)}"
    limit_401k = 23500.0 if year == "2025" else 23000.0

    code_amounts = {}
    for code in var["codes"]:
        if code == "D":
            code_amounts[code] = r2(min(gross * rnd.uniform(0.04, 0.10), limit_401k))
        elif code == "DD":
            code_amounts[code] = r2(rnd.uniform(6000, 24000))
        elif code == "W":
            code_amounts[code] = r2(rnd.uniform(1000, 8300))
        elif code == "C":
            code_amounts[code] = r2(rnd.uniform(30, 400))
    deferral = code_amounts.get("D", 0.0)
    hsa = code_amounts.get("W", 0.0)
    gtl = code_amounts.get("C", 0.0)

    box5 = r2(gross - hsa + gtl)
    box3 = r2(min(box5, SS_WAGE_BASE[year]))
    box1 = r2(box5 - deferral)
    box4 = r2(box3 * 0.062)
    box6 = r2(box5 * 0.0145 + max(0.0, box5 - 200000.0) * 0.009)
    box2 = r2(box1 * rnd.uniform(0.11, 0.22)) if var["fed"] else 0.0

    state = emp[5]
    if STATE_RATE[state]:
        lo, hi = STATE_RATE[state]
        box16 = box1
        box17 = r2(box16 * rnd.uniform(lo, hi))
    else:
        box16 = 0.0
        box17 = 0.0

    other = []
    if state == "CA":
        other.append(f"CASDI {money(r2(box5 * (0.011 if year == '2024' else 0.012)))}")
    if state == "NY":
        other.append(f"NYSDI {money(r2(min(box5, 0) + 31.20))}")
        other.append(f"NYPFL {money(r2(box5 * 0.00388))}")

    first, last = person[0].split(" ", 1)
    d = dict(
        n=n, year=year, ein=ein, last4=last4, control=control,
        employer=emp, person=person, first=first, last=last,
        box1=box1, box2=box2, box3=box3, box4=box4, box5=box5, box6=box6,
        codes=[(c, code_amounts[c]) for c in var["codes"]],
        retirement=("D" in code_amounts),
        state=state, state_id=emp[6], box16=box16, box17=box17, other=other,
    )
    gt = {
        "tax_year": year,
        "employer_ein": ein,
        "employer_name": emp[0],
        "employee_ssn_last4": last4,
        "employee_name": person[0],
        "box1_wages": canon(box1),
        "box2_fed_withheld": canon(box2),
        "box3_ss_wages": canon(box3),
        "box4_ss_tax": canon(box4),
        "box5_medicare_wages": canon(box5),
        "box6_medicare_tax": canon(box6),
        "box12a_code": d["codes"][0][0],
        "box12a_amount": canon(d["codes"][0][1]),
        "box15_state": state,
        "box16_state_wages": canon(box16),
        "box17_state_tax": canon(box17),
    }
    return d, gt


def box12(c, x, y, w, h, caption, code="", amount=""):
    box(c, x, y, w, h, caption)
    label(c, x + 3, y + 4, "Code", size=7)
    c.setLineWidth(0.4)
    c.line(x + 40, y, x + 40, y + h - 12)
    if code:
        c.setFont(VALB, VAL_SIZE)
        c.drawString(x + 24, y + 4, code)
    if amount:
        c.setFont(VALB, VAL_SIZE)
        c.drawRightString(x + w - 5, y + 4, amount)


def draw_w2(c, d):
    X0, LW, RW = 36, 252, 144
    XR1, XR2 = X0 + LW, X0 + LW + RW
    FW = LW + 2 * RW
    emp, p = d["employer"], d["person"]

    # header strip
    label(c, X0, 764, "22222", font=CAPB, size=9)
    check(c, X0 + 42, 762, False)
    label(c, X0 + 53, 764, "Void", size=7)
    label(c, X0 + FW, 764, "OMB No. 1545-0008", size=8, align="right")
    label(c, X0 + FW, 753, "Safe, accurate, FAST! Use e-file   Visit the IRS website at www.irs.gov/efile", size=7, align="right")

    y = 740
    # row 1
    y -= 34
    box(c, X0, y, LW, 34, "a Employee's social security number", redacted_ssn(d["last4"]))
    box(c, XR1, y, 2 * RW, 34, "")
    label(c, XR1 + 5, y + 20, "This information is being furnished to the Internal Revenue Service. If you are", size=7)
    label(c, XR1 + 5, y + 11, "required to file a tax return, a negligence penalty or other sanction may be", size=7)
    label(c, XR1 + 5, y + 2.5, "imposed on you if this income is taxable and you fail to report it.", size=7)
    # row 2
    y -= 34
    box(c, X0, y, LW, 34, "b Employer identification number (EIN)", d["ein"])
    mbox(c, XR1, y, RW, 34, "1 Wages, tips, other compensation", money(d["box1"]))
    mbox(c, XR2, y, RW, 34, "2 Federal income tax withheld", money(d["box2"]))
    # row 3 (3 sub-rows)
    y -= 102
    textblock(c, X0, y, LW, 102, "c Employer's name, address, and ZIP code",
              [emp[0], emp[1], f"{emp[2]}, {emp[3]} {emp[4]}"])
    mbox(c, XR1, y + 68, RW, 34, "3 Social security wages", money(d["box3"]))
    mbox(c, XR2, y + 68, RW, 34, "4 Social security tax withheld", money(d["box4"]))
    mbox(c, XR1, y + 34, RW, 34, "5 Medicare wages and tips", money(d["box5"]))
    mbox(c, XR2, y + 34, RW, 34, "6 Medicare tax withheld", money(d["box6"]))
    mbox(c, XR1, y, RW, 34, "7 Social security tips", "")
    mbox(c, XR2, y, RW, 34, "8 Allocated tips", "")
    # row 4
    y -= 34
    box(c, X0, y, LW, 34, "d Control number", d["control"])
    box(c, XR1, y, RW, 34, "9", "")
    mbox(c, XR2, y, RW, 34, "10 Dependent care benefits", "")
    # row 5 (34 + 38)
    y -= 72
    box(c, X0, y, LW, 72, "")
    label(c, X0 + 2.5, y + 72 - 8.5, "e Employee's first name and initial", size=7)
    label(c, X0 + 140, y + 72 - 8.5, "Last name", size=7)
    label(c, X0 + 228, y + 72 - 8.5, "Suff.", size=7)
    c.setFont(VALB, VAL_SIZE)
    c.drawString(X0 + 5, y + 72 - 24, d["first"])
    c.drawString(X0 + 140, y + 72 - 24, d["last"])
    c.setLineWidth(0.4)
    c.line(X0 + 136, y + 72 - 30, X0 + 136, y + 72)
    c.line(X0 + 224, y + 72 - 30, X0 + 224, y + 72)
    mbox(c, XR1, y + 38, RW, 34, "11 Nonqualified plans", "")
    codes = d["codes"] + [("", 0.0)] * 4
    box12(c, XR2, y + 38, RW, 34, "12a See instructions for box 12",
          codes[0][0], money(codes[0][1]) if codes[0][0] else "")
    # box 13
    box(c, XR1, y, RW, 38, "13")
    for k, (txt, on) in enumerate([("Statutory employee", False),
                                   ("Retirement plan", d["retirement"]),
                                   ("Third-party sick pay", False)]):
        cy = y + 38 - 18 - k * 9
        check(c, XR1 + 18, cy, on, size=7)
        label(c, XR1 + 28, cy + 0.5, txt, size=7)
    box12(c, XR2, y, RW, 38, "12b", codes[1][0], money(codes[1][1]) if codes[1][0] else "")
    # row 6 (34 + 34)
    y -= 68
    textblock(c, X0, y, LW, 68, "f Employee's address and ZIP code",
              [p[1], f"{p[2]}, {p[3]} {p[4]}"])
    box(c, XR1, y, RW, 68, "14 Other")
    c.setFont(VAL, 9)
    for k, ln in enumerate(d["other"]):
        c.drawString(XR1 + 5, y + 68 - 20 - k * 11, ln)
    box12(c, XR2, y + 34, RW, 34, "12c", codes[2][0], money(codes[2][1]) if codes[2][0] else "")
    box12(c, XR2, y, RW, 34, "12d", codes[3][0], money(codes[3][1]) if codes[3][0] else "")
    # row 7: state / local
    y -= 34
    widths = [36, 108, 84, 84, 84, 72, 72]
    caps = ["15 State", "Employer's state ID number", "16 State wages, tips, etc.",
            "17 State income tax", "18 Local wages, tips, etc.", "19 Local income tax",
            "20 Locality name"]
    vals = [d["state"], d["state_id"], money(d["box16"]), money(d["box17"]), "", "", ""]
    aligns = ["left", "left", "right", "right", "right", "right", "left"]
    x = X0
    for w, cap, v, al in zip(widths, caps, vals, aligns):
        box(c, x, y, w, 34, cap, v, align=al)
        x += w
    # footer / title
    y -= 22
    c.setFont(CAPB, 11)
    c.drawString(X0, y, "Form")
    c.setFont(CAPB, 16)
    c.drawString(X0 + 30, y - 1, "W-2")
    c.setFont(CAPB, 11)
    c.drawString(X0 + 72, y, "Wage and Tax Statement")
    c.setFont(CAPB, 16)
    c.drawString(X0 + 214, y - 1, d["year"])
    label(c, X0 + FW, y + 1, "Department of the Treasury—Internal Revenue Service", size=8, align="right")
    y -= 16
    label(c, X0, y, "Copy B—To Be Filed With Employee's FEDERAL Tax Return.", font=CAPB, size=9)
    hline(c, X0, X0 + FW, y - 6, lw=0.8)
    # Notice to employee block (realistic filler; nothing scoreable in it)
    y -= 22
    label(c, X0, y, "Notice to Employee", font=CAPB, size=8)
    notice = [
        "Do you have to file? Refer to the Form 1040 instructions to determine if you are required to file a tax return. Even if you",
        "don't have to file a tax return, you may be eligible for a refund if box 2 shows an amount or if you are eligible for any credit.",
        "Earned income credit (EIC). You may be able to take the EIC for " + d["year"] + " if your adjusted gross income (AGI) is less than a",
        "certain amount. The amount of the credit is based on income and family size. Workers without children could qualify for a smaller credit.",
        "Corrections. If your name, SSN, or address is incorrect, correct Copies B, C, and 2 and ask your employer to correct your employment",
        "record. Be sure to ask the employer to file Form W-2c, Corrected Wage and Tax Statement, with the Social Security Administration (SSA).",
        "Box 12. The following list explains the codes shown in box 12. C - Taxable cost of group-term life insurance over $50,000.",
        "D - Elective deferrals to a section 401(k) cash or deferred arrangement. DD - Cost of employer-sponsored health coverage (not taxable).",
        "W - Employer contributions (including employee contributions through a cafeteria plan) to an employee's health savings account (HSA).",
    ]
    for ln in notice:
        y -= 10
        label(c, X0, y, ln, size=7)


# ----------------------------------------------------------------------------
# generic 1099 layout (NEC / INT / DIV / R)
#   3 columns: payer/recipient (234) | numbered boxes (216) | title + copy (90)
#   cell = (caption, value, kind)  kind: m=money, t=text, c=checkbox,
#          code=1099-R box 7 (value=(code, ira_checked)), c2=1099-R box 2b
# ----------------------------------------------------------------------------
X0_1099, LW_1099, MW_1099, RW_1099 = 36, 234, 216, 90
ROW_1099 = 44


def cell(c, x, y, w, h, cap, val, kind):
    if kind == "m":
        box(c, x, y, w, h, cap, val, align="right")
    elif kind == "t":
        box(c, x, y, w, h, cap, val)
    elif kind == "c":
        box(c, x, y, w, h, cap)
        check(c, x + 5, y + 4, bool(val))
    elif kind == "code":
        code, ira = val
        box(c, x, y, w, h, cap, code)
        label(c, x + 36, y + 5, "IRA/SEP/SIMPLE", size=7)
        check(c, x + w - 14, y + 4, bool(ira))
    elif kind == "c2":
        tnd, total = val
        box(c, x, y, w, h, cap)
        check(c, x + 5, y + 4, bool(tnd))
        label(c, x + 16, y + 5, "Taxable amount not determined", size=7)
        check(c, x + 128, y + 4, bool(total))
        label(c, x + 139, y + 5, "Total distribution", size=7)
    else:
        box(c, x, y, w, h, cap)


def draw_1099(c, meta, payer, payer_tin, recipient, rec_last4, account, rows, state_row):
    X0 = X0_1099
    XM = X0 + LW_1099
    XR = XM + MW_1099
    FW = LW_1099 + MW_1099 + RW_1099
    top = 740
    # ---- left column
    y = top
    y -= 90
    textblock(c, X0, y, LW_1099, 90,
              "PAYER'S name, street address, city or town, state or province, country, ZIP or foreign postal code, and telephone no.",
              [payer[0], payer[1], f"{payer[2]}, {payer[3]} {payer[4]}", payer[5]])
    y -= ROW_1099
    box(c, X0, y, LW_1099 / 2, ROW_1099, "PAYER'S TIN", payer_tin)
    box(c, X0 + LW_1099 / 2, y, LW_1099 / 2, ROW_1099, "RECIPIENT'S TIN", redacted_ssn(rec_last4))
    y -= ROW_1099
    box(c, X0, y, LW_1099, ROW_1099, "RECIPIENT'S name", recipient[0])
    y -= ROW_1099
    box(c, X0, y, LW_1099, ROW_1099, "Street address (including apt. no.)", recipient[1], vfont=VAL)
    y -= ROW_1099
    box(c, X0, y, LW_1099, ROW_1099, "City or town, state or province, country, and ZIP or foreign postal code",
        f"{recipient[2]}, {recipient[3]} {recipient[4]}", vfont=VAL)
    y -= ROW_1099
    box(c, X0, y, LW_1099 - 60, ROW_1099, "Account number (see instructions)", account, vfont=VAL)
    box(c, X0 + LW_1099 - 60, y, 60, ROW_1099, "2nd TIN not.")
    check(c, X0 + LW_1099 - 60 + 5, y + 4, False)
    y_left = y
    # ---- middle column (numbered boxes)
    y = top
    for row in rows:
        y -= ROW_1099
        n = len(row)
        w = MW_1099 / n
        for k, (cap, val, kind) in enumerate(row):
            cell(c, XM + k * w, y, w, ROW_1099, cap, val, kind)
    y_mid = y
    # ---- right column (title / copy)
    box(c, XR, y_mid, RW_1099, top - y_mid, "")
    ty = top - 12
    label(c, XR + RW_1099 / 2, ty, f"OMB No. {meta['omb']}", size=8, align="center")
    ty -= 20
    label(c, XR + RW_1099 / 2, ty, meta["year"], font=CAPB, size=16, align="center")
    ty -= 13
    label(c, XR + RW_1099 / 2, ty, f"Form {meta['form']}", font=CAPB, size=10, align="center")
    if meta.get("rev"):
        ty -= 10
        label(c, XR + RW_1099 / 2, ty, meta["rev"], size=7, align="center")
        ty -= 10
        label(c, XR + RW_1099 / 2, ty, "For calendar year " + meta["year"], size=7, align="center")
    ty -= 12
    for ln in wrap(meta["title"], CAPB, 8, RW_1099 - 6):
        label(c, XR + RW_1099 / 2, ty, ln, font=CAPB, size=8, align="center")
        ty -= 9
    ty -= 8
    label(c, XR + RW_1099 / 2, ty, "Copy B", font=CAPB, size=10, align="center")
    ty -= 11
    label(c, XR + RW_1099 / 2, ty, "For Recipient", font=CAPB, size=8, align="center")
    ty -= 12
    blurb = ("This is important tax information and is being furnished to the IRS. If you are required to "
             "file a return, a negligence penalty or other sanction may be imposed on you if this income is "
             "taxable and the IRS determines that it has not been reported.")
    for ln in wrap(blurb, CAP, 7, RW_1099 - 6):
        if ty < y_mid + 6:
            break
        label(c, XR + 3, ty, ln, size=7)
        ty -= 8
    # ---- full-width bottom row (state boxes)
    y = min(y_left, y_mid) - ROW_1099
    n = len(state_row)
    w = FW / n
    for k, (cap, val, kind) in enumerate(state_row):
        cell(c, X0 + k * w, y, w, ROW_1099, cap, val, kind)
    # ---- footer
    y -= 14
    label(c, X0, y, f"Form {meta['form']}" + (f" {meta['rev']}" if meta.get("rev") else ""), font=CAPB, size=9)
    label(c, X0 + FW / 2, y, "(keep for your records)", size=8, align="center")
    label(c, X0 + FW, y, f"www.irs.gov/Form{meta['form'].replace('-', '')}", size=8, align="right")
    y -= 11
    label(c, X0 + FW, y, "Department of the Treasury - Internal Revenue Service", size=8, align="right")
    y -= 26
    label(c, X0, y, "Instructions for Recipient", font=CAPB, size=8)
    for ln in meta.get("instructions", []):
        y -= 10
        label(c, X0, y, ln, size=7)


def account_no(rnd):
    return f"{rnd.randint(1000, 9999)}-{rnd.randint(100000, 999999)}"


def phone(rnd):
    return f"({rnd.randint(201, 989)}) {rnd.randint(200, 999)}-{rnd.randint(1000, 9999)}"


# ---- 1099-NEC (doc05-doc06) ------------------------------------------------
def nec_data(n):
    rnd = doc_rnd(n)
    payer = NEC_PAYERS[n - 5]
    person = PEOPLE[n - 1]
    year = TAX_YEAR[n]
    ptin = make_ein(rnd)
    last4 = make_ssn_last4(rnd)
    box1 = r2(rnd.uniform(4000, 95000))
    box4 = 0.0 if n == 5 else r2(box1 * 0.24)  # doc06: backup withholding
    d = dict(n=n, year=year, payer=payer + (phone(rnd),), ptin=ptin, person=person, last4=last4,
             account=account_no(rnd), box1=box1, box4=box4)
    gt = {
        "tax_year": year,
        "payer_name": payer[0],
        "payer_tin": ptin,
        "recipient_name": person[0],
        "recipient_tin_last4": last4,
        "box1_nonemployee_comp": canon(box1),
        "box4_fed_withheld": canon(box4),
    }
    return d, gt


def draw_nec(c, d):
    meta = dict(form="1099-NEC", omb="1545-0116", year=d["year"], rev="(Rev. January 2024)",
                title="Nonemployee Compensation",
                instructions=[
                    "You received this form instead of Form W-2 because the payer did not consider you an employee and did not withhold",
                    "income tax or social security and Medicare tax. If you believe you are an employee and cannot get the payer to correct this",
                    "form, report the amount shown in box 1 on the line for 'Wages, salaries, tips, etc.' of Form 1040 and file Form 8919.",
                    "Box 1. Shows nonemployee compensation. If the amount in this box is self-employment (SE) income, report it on Schedule C or F",
                    "(Form 1040) if a sole proprietor, or on Form 1065 and Schedule K-1 (Form 1065) if a partnership, and the recipient/partner",
                    "completes Schedule SE (Form 1040).",
                    "Box 4. Shows backup withholding. A payer must backup withhold on certain payments if you did not give your TIN to the payer.",
                    "Include this amount on your income tax return as tax withheld.",
                ])
    rows = [
        [("1 Nonemployee compensation", money(d["box1"]), "m")],
        [("2 Payer made direct sales totaling $5,000 or more of consumer products to recipient for resale", False, "c")],
        [("3", "", "t"), ("4 Federal income tax withheld", money(d["box4"]) if d["box4"] else "", "m")],
    ]
    state_row = [("5 State tax withheld", "", "m"), ("6 State/Payer's state no.", "", "t"), ("7 State income", "", "m")]
    draw_1099(c, meta, d["payer"], d["ptin"], d["person"], d["last4"], d["account"], rows, state_row)


# ---- 1099-INT (doc07-doc08) ------------------------------------------------
def int_data(n):
    rnd = doc_rnd(n)
    payer = INT_PAYERS[n - 7]
    person = PEOPLE[n - 1]
    year = TAX_YEAR[n]
    ptin = make_ein(rnd)
    last4 = make_ssn_last4(rnd)
    if n == 7:
        box1 = r2(rnd.uniform(50, 6000))
        box3 = r2(rnd.uniform(100, 3000))
        box8 = r2(rnd.uniform(100, 2500))
        box4 = 0.0
    else:
        box1 = r2(rnd.uniform(12, 900))
        box3 = 0.0
        box8 = 0.0
        box4 = r2(box1 * 0.24)  # backup withholding
    d = dict(n=n, year=year, payer=payer + (phone(rnd),), ptin=ptin, person=person, last4=last4,
             account=account_no(rnd), box1=box1, box3=box3, box4=box4, box8=box8)
    gt = {
        "tax_year": year,
        "payer_name": payer[0],
        "recipient_name": person[0],
        "recipient_tin_last4": last4,
        "box1_interest_income": canon(box1),
        "box3_treasury_interest": canon(box3),
        "box4_fed_withheld": canon(box4),
        "box8_tax_exempt_interest": canon(box8),
    }
    return d, gt


def draw_int(c, d):
    meta = dict(form="1099-INT", omb="1545-0112", year=d["year"], rev="(Rev. January 2024)",
                title="Interest Income",
                instructions=[
                    "Box 1. Shows taxable interest paid to you during the calendar year by the payer. This does not include interest shown in box 3.",
                    "Box 3. Shows interest on U.S. Savings Bonds, Treasury bills, Treasury bonds, and Treasury notes. This may or may not all be",
                    "taxable. This interest is exempt from state and local income taxes. This interest is not included in box 1.",
                    "Box 4. Shows backup withholding. Generally, a payer must backup withhold if you did not furnish your TIN or you did not furnish",
                    "the correct TIN to the payer. Include this amount on your income tax return as tax withheld.",
                    "Box 8. Shows tax-exempt interest paid to you during the calendar year by the payer. Report this amount on line 2a of Form 1040.",
                ])
    z = money(0.0)
    rows = [
        [("Payer's RTN (optional)", "", "t"), ("1 Interest income", money(d["box1"]), "m")],
        [("2 Early withdrawal penalty", z, "m"), ("3 Interest on U.S. Savings Bonds and Treasury obligations", money(d["box3"]), "m")],
        [("4 Federal income tax withheld", money(d["box4"]), "m"), ("5 Investment expenses", z, "m")],
        [("6 Foreign tax paid", z, "m"), ("7 Foreign country or U.S. territory", "", "t")],
        [("8 Tax-exempt interest", money(d["box8"]), "m"), ("9 Specified private activity bond interest", z, "m")],
        [("10 Market discount", z, "m"), ("11 Bond premium", z, "m")],
        [("12 Bond premium on Treasury obligations", z, "m"), ("13 Bond premium on tax-exempt bond", z, "m")],
    ]
    state_row = [("14 Tax-exempt and tax credit bond CUSIP no.", "", "t"), ("15 State", "", "t"),
                 ("16 State identification no.", "", "t"), ("17 State tax withheld", "", "m")]
    draw_1099(c, meta, d["payer"], d["ptin"], d["person"], d["last4"], d["account"], rows, state_row)


# ---- 1099-DIV (doc09-doc10) ------------------------------------------------
def div_data(n):
    rnd = doc_rnd(n)
    payer = DIV_PAYERS[n - 9]
    person = PEOPLE[n - 1]
    year = TAX_YEAR[n]
    ptin = make_ein(rnd)
    last4 = make_ssn_last4(rnd)
    box1a = r2(rnd.uniform(300, 18000))
    box1b = r2(box1a * rnd.uniform(0.55, 0.95))
    box2a = r2(rnd.uniform(0, 5000))
    box5 = r2(box1a * rnd.uniform(0.0, 0.10))
    if n == 9:
        box7 = r2(rnd.uniform(15, 400))
        box4 = 0.0
    else:
        box7 = 0.0
        box4 = r2(box1a * 0.24)
    d = dict(n=n, year=year, payer=payer + (phone(rnd),), ptin=ptin, person=person, last4=last4,
             account=account_no(rnd), box1a=box1a, box1b=box1b, box2a=box2a, box4=box4, box5=box5, box7=box7)
    gt = {
        "tax_year": year,
        "payer_name": payer[0],
        "recipient_name": person[0],
        "recipient_tin_last4": last4,
        "box1a_total_ordinary_dividends": canon(box1a),
        "box1b_qualified_dividends": canon(box1b),
        "box2a_total_capital_gain": canon(box2a),
        "box4_fed_withheld": canon(box4),
        "box7_foreign_tax_paid": canon(box7),
    }
    return d, gt


def draw_div(c, d):
    meta = dict(form="1099-DIV", omb="1545-0110", year=d["year"], rev="(Rev. January 2024)",
                title="Dividends and Distributions",
                instructions=[
                    "Box 1a. Shows total ordinary dividends that are taxable. Report this amount on line 3b of Form 1040 or 1040-SR.",
                    "Box 1b. Shows the portion of the amount in box 1a that may be eligible for reduced capital gains rates.",
                    "Box 2a. Shows total capital gain distributions from a regulated investment company (RIC) or real estate investment trust (REIT).",
                    "Box 4. Shows backup withholding. Include this amount on your income tax return as tax withheld.",
                    "Box 7. Shows the foreign tax that you may be able to claim as a deduction or a credit on Form 1040 or 1040-SR.",
                ])
    z = money(0.0)
    rows = [
        [("1a Total ordinary dividends", money(d["box1a"]), "m"), ("1b Qualified dividends", money(d["box1b"]), "m")],
        [("2a Total capital gain distr.", money(d["box2a"]), "m"), ("2b Unrecap. Sec. 1250 gain", z, "m")],
        [("2c Section 1202 gain", z, "m"), ("2d Collectibles (28%) gain", z, "m")],
        [("2e Section 897 ordinary dividends", z, "m"), ("2f Section 897 capital gain", z, "m")],
        [("3 Nondividend distributions", z, "m"), ("4 Federal income tax withheld", money(d["box4"]), "m")],
        [("5 Section 199A dividends", money(d["box5"]), "m"), ("6 Investment expenses", z, "m")],
        [("7 Foreign tax paid", money(d["box7"]), "m"), ("8 Foreign country or U.S. territory", "VARIOUS" if d["box7"] else "", "t")],
        [("9 Cash liquidation distributions", z, "m"), ("10 Noncash liquidation distributions", z, "m")],
        [("11 FATCA filing requirement", False, "c"), ("12 Exempt-interest dividends", z, "m")],
    ]
    state_row = [("13 Specified private activity bond interest dividends", z, "m"), ("14 State", "", "t"),
                 ("15 State identification no.", "", "t"), ("16 State tax withheld", "", "m")]
    draw_1099(c, meta, d["payer"], d["ptin"], d["person"], d["last4"], d["account"], rows, state_row)


# ---- 1099-R (doc11) ---------------------------------------------------------
def r_data(n):
    rnd = doc_rnd(n)
    payer = R_PAYER
    person = PEOPLE[n - 1]
    year = TAX_YEAR[n]
    ptin = make_ein(rnd)
    last4 = make_ssn_last4(rnd)
    gross = r2(rnd.uniform(5000, 60000))
    taxable = gross
    box4 = r2(gross * 0.10)
    code = "7"
    d = dict(n=n, year=year, payer=payer + (phone(rnd),), ptin=ptin, person=person, last4=last4,
             account=account_no(rnd), gross=gross, taxable=taxable, box4=box4, code=code, ira=True)
    gt = {
        "tax_year": year,
        "payer_name": payer[0],
        "recipient_name": person[0],
        "recipient_tin_last4": last4,
        "box1_gross_distribution": canon(gross),
        "box2a_taxable_amount": canon(taxable),
        "box4_fed_withheld": canon(box4),
        "box7_distribution_code": code,
        "ira_sep_simple_checkbox": True,
    }
    return d, gt


def draw_r(c, d):
    meta = dict(form="1099-R", omb="1545-0119", year=d["year"],
                title="Distributions From Pensions, Annuities, Retirement or Profit-Sharing Plans, IRAs, Insurance Contracts, etc.",
                instructions=[
                    "Box 1. Shows the total amount you received this year. The amount may have been a direct rollover, a transfer or conversion to a",
                    "Roth IRA, a recharacterized IRA contribution; or you may have received it as periodic payments, as nonperiodic payments, or as a",
                    "total distribution. Report the amount on Form 1040 or 1040-SR on the line for 'IRA distributions' or 'Pensions and annuities'.",
                    "Box 2a. This part of the distribution is generally taxable. If there is no entry in this box, the payer may not have all the facts",
                    "needed to figure the taxable amount.",
                    "Box 4. Shows federal income tax withheld. Include this amount on your income tax return as tax withheld.",
                    "Box 7. The following codes identify the distribution you received: 1-Early distribution, no known exception; 7-Normal distribution;",
                    "G-Direct rollover of a distribution to a qualified plan. IRA/SEP/SIMPLE box is checked if the distribution is from a traditional IRA.",
                ])
    z = money(0.0)
    rows = [
        [("1 Gross distribution", money(d["gross"]), "m"), ("2a Taxable amount", money(d["taxable"]), "m")],
        [("2b", (False, False), "c2")],
        [("3 Capital gain (included in box 2a)", z, "m"), ("4 Federal income tax withheld", money(d["box4"]), "m")],
        [("5 Employee contributions/Designated Roth contributions or insurance premiums", z, "m"),
         ("6 Net unrealized appreciation in employer's securities", z, "m")],
        [("7 Distribution code(s)", (d["code"], d["ira"]), "code"), ("8 Other", "", "m")],
        [("9a Your percentage of total distribution", "", "t"), ("9b Total employee contributions", z, "m")],
        [("10 Amount allocable to IRR within 5 years", "", "m"), ("11 1st year of desig. Roth contrib.", "", "t")],
        [("12 FATCA filing requirement", False, "c"), ("13 Date of payment", "", "t")],
    ]
    state_row = [("14 State tax withheld", "", "m"), ("15 State/Payer's state no.", "", "t"),
                 ("16 State distribution", "", "m"), ("17 Local tax withheld", "", "m"),
                 ("18 Name of locality", "", "t"), ("19 Local distribution", "", "m")]
    draw_1099(c, meta, d["payer"], d["ptin"], d["person"], d["last4"], d["account"], rows, state_row)


# ----------------------------------------------------------------------------
# consolidated 1099-B summary page (doc12)
# ----------------------------------------------------------------------------
def b_data(n):
    rnd = doc_rnd(n)
    payer = B_PAYER
    person = PEOPLE[n - 1]
    year = TAX_YEAR[n]
    ptin = make_ein(rnd)
    last4 = make_ssn_last4(rnd)

    def sect(p_lo, p_hi, w_lo, w_hi):
        proceeds = r2(rnd.uniform(p_lo, p_hi))
        cost = r2(proceeds * rnd.uniform(0.82, 1.12))
        wash = r2(rnd.uniform(w_lo, w_hi))
        return proceeds, cost, wash

    st_a = sect(20000, 180000, 100, 4000)      # short-term, basis reported (Box A)
    st_b = sect(500, 15000, 0, 0)              # short-term, basis not reported (Box B)
    lt_d = sect(30000, 250000, 50, 1500)       # long-term, basis reported (Box D)
    lt_e = sect(1000, 20000, 0, 0)             # long-term, basis not reported (Box E)
    st_tot = tuple(r2(a + b) for a, b in zip(st_a, st_b))
    lt_tot = tuple(r2(a + b) for a, b in zip(lt_d, lt_e))
    wash_total = r2(st_tot[2] + lt_tot[2])
    box4 = r2(st_b[0] * 0.24)
    d = dict(n=n, year=year, payer=payer + (phone(rnd),), ptin=ptin, person=person, last4=last4,
             account=f"{rnd.randint(100, 999)}-{rnd.randint(10000, 99999)}-{rnd.randint(100, 999)}",
             st_a=st_a, st_b=st_b, st_tot=st_tot, lt_d=lt_d, lt_e=lt_e, lt_tot=lt_tot,
             wash_total=wash_total, box4=box4,
             stmt_date=f"02/{rnd.randint(10, 27):02d}/{int(year) + 1}")
    gt = {
        "tax_year": year,
        "payer_name": payer[0],
        "recipient_name": person[0],
        "recipient_tin_last4": last4,
        "short_term_proceeds": canon(st_tot[0]),
        "short_term_cost_basis": canon(st_tot[1]),
        "long_term_proceeds": canon(lt_tot[0]),
        "long_term_cost_basis": canon(lt_tot[1]),
        "wash_sale_loss_disallowed": canon(wash_total),
        "box4_fed_withheld": canon(box4),
    }
    return d, gt


def draw_b(c, d):
    X0, FW = 36, 540
    payer, p = d["payer"], d["person"]
    # masthead
    label(c, X0, 752, payer[0], font=CAPB, size=14)
    label(c, X0, 741, f"{payer[1]}, {payer[2]}, {payer[3]} {payer[4]}", size=8)
    label(c, X0, 731, f"Client Service: {payer[5]}", size=8)
    label(c, X0 + FW, 752, f"{d['year']} CONSOLIDATED FORMS 1099", font=CAPB, size=11, align="right")
    label(c, X0 + FW, 741, f"Statement Date: {d['stmt_date']}", size=8, align="right")
    label(c, X0 + FW, 731, "Page 1 of 1", size=8, align="right")
    hline(c, X0, X0 + FW, 724, lw=1.2)
    # recipient / account block
    y = 724 - 70
    textblock(c, X0, y, 270, 70, "RECIPIENT'S name, street address, city or town, state, and ZIP code",
              [p[0], p[1], f"{p[2]}, {p[3]} {p[4]}"])
    box(c, X0 + 270, y + 35, 135, 35, "Account Number", d["account"], vfont=VAL)
    box(c, X0 + 405, y + 35, 135, 35, "PAYER'S TIN", d["ptin"], vfont=VAL)
    box(c, X0 + 270, y, 135, 35, "RECIPIENT'S TIN", redacted_ssn(d["last4"]), vfont=VAL)
    box(c, X0 + 405, y, 135, 35, "Tax Year", d["year"], vfont=VAL)
    # section title
    y -= 26
    label(c, X0, y, "SUMMARY OF PROCEEDS, GAINS & LOSSES, ADJUSTMENTS AND WITHHOLDING", font=CAPB, size=10)
    y -= 11
    label(c, X0, y, "Form 1099-B  Proceeds From Broker and Barter Exchange Transactions   OMB No. 1545-0715   Copy B For Recipient", size=8)
    y -= 10
    label(c, X0, y, "This is important tax information and is being furnished to the IRS. If you are required to file a return, a negligence penalty or", size=7)
    y -= 9
    label(c, X0, y, "other sanction may be imposed on you if this income is taxable and the IRS determines that it has not been reported.", size=7)
    # table
    y -= 8
    widths = [170, 80, 80, 64, 72, 74]
    heads = ["", "Proceeds (Box 1d)", "Cost basis (Box 1e)", "Market discount (Box 1f)",
             "Wash sale loss disallowed (Box 1g)", "Net gain or loss (-)"]
    HH = 28
    y -= HH
    x = X0
    c.setLineWidth(0.8)
    for w, h in zip(widths, heads):
        box(c, x, y, w, HH, h, cfont=CAPB)
        x += w
    c.setLineWidth(0.6)

    def row(y, lines, vals, bold=False, h=26):
        x = X0
        c.rect(x, y, widths[0], h)
        c.setFont(CAPB if bold else CAP, 7)
        ty = y + h - 9
        for ln in lines:
            c.drawString(x + 3, ty, ln)
            ty -= 8.5
        x += widths[0]
        c.setFont(VALB if bold else VAL, 9)
        for w, v in zip(widths[1:], vals):
            c.rect(x, y, w, h)
            c.drawRightString(x + w - 4, y + 9, v)
            x += w

    def vals(sec):
        pr, cost, wash = sec
        net = r2(pr - cost + wash)
        return [money(pr), money(cost), money(0.0), money(wash), money(net)]

    rows = [
        (["SHORT-TERM TRANSACTIONS"], None, True, 14),
        (["Box A - basis reported to the IRS", "Report on Form 8949, Part I, with Box A checked"], vals(d["st_a"]), False, 26),
        (["Box B - basis NOT reported to the IRS", "Report on Form 8949, Part I, with Box B checked"], vals(d["st_b"]), False, 26),
        (["Total Short-term"], vals(d["st_tot"]), True, 18),
        (["LONG-TERM TRANSACTIONS"], None, True, 14),
        (["Box D - basis reported to the IRS", "Report on Form 8949, Part II, with Box D checked"], vals(d["lt_d"]), False, 26),
        (["Box E - basis NOT reported to the IRS", "Report on Form 8949, Part II, with Box E checked"], vals(d["lt_e"]), False, 26),
        (["Total Long-term"], vals(d["lt_tot"]), True, 18),
        (["Undetermined term (Box C / Box F)"], [money(0.0)] * 5, False, 18),
    ]
    for lines, v, bold, h in rows:
        y -= h
        if v is None:
            c.rect(X0, y, FW, h)
            c.setFont(CAPB, 8)
            c.drawString(X0 + 3, y + 4, lines[0])
        else:
            row(y, lines, v, bold, h)
    grand = [r2(a + b) for a, b in zip(d["st_tot"], d["lt_tot"])]
    y -= 20
    row(y, ["GRAND TOTAL"], vals(tuple(grand)), True, 20)
    # withholding block
    y -= 34
    label(c, X0, y + 4, "WITHHOLDING", font=CAPB, size=9)
    y -= 22
    box(c, X0, y, 330, 22, "")
    label(c, X0 + 3, y + 8, "Federal income tax withheld (Box 4)", font=CAPB, size=8)
    c.setFont(VALB, 10)
    c.drawRightString(X0 + 330 + 80 - 4, y + 7, money(d["box4"]))
    c.rect(X0 + 330, y, 80, 22)
    box(c, X0 + 410, y, 130, 22, "")
    label(c, X0 + 413, y + 8, "State tax withheld (Box 16)", size=7)
    c.setFont(VAL, 9)
    c.drawRightString(X0 + 540 - 4, y + 7, money(0.0))
    # notes
    y -= 20
    notes = [
        "Wash sale loss disallowed (Box 1g): the disallowed loss has been added to the cost basis of the replacement shares. Amounts shown",
        "in the wash sale column are already reflected in the Net gain or loss column. Report each section on the Form 8949 Part and Box noted.",
        "Proceeds are reported net of commissions and option premiums. Cost basis for noncovered securities (Boxes B and E) is provided for",
        "informational purposes only and is not reported to the IRS; you are responsible for verifying this basis on your tax return.",
        "Gain/loss amounts in this summary are not reported to the IRS. Refer to the detail pages of the Form 1099-B for transaction-level",
        "data, including dates acquired and sold, quantity and CUSIP.",
    ]
    for ln in notes:
        y -= 10
        label(c, X0, y, ln, size=7)
    y -= 18
    hline(c, X0, X0 + FW, y + 10)
    label(c, X0, y, "Form 1099-B (Rev. January 2024)", font=CAPB, size=8)
    label(c, X0 + FW, y, "Department of the Treasury - Internal Revenue Service   www.irs.gov/Form1099B", size=8, align="right")


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------
BUILDERS = {
    1: ("W2", w2_data, draw_w2), 2: ("W2", w2_data, draw_w2),
    3: ("W2", w2_data, draw_w2), 4: ("W2", w2_data, draw_w2),
    5: ("1099NEC", nec_data, draw_nec), 6: ("1099NEC", nec_data, draw_nec),
    7: ("1099INT", int_data, draw_int), 8: ("1099INT", int_data, draw_int),
    9: ("1099DIV", div_data, draw_div), 10: ("1099DIV", div_data, draw_div),
    11: ("1099R", r_data, draw_r),
    12: ("1099B_SUMMARY", b_data, draw_b),
}


def build(n):
    typ, data_fn, draw_fn = BUILDERS[n]
    d, fields = data_fn(n)
    expected = FIELDS[typ]
    missing = [f for f in expected if f not in fields]
    extra = [f for f in fields if f not in expected]
    if missing or extra:
        raise SystemExit(f"doc{n:02d} {typ}: ground-truth field mismatch missing={missing} extra={extra}")
    fields = {f: fields[f] for f in expected}  # fields.json order
    pdf_name = f"doc{n:02d}_{typ}.pdf"
    png_name = f"doc{n:02d}_{typ}.png"
    pdf_path = os.path.join(OUT, pdf_name)
    png_path = os.path.join(OUT, png_name)
    c = canvas.Canvas(pdf_path, pagesize=letter, invariant=1, pageCompression=0)
    c.setTitle(f"Form {typ} tax year {d['year']} (synthetic)")
    c.setAuthor("hf-model-eval synthetic generator")
    draw_fn(c, d)
    c.showPage()
    c.save()
    profile = profile_for(n)
    pdf_to_scan(pdf_path, png_path, profile, seed=n)
    return f"doc{n:02d}", {
        "doc_id": f"doc{n:02d}",
        "doc_type": typ,
        "png": png_name,
        "pdf": pdf_name,
        "scan_profile": profile,
        "fields": fields,
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    gt = {}
    for n in range(1, 13):
        doc_id, rec = build(n)
        gt[doc_id] = rec
        print(f"{doc_id} {rec['doc_type']:<14} {rec['scan_profile']:<10} {rec['png']}")
    gt_path = os.path.join(OUT, "ground_truth_forms.json")
    with open(gt_path, "w") as f:
        json.dump(gt, f, indent=2)
        f.write("\n")
    print("wrote", gt_path)
    for rec in gt.values():
        for k in ("pdf", "png"):
            path = os.path.join(OUT, rec[k])
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                raise SystemExit(f"missing output: {path}")
    print("verified", len(gt), "docs")


if __name__ == "__main__":
    main()
