#!/usr/bin/env python3
"""Synthetic Schedule K-1 / IRS transcript / IRS notice generator (doc13-doc20).

Writes to data/docs/:
    doc13_K1_1065.pdf/.png                      doc17_IRS_ACCOUNT_TRANSCRIPT.pdf/.png
    doc14_K1_1065.pdf/.png                      doc18_CP2000.pdf/.png
    doc15_K1_1120S.pdf/.png                     doc19_CP14.pdf/.png
    doc16_IRS_WAGE_INCOME_TRANSCRIPT.pdf/.png   doc20_CP503.pdf/.png
    ground_truth_k1_irs.json

All identities are synthetic (invented people and businesses). SSNs/TINs are printed
masked as XXX-XX-NNNN like redacted client copies; EINs are full synthetic numbers.
Every document is seeded with random.Random(doc_number) so reruns are byte-identical.
Scan degradation profile = scan_effects.PROFILES[(n-1) % 4], seed=n.

Run:  python gen/gen_docs_k1_irs.py
"""
import json
import os
import random
import sys
from datetime import date, timedelta

from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from scan_effects import PROFILES, pdf_to_scan  # noqa: E402

OUT_DIR = os.path.join(ROOT, "data", "docs")
GT_PATH = os.path.join(OUT_DIR, "ground_truth_k1_irs.json")
with open(os.path.join(ROOT, "eval", "fields.json")) as fh:
    FIELDS = json.load(fh)["doc_types"]

PW, PH = letter  # 612 x 792
CAP, CAPB = "Helvetica", "Helvetica-Bold"
MONO, VAL = "Courier", "Courier-Bold"
CAP_SZ, VAL_SZ = 7.0, 9.5

# --------------------------------------------------------------------------- formatting
def money(cents, paren=True, dollar=False):
    """Printed form: 12,480.00 / (12,480.00) / -$9,100.00"""
    a = abs(cents)
    s = f"{a // 100:,}.{a % 100:02d}"
    if dollar:
        s = "$" + s
    if cents < 0:
        return f"({s})" if paren else f"-{s}"
    return s


def canon(cents):
    """Ground-truth canonical form: 12480.00 / -12480.00"""
    a = abs(cents)
    s = f"{a // 100}.{a % 100:02d}"
    return "-" + s if cents < 0 else s


def long_date(d):
    return f"{d.strftime('%B')} {d.day}, {d.year}"


def mdy(d):
    return d.strftime("%m-%d-%Y")


def irs_abbr(d):
    return f"{d.strftime('%b')}. {d.day:02d}, {d.year}"


def sw(s, font, size):
    return pdfmetrics.stringWidth(s, font, size)


def wrap(s, font, size, maxw):
    words, lines, cur = s.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if not cur or sw(t, font, size) <= maxw:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


# --------------------------------------------------------------------------- identities
FIRST = ["MARGARET", "DANIEL", "PRIYA", "THOMAS", "ELENA", "MARCUS", "SOPHIA", "RAYMOND",
         "NADIA", "JULIAN", "HELEN", "VICTOR"]
LAST = ["OKAFOR", "WHITFIELD", "RAMIREZ", "LINDQVIST", "CASTELLANO", "NAKAMURA", "BRENNAN",
        "DESAI", "HOLLOWAY", "FITZGERALD", "MORAVEC", "ABERNATHY"]
STREETS = ["WILLOW CREST DR", "SYCAMORE LN", "HARBOR POINT RD", "MAPLE RIDGE CT", "FOXGLOVE AVE",
           "CANYON VIEW BLVD", "OLD MILL RD", "LAKESHORE DR", "PRAIRIE WIND WAY", "BIRCHWOOD TER"]
CITIES = [("SPRINGFIELD", "MO", "65807"), ("TAMPA", "FL", "33602"), ("DAYTON", "OH", "45414"),
          ("BOISE", "ID", "83702"), ("ALBUQUERQUE", "NM", "87102"), ("RALEIGH", "NC", "27601"),
          ("TUCSON", "AZ", "85701"), ("MADISON", "WI", "53703"), ("RENO", "NV", "89501"),
          ("CHATTANOOGA", "TN", "37402")]
PARTNERSHIPS = ["NORTHGATE LOGISTICS LLC", "BLUE HERON PROPERTIES LP", "CEDAR RIDGE DENTAL PARTNERS LLC",
                "HALVORSEN PRICE CONSULTING LLC", "SUMMIT CREEK HOLDINGS LP", "RIVERBEND FAMILY FARMS LLC"]
SCORPS = ["PINECREST AUTOMOTIVE INC", "MARLOWE DESIGN STUDIO INC", "KESTREL HVAC SERVICES INC",
          "BRIGHTWATER PHYSICAL THERAPY PC"]
EMPLOYERS = ["TRIDENT MEDICAL GROUP INC", "OAKLINE BUILDERS LLC", "GRANITE PEAK SOFTWARE INC",
             "LAKESHORE HOSPITALITY GROUP LLC", "MERIDIAN FREIGHT SYSTEMS INC", "COPPERFIELD PUBLIC SCHOOLS"]
BANKS = ["FIRST MERIDIAN BANK", "CASCADE COMMUNITY CREDIT UNION", "HARBORVIEW SAVINGS BANK"]
NEC_PAYERS = ["VELOCITY MARKETING PARTNERS LLC", "SILVERLINE CONSULTING GROUP INC",
              "REDWOOD CREATIVE AGENCY LLC"]


def person(rng):
    first, last = rng.choice(FIRST), rng.choice(LAST)
    mi = rng.choice(["", "", rng.choice("ABCDEJKLMRST")])
    city, state, z = rng.choice(CITIES)
    return {
        "name": " ".join(x for x in (first, mi, last) if x),
        "street": f"{rng.randint(100, 9899)} {rng.choice(STREETS)}",
        "csz": f"{city}, {state} {z}",
        "csz_notice": f"{city} {state} {z}-{rng.randint(1000, 9999):04d}",
        "ssn4": f"{rng.randint(1000, 9999)}",
    }


def business(rng, pool):
    city, state, z = rng.choice(CITIES)
    return {
        "name": rng.choice(pool),
        "ein": f"{rng.randint(10, 99)}-{rng.randint(1000000, 9999999)}",
        "street": f"{rng.randint(100, 9899)} {rng.choice(STREETS)}",
        "csz": f"{city}, {state} {z}",
    }


# --------------------------------------------------------------------------- page primitives
class Page:
    """Thin canvas wrapper; all coordinates are (x, top) measured from the top-left corner."""

    def __init__(self, path, title):
        self.c = canvas.Canvas(path, pagesize=letter)
        self.c.setTitle(title)

    def text(self, x, top, s, font=CAP, size=CAP_SZ, align="left", gray=0.0):
        self.c.setFillGray(gray)
        self.c.setFont(font, size)
        y = PH - top
        if align == "left":
            self.c.drawString(x, y, s)
        elif align == "right":
            self.c.drawRightString(x, y, s)
        else:
            self.c.drawCentredString(x, y, s)
        self.c.setFillGray(0)

    def rect(self, x, top, w, h, fill=None, lw=0.5):
        self.c.setLineWidth(lw)
        if fill is not None:
            self.c.setFillGray(fill)
        self.c.rect(x, PH - top - h, w, h, stroke=1, fill=1 if fill is not None else 0)
        self.c.setFillGray(0)

    def line(self, x1, top1, x2, top2, lw=0.5, dashed=False):
        self.c.setLineWidth(lw)
        if dashed:
            self.c.setDash([3, 3])
        self.c.line(x1, PH - top1, x2, PH - top2)
        if dashed:
            self.c.setDash([])

    def bar(self, x, top, w, h, lines, size=8):
        """Black section header bar with white bold text (one or more lines)."""
        self.rect(x, top, w, h, fill=0.15)
        if isinstance(lines, str):
            lines = [lines]
        y = top + 9.5
        for ln in lines:
            self.text(x + 4, y, ln, CAPB, size, gray=1.0)
            y += size + 1.5

    def checkbox(self, x, top, checked, size=7):
        self.rect(x, top, size, size, lw=0.5)
        if checked:
            self.text(x + size / 2, top + size - 1, "X", VAL, size, align="center")

    def para(self, x, top, w, s, font=CAP, size=9, leading=11):
        for ln in wrap(s, font, size, w):
            self.text(x, top, ln, font, size)
            top += leading
        return top

    def mono_lines(self, x, top, lines, size=9, leading=9.6, center_w=None):
        """Lines of Courier text. A line given as ('c', text) is centred on center_w."""
        for ln in lines:
            if isinstance(ln, tuple):
                self.text(x + center_w / 2, top, ln[1], MONO, size, align="center")
            elif ln:
                self.text(x, top, ln, MONO, size)
            top += leading
        return top

    def save(self):
        self.c.showPage()
        self.c.save()


def grid_box(p, x, top, w, h, num, caption, entries=(), check=None):
    """Part III style numbered box: number+caption at top, code/amount lines below."""
    p.rect(x, top, w, h)
    nx = x + 3
    p.text(nx, top + 8.5, num, CAPB, CAP_SZ + 0.5)
    cx = nx + sw(num, CAPB, CAP_SZ + 0.5) + 3
    for i, ln in enumerate(wrap(caption, CAP, CAP_SZ, x + w - cx - 3)):
        p.text(cx, top + 8.5 + i * 7.8, ln, CAP, CAP_SZ)
    y = top + 26.5
    for code, val in entries:
        assert y + 1 <= top + h, f"box {num}: too many entries"
        if code:
            p.text(nx, y, code, VAL, VAL_SZ)
        if val:
            p.text(x + w - 3.5, y, val, VAL, VAL_SZ, align="right")
        y += 11
    if check is not None:
        p.checkbox(x + w - 12, top + h - 12, check)


def field(p, x, top, w, label, caption, values=()):
    """Part I/II style lettered field with caption and typed value lines. Returns height."""
    cap_lines = wrap(caption, CAP, CAP_SZ, w - 20)
    h = 4 + len(cap_lines) * 8 + len(values) * 11 + 3
    p.rect(x, top, w, h)
    p.text(x + 3, top + 9.5, label, CAPB, CAP_SZ + 0.5)
    cx = x + 15
    for i, ln in enumerate(cap_lines):
        p.text(cx, top + 9.5 + i * 8, ln, CAP, CAP_SZ)
    y = top + 9.5 + (len(cap_lines) - 1) * 8 + 11.5
    for v in values:
        p.text(cx, y, v, VAL, VAL_SZ)
        y += 11
    return h


def check_row(p, x, top, w, label, items):
    """Lettered row of checkbox+text pairs: items = [(checked, text, x_offset), ...]. Returns height."""
    h = 15
    p.rect(x, top, w, h)
    if label:
        p.text(x + 3, top + 9.5, label, CAPB, CAP_SZ + 0.5)
    for checked, txt, off in items:
        p.checkbox(x + off, top + 3.5, checked)
        p.text(x + off + 10, top + 9.5, txt, CAP, CAP_SZ)
    return h


def mini_table(p, x, top, w, label, caption, col_titles, rows, extra=()):
    """Lettered field with a Beginning/Ending style two-column table. Returns height."""
    h = 10 + 9 + len(rows) * 11 + len(extra) * 11 + 3
    p.rect(x, top, w, h)
    p.text(x + 3, top + 9.5, label, CAPB, CAP_SZ + 0.5)
    p.text(x + 15, top + 9.5, caption, CAP, CAP_SZ)
    xr = [x + w - 95, x + w - 5]
    y = top + 18.5
    for xe, t in zip(xr, col_titles):
        p.text(xe, y, t, CAPB, CAP_SZ, align="right")
    p.line(x + 15, y + 2, x + w - 3, y + 2)
    y += 10.5
    for cap, v1, v2 in rows:
        p.text(x + 15, y, cap, CAP, CAP_SZ)
        p.text(xr[0], y, v1, VAL, VAL_SZ, align="right")
        p.text(xr[1], y, v2, VAL, VAL_SZ, align="right")
        y += 11
    for checked, txt in extra:
        if checked is not None:
            p.checkbox(x + 15, y - 6.5, checked)
            p.text(x + 27, y, txt, CAP, CAP_SZ)
        else:
            p.text(x + 15, y, txt, CAP, CAP_SZ)
        y += 11
    return h


def amount_table(p, x, top, w, label, caption, rows):
    """Lettered field with caption + rows of (text, amount right-aligned). Returns height."""
    h = 10 + len(rows) * 11 + 3
    p.rect(x, top, w, h)
    p.text(x + 3, top + 9.5, label, CAPB, CAP_SZ + 0.5)
    p.text(x + 15, top + 9.5, caption, CAP, CAP_SZ)
    y = top + 20.5
    for cap, v in rows:
        p.text(x + 15, y, cap, CAP, CAP_SZ)
        if v is not None:
            p.text(x + w - 5, y, v, VAL, VAL_SZ, align="right")
        y += 11
    return h


def k1_header(p, form_no, year, title, code_no):
    p.rect(30, 28, 552, 722, lw=1)
    p.line(320, 28, 320, 750)
    p.text(33, 37, code_no, CAP, 6.5)
    p.text(33, 56, "Schedule K-1", CAPB, 13)
    p.text(33, 67, f"({form_no})", CAP, 9)
    p.text(33, 79, "Department of the Treasury", CAP, 7)
    p.text(33, 87, "Internal Revenue Service", CAP, 7)
    p.text(33, 99, f"For calendar year {year}, or tax year", CAP, 7)
    p.text(33, 108, f"beginning ___________ {year}    ending ___________ 20____", CAP, 7)
    p.text(316, 64, year, CAPB, 22, align="right")
    p.line(30, 114, 320, 114)
    p.text(33, 125, title, CAPB, 9.5)
    p.text(316, 134, "See separate instructions.", CAP, 7, align="right")
    p.line(30, 140, 320, 140)
    return 140


def k1_footer(p, form_no, year, cat_no, url):
    p.text(30, 760, f"For Paperwork Reduction Act Notice, see the Instructions for {form_no}.", CAP, 7)
    p.text(345, 760, url, CAP, 7)
    p.text(445, 760, f"Cat. No. {cat_no}", CAP, 7)
    p.text(582, 760, f"Schedule K-1 ({form_no}) {year}", CAP, 7, align="right")


def part3_grid(p, rx, top, rw, row_h, left_boxes, right_boxes):
    """left_boxes: [(num, caption, entries)], right_boxes: [(num, caption, rows, entries, check)]"""
    half = rw / 2
    y = top
    for num, cap, entries in left_boxes:
        grid_box(p, rx, y, half, row_h, num, cap, entries)
        y += row_h
    y2 = top
    for num, cap, rows, entries, check in right_boxes:
        grid_box(p, rx + half, y2, half, row_h * rows, num, cap, entries, check)
        y2 += row_h * rows
    assert abs(y - y2) < 0.01, "Part III columns must be the same height"
    return y


def irs_use_only(p, x, top, w, h):
    p.rect(x, top, w, h)
    p.text(x + 4, top + 11, "For IRS Use Only", CAPB, 7.5)


def assemble(doc_type, values):
    """Build the ground-truth fields dict in fields.json order; every field must be present."""
    out = {}
    for f in FIELDS[doc_type]:
        assert f in values, f"{doc_type}: missing field {f}"
        out[f] = values[f]
    extra = set(values) - set(FIELDS[doc_type])
    assert not extra, f"{doc_type}: unexpected fields {extra}"
    return out


# =========================================================================== Schedule K-1 (Form 1065)
def gen_k1_1065(n, pdf_path, partner_type, loss):
    rng = random.Random(n)
    year = "2025"
    pship = business(rng, PARTNERSHIPS)
    partner = person(rng)
    pct = rng.choice([20, 25, 30, 35, 40, 50])
    pct_s = f"{pct:.2f}"
    if loss:
        box1 = -rng.randint(5000, 20000) * 100
    else:
        box1 = rng.randint(30000, 90000) * 100
    box2 = rng.randint(2000, 16000) * 100
    box4a = rng.randint(9000, 42000) * 100
    box5 = rng.randint(80, 2500) * 100
    box14a = box1 + box4a if partner_type == "GENERAL" else box4a
    box19a = rng.randint(5000, 45000) * 100
    cap_begin = rng.randint(60000, 220000) * 100
    cap_contrib = rng.choice([0, rng.randint(2000, 15000) * 100])
    net_inc = box1 + box2 + box5
    cap_end = cap_begin + cap_contrib + net_inc - box19a
    liab_nr = rng.randint(10000, 90000) * 100
    liab_rec = rng.randint(0, 30000) * 100

    p = Page(pdf_path, f"Schedule K-1 (Form 1065) {year}")
    k1_header(p, "Form 1065", year, "Partner's Share of Income, Deductions, Credits, etc.", "651123")

    # ---- right column: Part III
    rx, rw = 322, 258
    p.text(rx + 4, 39, "Final K-1", CAP, 7)
    p.checkbox(rx + 36, 32.5, False)
    p.text(rx + 50, 39, "Amended K-1", CAP, 7)
    p.checkbox(rx + 94, 32.5, False)
    p.text(rx + rw - 2, 39, "OMB No. 1545-0123", CAP, 7, align="right")
    p.bar(rx, 44, rw, 22, ["Part III   Partner's Share of Current Year Income,",
                           "Deductions, Credits, and Other Items"], 7.5)
    left_boxes = [
        ("1", "Ordinary business income (loss)", [("", money(box1))]),
        ("2", "Net rental real estate income (loss)", [("", money(box2))]),
        ("3", "Other net rental income (loss)", []),
        ("4a", "Guaranteed payments for services", [("", money(box4a))]),
        ("4b", "Guaranteed payments for capital", []),
        ("4c", "Total guaranteed payments", [("", money(box4a))]),
        ("5", "Interest income", [("", money(box5))]),
        ("6a", "Ordinary dividends", []),
        ("6b", "Qualified dividends", []),
        ("6c", "Dividend equivalents", []),
        ("7", "Royalties", []),
        ("8", "Net short-term capital gain (loss)", []),
        ("9a", "Net long-term capital gain (loss)", []),
        ("9b", "Collectibles (28%) gain (loss)", []),
        ("9c", "Unrecaptured section 1250 gain", []),
        ("10", "Net section 1231 gain (loss)", []),
        ("11", "Other income (loss)", []),
        ("12", "Section 179 deduction", []),
        ("13", "Other deductions", []),
    ]
    right_boxes = [
        ("14", "Self-employment earnings (loss)", 2, [("A", money(box14a))], None),
        ("15", "Credits", 2, [], None),
        ("16", "Schedule K-3 is attached if checked", 1, [], False),
        ("17", "Alternative minimum tax (AMT) items", 2, [], None),
        ("18", "Tax-exempt income and nondeductible expenses", 2, [], None),
        ("19", "Distributions", 2, [("A", money(box19a))], None),
        ("20", "Other information", 5, [("A", money(box5)), ("Z", "STMT")], None),
        ("21", "Foreign taxes paid or accrued", 1, [], None),
        ("22", "More than one activity for at-risk purposes*", 1, [], False),
        ("23", "More than one activity for passive activity purposes*", 1, [], False),
    ]
    gy = part3_grid(p, rx, 66, rw, 30, left_boxes, right_boxes)
    p.text(rx + 4, gy + 12, "*See attached statement for additional information.", CAP, 7)

    # ---- left column: Part I / Part II
    x, w, top = 30, 290, 140
    p.bar(x, top, w, 13, "Part I       Information About the Partnership")
    top += 13
    top += field(p, x, top, w, "A", "Partnership's employer identification number", [pship["ein"]])
    top += field(p, x, top, w, "B", "Partnership's name, address, city, state, and ZIP code",
                 [pship["name"], pship["street"], pship["csz"]])
    top += field(p, x, top, w, "C", "IRS center where partnership filed return:", ["e-file"])
    top += check_row(p, x, top, w, "D", [(False, "Check if this is a publicly traded partnership (PTP)", 15)])
    p.bar(x, top, w, 13, "Part II      Information About the Partner")
    top += 13
    top += field(p, x, top, w, "E",
                 "Partner's SSN or TIN (Do not use TIN of a disregarded entity. See instructions.)",
                 [f"XXX-XX-{partner['ssn4']}"])
    top += field(p, x, top, w, "F",
                 "Name, address, city, state, and ZIP code for partner entered in E. See instructions.",
                 [partner["name"], partner["street"], partner["csz"]])
    top += check_row(p, x, top, w, "G", [
        (partner_type == "GENERAL", "General partner or LLC member-manager", 15),
        (partner_type == "LIMITED", "Limited partner or other LLC member", 155)])
    top += check_row(p, x, top, w, "H1", [(True, "Domestic partner", 15), (False, "Foreign partner", 155)])
    top += check_row(p, x, top, w, "H2", [
        (False, "If the partner is a disregarded entity (DE), enter the partner's:   TIN ________   Name ____________", 15)])
    top += field(p, x, top, w, "I1", "What type of entity is this partner?", ["INDIVIDUAL"])
    top += check_row(p, x, top, w, "I2", [
        (False, "If this partner is a retirement plan (IRA/SEP/Keogh/etc.), check here", 15)])
    top += mini_table(p, x, top, w, "J", "Partner's share of profit, loss, and capital (see instructions):",
                      ("Beginning", "Ending"),
                      [("Profit", f"{pct_s} %", f"{pct_s} %"), ("Loss", f"{pct_s} %", f"{pct_s} %"),
                       ("Capital", f"{pct_s} %", f"{pct_s} %")],
                      extra=[(False, "Check if decrease is due to:  Sale or  Exchange of partnership interest")])
    top += mini_table(p, x, top, w, "K1", "Partner's share of liabilities:", ("Beginning", "Ending"),
                      [("Nonrecourse  $", money(liab_nr), money(liab_nr)),
                       ("Qualified nonrecourse financing  $", "0.00", "0.00"),
                       ("Recourse  $", money(liab_rec), money(liab_rec))],
                      extra=[(False, "Check this box if item K1 includes liability amounts from lower-tier partnerships"),
                             (False, "Check if any of the above liability is subject to guarantees or other payment obligations by the partner")])
    top += amount_table(p, x, top, w, "L", "Partner's Capital Account Analysis", [
        ("Beginning capital account", money(cap_begin)),
        ("Capital contributed during the year", money(cap_contrib)),
        ("Current year net income (loss)", money(net_inc)),
        ("Other increase (decrease) (attach explanation)", "0.00"),
        ("Withdrawals and distributions", money(-box19a, paren=True)),
        ("Ending capital account", money(cap_end)),
    ])
    top += check_row(p, x, top, w, "M", [
        (False, "Yes", 205), (True, "No", 235)])
    p.text(x + 15, top - 5.5, "Did the partner contribute property with a built-in gain (loss)?", CAP, CAP_SZ)
    top += amount_table(p, x, top, w, "N", "Partner's Share of Net Unrecognized Section 704(c) Gain or (Loss)", [
        ("Beginning", "0.00"), ("Ending", "0.00")])
    if top < 710:
        irs_use_only(p, x, top, w, 750 - top)
    k1_footer(p, "Form 1065", year, "11394R", "www.irs.gov/Form1065")
    p.save()

    return assemble("K1_1065", {
        "tax_year": year,
        "partnership_name": pship["name"],
        "partnership_ein": pship["ein"],
        "partner_name": partner["name"],
        "partner_tin_last4": partner["ssn4"],
        "partner_type": partner_type,
        "profit_share_pct_ending": pct_s,
        "box1_ordinary_income": canon(box1),
        "box2_net_rental_real_estate": canon(box2),
        "box4a_guaranteed_payments_services": canon(box4a),
        "box5_interest_income": canon(box5),
        "box14a_self_employment_earnings": canon(box14a),
        "box19a_distributions": canon(box19a),
    })


# =========================================================================== Schedule K-1 (Form 1120-S)
def gen_k1_1120s(n, pdf_path):
    rng = random.Random(n)
    year = "2025"
    corp = business(rng, SCORPS)
    sh = person(rng)
    total_shares = rng.choice([100, 1000, 10000])
    pct = rng.choice([25, 40, 50, 60, 75, 100])
    pct_s = f"{pct:.2f}"
    sh_shares = total_shares * pct // 100
    box1 = rng.randint(40000, 160000) * 100
    box2 = rng.randint(3000, 24000) * 100
    box4 = rng.randint(120, 4000) * 100
    box5a = rng.randint(200, 6000) * 100
    box5b = box5a * rng.choice([60, 75, 90]) // 100
    box16c = rng.randint(300, 2500) * 100
    box16d = rng.randint(10000, 95000) * 100
    gross = rng.randint(400000, 2500000) * 100

    p = Page(pdf_path, f"Schedule K-1 (Form 1120-S) {year}")
    k1_header(p, "Form 1120-S", year, "Shareholder's Share of Income, Deductions, Credits, etc.", "671123")

    rx, rw = 322, 258
    p.text(rx + 4, 39, "Final K-1", CAP, 7)
    p.checkbox(rx + 36, 32.5, False)
    p.text(rx + 50, 39, "Amended K-1", CAP, 7)
    p.checkbox(rx + 94, 32.5, False)
    p.text(rx + rw - 2, 39, "OMB No. 1545-0123", CAP, 7, align="right")
    p.bar(rx, 44, rw, 22, ["Part III   Shareholder's Share of Current Year Income,",
                           "Deductions, Credits, and Other Items"], 7.5)
    left_boxes = [
        ("1", "Ordinary business income (loss)", [("", money(box1))]),
        ("2", "Net rental real estate income (loss)", [("", money(box2))]),
        ("3", "Other net rental income (loss)", []),
        ("4", "Interest income", [("", money(box4))]),
        ("5a", "Ordinary dividends", [("", money(box5a))]),
        ("5b", "Qualified dividends", [("", money(box5b))]),
        ("6", "Royalties", []),
        ("7", "Net short-term capital gain (loss)", []),
        ("8a", "Net long-term capital gain (loss)", []),
        ("8b", "Collectibles (28%) gain (loss)", []),
        ("8c", "Unrecaptured section 1250 gain", []),
        ("9", "Net section 1231 gain (loss)", []),
        ("10", "Other income (loss)", []),
        ("11", "Section 179 deduction", []),
        ("12", "Other deductions", []),
    ]
    right_boxes = [
        ("13", "Credits", 3, [], None),
        ("14", "Schedule K-3 is attached if checked", 1, [], False),
        ("15", "Alternative minimum tax (AMT) items", 2, [], None),
        ("16", "Items affecting shareholder basis", 3, [("C", money(box16c)), ("D", money(box16d))], None),
        ("17", "Other information", 4, [("V", "STMT"), ("AC", money(gross))], None),
        ("18", "More than one activity for at-risk purposes*", 1, [], False),
        ("19", "More than one activity for passive activity purposes*", 1, [], False),
    ]
    gy = part3_grid(p, rx, 66, rw, 34, left_boxes, right_boxes)
    p.text(rx + 4, gy + 12, "*See attached statement for additional information.", CAP, 7)

    x, w, top = 30, 290, 140
    p.bar(x, top, w, 13, "Part I       Information About the Corporation")
    top += 13
    top += field(p, x, top, w, "A", "Corporation's employer identification number", [corp["ein"]])
    top += field(p, x, top, w, "B", "Corporation's name, address, city, state, and ZIP code",
                 [corp["name"], corp["street"], corp["csz"]])
    top += field(p, x, top, w, "C", "IRS center where corporation filed return:", ["e-file"])
    top += mini_table(p, x, top, w, "D", "Corporation's total number of shares", ("Beginning of tax year", "End of tax year"),
                      [("", f"{total_shares:,}", f"{total_shares:,}")])
    p.bar(x, top, w, 13, "Part II      Information About the Shareholder")
    top += 13
    top += field(p, x, top, w, "E", "Shareholder's identifying number", [f"XXX-XX-{sh['ssn4']}"])
    top += field(p, x, top, w, "F", "Shareholder's name, address, city, state, and ZIP code",
                 [sh["name"], sh["street"], sh["csz"]])
    top += field(p, x, top, w, "G", "Current year allocation percentage", [f"{pct_s} %"])
    top += mini_table(p, x, top, w, "H", "Shareholder's number of shares", ("Beginning of tax year", "End of tax year"),
                      [("", f"{sh_shares:,}", f"{sh_shares:,}")])
    top += mini_table(p, x, top, w, "I", "Loans from shareholder", ("Beginning of tax year", "End of tax year"),
                      [("$", "0.00", "0.00")])
    irs_use_only(p, x, top + 8, w, 750 - top - 8)
    k1_footer(p, "Form 1120-S", year, "11520D", "www.irs.gov/Form1120S")
    p.save()

    return assemble("K1_1120S", {
        "tax_year": year,
        "corporation_name": corp["name"],
        "corporation_ein": corp["ein"],
        "shareholder_name": sh["name"],
        "shareholder_tin_last4": sh["ssn4"],
        "ownership_pct": pct_s,
        "box1_ordinary_income": canon(box1),
        "box2_net_rental_real_estate": canon(box2),
        "box4_interest_income": canon(box4),
        "box5a_ordinary_dividends": canon(box5a),
        "box16d_distributions": canon(box16d),
    })


# =========================================================================== IRS Wage and Income Transcript
def gen_wage_income_transcript(n, pdf_path):
    rng = random.Random(n)
    year = "2025"
    tp = person(rng)
    req = date(2026, rng.randint(2, 9), rng.randint(1, 28))
    tracking = f"1{rng.randint(10**10, 10**11 - 1)}"
    pool = list(EMPLOYERS)
    rng.shuffle(pool)
    w2s = []
    for name in pool[:2]:
        emp = business(rng, [name])
        wages = rng.randint(28000, 140000) * 100
        fed = wages * rng.randint(9, 19) // 100 // 100 * 100
        w2s.append({
            "emp": emp, "wages": wages, "fed": fed,
            "ss_tax": round(wages * 0.062 / 100) * 100,
            "med_tax": round(wages * 0.0145 / 100) * 100,
            "deferred": rng.choice([0, rng.randint(1500, 12000) * 100]),
            "dd": rng.randint(2000, 11000) * 100,
        })
    bank = business(rng, BANKS)
    interest = rng.randint(60, 3200) * 100
    nec_payer = business(rng, NEC_PAYERS)
    nec_amt = rng.randint(4000, 38000) * 100
    masked_ssn = f"XXX-XX-{tp['ssn4']}"

    def kv(k, v, width=44):
        return f"{k:<{width}}{v}"

    def emp_block(label, b, idlabel):
        return [f"{label}:", f"    {idlabel}  {b['ein']}", f"    {b['name']}", f"    {b['street']}",
                f"    {b['csz']}"]

    def person_block(label, idlabel):
        return [f"{label}:", f"    {idlabel}  {masked_ssn}", f"    {tp['name']}"]

    L = [
        ("c", "This Product Contains Sensitive Taxpayer Data"),
        "",
        ("c", "Wage and Income Transcript"),
        "",
        kv("Request Date:", mdy(req), 26),
        kv("Response Date:", mdy(req), 26),
        kv("Tracking Number:", tracking, 26),
        "",
        kv("SSN Provided:", masked_ssn, 26),
        kv("Tax Period Requested:", f"December, {year}", 26),
        "",
    ]
    for w in w2s:
        L += [("c", "Form W-2 Wage and Tax Statement")]
        L += emp_block("Employer", w["emp"], "Employer Identification Number (EIN):")
        L += person_block("Employee", "Employee's Social Security Number:")
        L += [
            kv("Submission Type:", "Original document"),
            kv("Wages, Tips and Other Compensation:", money(w["wages"], dollar=True)),
            kv("Federal Income Tax Withheld:", money(w["fed"], dollar=True)),
            kv("Social Security Wages:", money(w["wages"], dollar=True)),
            kv("Social Security Tax Withheld:", money(w["ss_tax"], dollar=True)),
            kv("Medicare Wages and Tips:", money(w["wages"], dollar=True)),
            kv("Medicare Tax Withheld:", money(w["med_tax"], dollar=True)),
            kv('Code "DD" Cost of Employer-Sponsored Health Coverage:', money(w["dd"], dollar=True), 56),
            kv("Retirement Plan Indicator:", "Yes" if w["deferred"] else "Unanswered"),
            "",
        ]
    L += [("c", "Form 1099-INT")]
    L += emp_block("Payer", bank, "Payer's Federal Identification Number (FIN):")
    L += person_block("Recipient", "Recipient's Identification Number:")
    L += [
        kv("Submission Type:", "Original document"),
        kv("Interest:", money(interest, dollar=True)),
        kv("Tax Withheld:", "$0.00"),
        kv("Tax-Exempt Interest:", "$0.00"),
        "",
        ("c", "Form 1099-NEC"),
    ]
    L += emp_block("Payer", nec_payer, "Payer's Federal Identification Number (FIN):")
    L += person_block("Recipient", "Recipient's Identification Number:")
    L += [
        kv("Submission Type:", "Original document"),
        kv("Non-Employee Compensation:", money(nec_amt, dollar=True)),
        kv("Federal Income Tax Withheld:", "$0.00"),
        kv("Direct Sales Indicator:", "No"),
    ]
    p = Page(pdf_path, "Wage and Income Transcript")
    end = p.mono_lines(36, 34, L, size=9, leading=9.4, center_w=540)
    assert end < 780, f"wage transcript overflow: {end}"
    p.save()

    return assemble("IRS_WAGE_INCOME_TRANSCRIPT", {
        "tax_year": year,
        "taxpayer_ssn_last4": tp["ssn4"],
        "taxpayer_name": tp["name"],
        "form_count": "4",
        "w2_1_employer_name": w2s[0]["emp"]["name"],
        "w2_1_wages": canon(w2s[0]["wages"]),
        "w2_1_fed_withheld": canon(w2s[0]["fed"]),
        "int_1_payer_name": bank["name"],
        "int_1_interest": canon(interest),
        "nec_1_payer_name": nec_payer["name"],
        "nec_1_amount": canon(nec_amt),
    })


# =========================================================================== IRS Account Transcript
BRACKETS = {  # 2025 ordinary-income brackets (10/12/22 %), standard deduction
    "SINGLE": (1500000, [(1192500, 10), (4847500, 12), (10335000, 22)]),
    "MARRIED FILING JOINT": (3150000, [(2385000, 10), (9695000, 12), (20670000, 22)]),
    "HEAD OF HOUSEHOLD": (2250000, [(1700000, 10), (6485000, 12), (10335000, 22)]),
}
FS_PRINT = {"SINGLE": "Single", "MARRIED FILING JOINT": "Married Filing Joint",
            "HEAD OF HOUSEHOLD": "Head of Household"}


def income_tax(taxable, brackets):
    tax, lo = 0, 0
    for hi, rate in brackets:
        if taxable <= lo:
            break
        tax += (min(taxable, hi) - lo) * rate // 100
        lo = hi
    return tax // 100 * 100  # whole dollars


def gen_account_transcript(n, pdf_path):
    rng = random.Random(n)
    year = "2025"
    tp = person(rng)
    fs = rng.choice(list(BRACKETS))
    std, br = BRACKETS[fs]
    agi = rng.randint(42000, 180000) * 100
    taxable = max(0, agi - std)
    tax = income_tax(taxable, br)
    withholding = round(tax * rng.uniform(1.04, 1.32) / 100) * 100
    credit = rng.choice([1700, 2000, 1000, 0]) * 100
    refund = withholding + credit - tax
    assert refund > 0
    proc = date(2026, rng.randint(2, 4), rng.randint(1, 28))
    refund_date = proc + timedelta(days=2)
    req = proc + timedelta(days=rng.randint(45, 160))
    asof = req + timedelta(days=(7 - req.weekday()) % 7 or 7)  # next Monday
    cycle = f"{proc.year}{proc.isocalendar()[1]:02d}0{rng.randint(1, 5)}"
    tracking = f"1{rng.randint(10**10, 10**11 - 1)}"
    dln = f"{rng.randint(10000, 99999)}-{rng.randint(100, 999)}-{rng.randint(10000, 99999)}-{rng.randint(0, 9)}"
    spouse4 = f"{rng.randint(1000, 9999)}" if fs == "MARRIED FILING JOINT" else ""
    exemptions = "02" if fs != "SINGLE" else "01"

    def kv(k, v, width=46):
        return f"{k:<{width}}{v:>14}"

    def tx(code, expl, cyc, d, amt):
        return f"{code:<5}{expl:<42}{cyc:<10}{d:<13}{amt:>13}"

    L = [
        ("c", "This Product Contains Sensitive Taxpayer Data"),
        "",
        ("c", "Account Transcript"),
        "",
        f"{'Request Date:':<30}{mdy(req)}",
        f"{'Response Date:':<30}{mdy(req)}",
        f"{'Tracking Number:':<30}{tracking}",
        "",
        f"{'FORM NUMBER:':<30}1040",
        f"{'TAX PERIOD:':<30}Dec. 31, {year}",
        "",
        f"{'TAXPAYER IDENTIFICATION NUMBER:':<40}XXX-XX-{tp['ssn4']}",
        f"{'SPOUSE TAXPAYER IDENTIFICATION NUMBER:':<40}" + (f"XXX-XX-{spouse4}" if spouse4 else ""),
        "",
        tp["name"], tp["street"], tp["csz"],
        "",
        ("c", "<<<<POWER OF ATTORNEY/TAX INFORMATION AUTHORIZATION (POA/TIA) ON FILE>>>>"),
        "",
        "--- ANY MINUS SIGN SHOWN BELOW SIGNIFIES A CREDIT AMOUNT ---",
        "",
        kv("ACCOUNT BALANCE:", "0.00"),
        kv("ACCRUED INTEREST:", "0.00") + f"         AS OF: {irs_abbr(asof)}",
        kv("ACCRUED PENALTY:", "0.00") + f"         AS OF: {irs_abbr(asof)}",
        "",
        "ACCOUNT BALANCE PLUS ACCRUALS",
        kv("(this is not a payoff amount):", "0.00"),
        "",
        ("c", "** INFORMATION FROM THE RETURN OR AS ADJUSTED **"),
        "",
        kv("EXEMPTIONS:", exemptions),
        f"{'FILING STATUS:':<46}{FS_PRINT[fs]}",
        kv("ADJUSTED GROSS INCOME:", money(agi)),
        kv("TAXABLE INCOME:", money(taxable)),
        kv("TAX PER RETURN:", money(tax)),
        kv("SE TAXABLE INCOME TAXPAYER:", "0.00"),
        kv("SE TAXABLE INCOME SPOUSE:", "0.00"),
        kv("TOTAL SELF EMPLOYMENT TAX:", "0.00"),
        "",
        f"{'RETURN DUE DATE OR RETURN RECEIVED DATE (WHICHEVER IS LATER)':<64}{irs_abbr(date(2026, 4, 15))}",
        f"{'PROCESSING DATE':<64}{irs_abbr(proc)}",
        "",
        ("c", "TRANSACTIONS"),
        "",
        tx("CODE", "EXPLANATION OF TRANSACTION", "CYCLE", "DATE", "AMOUNT"),
        tx("150", "Tax return filed", cycle, mdy(proc), money(tax, dollar=True)),
        f"{'':<5}n/a  {dln}",
        tx("806", "W-2 or 1099 withholding", "", mdy(date(2026, 4, 15)), money(-withholding, paren=False, dollar=True)),
        tx("766", "Credit to your account", "", mdy(date(2026, 4, 15)), money(-credit, paren=False, dollar=True)),
        tx("846", "Refund issued", "", mdy(refund_date), money(refund, dollar=True)),
        "",
        ("c", "This Product Contains Sensitive Taxpayer Data"),
    ]
    p = Page(pdf_path, "Account Transcript")
    end = p.mono_lines(36, 40, L, size=9.5, leading=12.2, center_w=540)
    assert end < 780, f"account transcript overflow: {end}"
    p.save()

    return assemble("IRS_ACCOUNT_TRANSCRIPT", {
        "tax_year": year,
        "taxpayer_ssn_last4": tp["ssn4"],
        "taxpayer_name": tp["name"],
        "filing_status": fs,
        "adjusted_gross_income": canon(agi),
        "taxable_income": canon(taxable),
        "tax_per_return": canon(tax),
        "account_balance": "0.00",
        "accrued_interest": "0.00",
        "accrued_penalty": "0.00",
        "tc150_date": proc.isoformat(),
        "tc806_withholding": canon(withholding),
        "tc846_refund_amount": canon(refund),
        "tc846_date": refund_date.isoformat(),
    })


# =========================================================================== IRS notices (CP2000 / CP14 / CP503)
def notice_header(p, notice_no, year, ndate, ssn4, extra, pages, irs_addr):
    p.text(36, 46, "Department of the Treasury", CAPB, 9.5)
    p.text(36, 57, "Internal Revenue Service", CAP, 9.5)
    for i, ln in enumerate(irs_addr):
        p.text(36, 70 + i * 9.5, ln, CAP, 8)
    rows = [("Notice", notice_no), ("Tax year", year), ("Notice date", long_date(ndate)),
            ("Social Security number", f"XXX-XX-{ssn4}")] + list(extra)
    top = 46
    for k, v in rows:
        p.text(350, top, k, CAP, 8)
        p.text(470, top, v, VAL, 9.5)
        top += 11
    p.text(350, top, f"Page 1 of {pages}", CAP, 8)
    top += 8
    p.line(36, top, 576, top, 0.8)
    return top


def notice_address(p, top, tp):
    p.text(36, top, tp["name"], VAL, VAL_SZ)
    p.text(36, top + 11, tp["street"], VAL, VAL_SZ)
    p.text(36, top + 22, tp["csz_notice"], VAL, VAL_SZ)
    return top + 22


def notice_footer(p, notice_no, year, ndate, ssn4, pages):
    p.line(36, 760, 576, 760, 0.5)
    p.text(36, 770, f"Notice {notice_no}", CAP, 7.5)
    p.text(140, 770, f"Tax year {year}", CAP, 7.5)
    p.text(230, 770, f"Notice date {long_date(ndate)}", CAP, 7.5)
    p.text(400, 770, f"Social Security number XXX-XX-{ssn4}", CAP, 7.5)
    p.text(576, 770, f"Page 1 of {pages}", CAP, 7.5, align="right")


def section(p, top, title):
    p.text(36, top, title, CAPB, 10.5)
    p.line(36, top + 3.5, 576, top + 3.5, 0.5)
    return top + 16


def kv_rows(p, top, rows, x=36, xv=330, bold_last=True, leading=12.5):
    for i, (k, v) in enumerate(rows):
        last = bold_last and i == len(rows) - 1
        if last:
            p.line(x, top - 9.5, xv, top - 9.5, 0.5)
        p.text(x, top, k, CAPB if last else CAP, 9)
        p.text(xv, top, v, VAL, VAL_SZ, align="right")
        top += leading
    return top


def bullets(p, top, items, x=36, w=536):
    for it in items:
        p.text(x, top, "•", CAP, 9)
        top = p.para(x + 10, top, w - 10, it)
        top += 2
    return top


def payment_stub(p, top, notice_no, ndate, tp, year, due_date, amount):
    p.text(306, top - 4, "Detach here and return this stub with your payment", CAP, 7, align="center")
    p.line(36, top, 576, top, 0.6, dashed=True)
    y = top + 18
    p.text(36, y, "INTERNAL REVENUE SERVICE", CAPB, 9)
    p.text(36, y + 11, "PO BOX 802501", CAP, 9)
    p.text(36, y + 22, "CINCINNATI OH 45280-2501", CAP, 9)
    rows = [("Notice", notice_no), ("Notice date", long_date(ndate)),
            ("Social Security number", f"XXX-XX-{tp['ssn4']}"),
            (f"Amount due by {long_date(due_date)}", money(amount, dollar=True)),
            ("Amount enclosed", "$ _______________")]
    ry = y
    for k, v in rows:
        p.text(330, ry, k, CAP, 8)
        p.text(576, ry, v, VAL, VAL_SZ, align="right")
        ry += 11.5
    p.text(36, y + 44, tp["name"], VAL, VAL_SZ)
    p.text(36, y + 55, tp["street"], VAL, VAL_SZ)
    p.text(36, y + 66, tp["csz_notice"], VAL, VAL_SZ)
    p.para(36, y + 84, 540, "Make your check or money order payable to the United States Treasury. Write your "
           f"Social Security number (XXX-XX-{tp['ssn4']}), the tax year ({year}), and the form number (1040) "
           "on your payment.", size=8, leading=10)


def gen_cp2000(n, pdf_path):
    rng = random.Random(n)
    year = "2024"
    tp = person(rng)
    payer = business(rng, NEC_PAYERS)
    ndate = date(2026, rng.randint(1, 9), rng.randint(1, 28))
    respond_by = ndate + timedelta(days=30)
    unreported = rng.randint(8000, 40000) * 100
    tax = round(unreported * rng.uniform(0.24, 0.34) / 100) * 100 + rng.randint(0, 99)
    penalty = round(tax * 0.20)
    interest = round(tax * rng.uniform(0.05, 0.09))
    total = tax + penalty + interest
    aur = f"{rng.randint(10000, 99999)}-{rng.randint(1000, 9999)}"

    p = Page(pdf_path, "Notice CP2000")
    top = notice_header(p, "CP2000", year, ndate, tp["ssn4"],
                        [("AUR control number", aur), ("To contact us", "Phone 800-829-8310"),
                         ("", "Fax 877-477-0583")], 7,
                        ["AUR CORR 5-E08-113", "PHILADELPHIA PA 19255-0521"])
    top = notice_address(p, top + 18, tp)
    top += 24
    p.text(36, top, f"Proposed changes to your {year} Form 1040", CAPB, 13)
    top += 14
    top = p.para(36, top, 540, f"We are proposing changes to your {year} Form 1040 tax return. This is not a bill. "
                 "Our records show that the income reported to us by payers does not match the income reported "
                 "on your return.")
    top += 6
    p.rect(36, top, 540, 30)
    p.text(46, top + 19, "Proposed amount due", CAPB, 10)
    p.text(205, top + 19, money(total, dollar=True), VAL, 11)
    p.text(330, top + 19, "Respond by", CAPB, 10)
    p.text(410, top + 19, long_date(respond_by), VAL, 11)
    top += 44
    top = section(p, top, "Summary of proposed changes")
    top = kv_rows(p, top, [("Tax you owe", money(tax, dollar=True)),
                           ("Penalty (accuracy-related)", money(penalty, dollar=True)),
                           ("Interest", money(interest, dollar=True)),
                           ("Proposed amount due", money(total, dollar=True))])
    top += 6
    top = section(p, top, f"Changes to your {year} tax return")
    p.text(36, top, "Payer and form", CAPB, 8)
    p.text(360, top, "Reported to IRS by others", CAPB, 8, align="right")
    p.text(460, top, "Shown on return", CAPB, 8, align="right")
    p.text(576, top, "Difference", CAPB, 8, align="right")
    p.line(36, top + 3.5, 576, top + 3.5)
    top += 15
    p.text(36, top, payer["name"], VAL, VAL_SZ)
    p.text(360, top, money(unreported, dollar=True), VAL, VAL_SZ, align="right")
    p.text(460, top, "$0.00", VAL, VAL_SZ, align="right")
    p.text(576, top, money(unreported, dollar=True), VAL, VAL_SZ, align="right")
    top += 10
    p.text(36, top, "Form 1099-NEC, Nonemployee compensation", CAP, 8)
    top += 10
    p.text(36, top, "Total income change", CAPB, 8)
    p.text(576, top, money(unreported, dollar=True), VAL, VAL_SZ, align="right")
    top += 18
    top = section(p, top, "What you need to do immediately")
    top = bullets(p, top, [
        f"Review this notice and compare our changes to the information on your {year} tax return.",
        f"Complete the enclosed Response form and return it to us by {long_date(respond_by)}, "
        "even if you disagree, so that we receive it by that date.",
        f"If you agree with the changes, sign the Response form and return it with your payment of "
        f"{money(total, dollar=True)}. Make your check or money order payable to the United States Treasury.",
        "If you don't agree, send us a signed statement explaining why, along with documents that support "
        "your position. Do not file an amended return for this issue.",
    ])
    top += 4
    top = section(p, top, "If we don't hear from you")
    top = p.para(36, top, 540, f"If we don't receive your response by {long_date(respond_by)}, we will send you a "
                 "Statutory Notice of Deficiency followed by a bill for the proposed amount due, plus additional "
                 "interest. Interest continues to accrue until the balance is paid in full.")
    top += 4
    top = section(p, top, "Additional information")
    p.para(36, top, 540, "Visit www.irs.gov/cp2000 for more information about this notice. Keep this notice "
           "for your records. For tax forms, instructions, and publications, visit www.irs.gov or call "
           "800-TAX-FORM (800-829-3676).")
    notice_footer(p, "CP2000", year, ndate, tp["ssn4"], 7)
    p.save()

    return assemble("CP2000", {
        "notice_type": "CP2000",
        "tax_year": year,
        "notice_date": ndate.isoformat(),
        "taxpayer_ssn_last4": tp["ssn4"],
        "taxpayer_name": tp["name"],
        "response_deadline": respond_by.isoformat(),
        "proposed_tax_increase": canon(tax),
        "proposed_penalty": canon(penalty),
        "proposed_interest": canon(interest),
        "total_proposed_amount_due": canon(total),
        "unreported_payer_name": payer["name"],
        "unreported_amount": canon(unreported),
    })


def gen_cp14(n, pdf_path):
    rng = random.Random(n)
    year = "2025"
    tp = person(rng)
    ndate = date(2026, rng.randint(5, 8), rng.randint(1, 28))
    pay_by = ndate + timedelta(days=21)
    owed = rng.randint(1500, 9000) * 100 + rng.randint(0, 99)
    penalty = round(owed * 0.005 * rng.randint(1, 3))
    interest = round(owed * 0.07 * rng.randint(30, 90) / 365)
    total = owed + penalty + interest
    caller = f"{rng.randint(100000, 999999)}"

    p = Page(pdf_path, "Notice CP14")
    top = notice_header(p, "CP14", year, ndate, tp["ssn4"],
                        [("To contact us", "Phone 800-829-0922"), ("Your Caller ID", caller)], 4,
                        ["PO BOX 9019", "HOLTSVILLE NY 11742-9019"])
    top = notice_address(p, top + 18, tp)
    top += 24
    p.text(36, top, f"You have unpaid taxes for {year}", CAPB, 13)
    p.text(576, top, f"Amount due: {money(total, dollar=True)}", CAPB, 13, align="right")
    top += 14
    top = p.para(36, top, 540, f"Our records show you have unpaid taxes on your {year} Form 1040. "
                 f"Pay by {long_date(pay_by)} to avoid additional penalty and interest charges.")
    top += 6
    p.rect(36, top, 540, 30)
    p.text(46, top + 19, f"Amount due by {long_date(pay_by)}", CAPB, 10)
    p.text(566, top + 19, money(total, dollar=True), VAL, 11, align="right")
    top += 44
    top = section(p, top, "Billing summary")
    top = kv_rows(p, top, [("Amount you owed", money(owed, dollar=True)),
                           ("Failure-to-pay penalty", money(penalty, dollar=True)),
                           ("Interest charges", money(interest, dollar=True)),
                           (f"Amount due by {long_date(pay_by)}", money(total, dollar=True))])
    top += 6
    top = section(p, top, "What you need to do immediately")
    top = bullets(p, top, [
        f"Pay {money(total, dollar=True)} by {long_date(pay_by)}. Pay online at www.irs.gov/payments, or "
        "send your payment with the payment stub below.",
        "If you can't pay the full amount, pay as much as you can now and visit www.irs.gov/paymentplan "
        "to apply for a payment plan.",
        "If you already paid this amount or think we made a mistake, call us at the number listed above.",
    ])
    top += 4
    top = section(p, top, "Penalties and interest")
    top = p.para(36, top, 540, "Failure-to-pay penalty: We charge a penalty of 0.5% of the unpaid tax for each "
                 "month or part of a month the tax remains unpaid, up to 25%. Interest: We charge interest on "
                 "unpaid tax from the due date of the return until the date the tax is paid in full. The "
                 "current interest rate is 7% per year, compounded daily.")
    payment_stub(p, 636, "CP14", ndate, tp, year, pay_by, total)
    notice_footer(p, "CP14", year, ndate, tp["ssn4"], 4)
    p.save()

    return assemble("CP14", {
        "notice_type": "CP14",
        "tax_year": year,
        "notice_date": ndate.isoformat(),
        "taxpayer_ssn_last4": tp["ssn4"],
        "taxpayer_name": tp["name"],
        "pay_by_date": pay_by.isoformat(),
        "tax_owed": canon(owed),
        "failure_to_pay_penalty": canon(penalty),
        "interest": canon(interest),
        "amount_due": canon(total),
    })


def gen_cp503(n, pdf_path):
    rng = random.Random(n)
    year = "2024"
    tp = person(rng)
    ndate = date(2026, rng.randint(2, 9), rng.randint(1, 28))
    pay_by = ndate + timedelta(days=10)
    owed = rng.randint(1200, 12000) * 100 + rng.randint(0, 99)
    penalty = round(owed * 0.005 * rng.randint(3, 8))
    interest = round(owed * 0.07 * rng.randint(100, 240) / 365)
    total = owed + penalty + interest
    caller = f"{rng.randint(100000, 999999)}"

    p = Page(pdf_path, "Notice CP503")
    top = notice_header(p, "CP503", year, ndate, tp["ssn4"],
                        [("To contact us", "Phone 800-829-3903"), ("Your Caller ID", caller)], 3,
                        ["PO BOX 219690", "KANSAS CITY MO 64121-9690"])
    top = notice_address(p, top + 18, tp)
    top += 24
    p.text(36, top, f"Second reminder: You have unpaid taxes for {year}", CAPB, 13)
    top += 14
    p.text(36, top + 2, f"Amount due: {money(total, dollar=True)}", CAPB, 12)
    top += 16
    top = p.para(36, top, 540, f"Our records show you still have unpaid taxes on your {year} Form 1040 and we "
                 f"haven't received a response to our previous notice. If you don't pay {money(total, dollar=True)} "
                 f"by {long_date(pay_by)}, interest will increase and additional penalties may apply.")
    top += 6
    p.rect(36, top, 540, 30)
    p.text(46, top + 19, f"Pay by {long_date(pay_by)}", CAPB, 10)
    p.text(566, top + 19, money(total, dollar=True), VAL, 11, align="right")
    top += 44
    top = section(p, top, "Billing summary")
    top = kv_rows(p, top, [("Amount you owed", money(owed, dollar=True)),
                           ("Failure-to-pay penalty", money(penalty, dollar=True)),
                           ("Interest charges", money(interest, dollar=True)),
                           (f"Amount due by {long_date(pay_by)}", money(total, dollar=True))])
    top += 6
    top = section(p, top, "What you need to do immediately")
    top = bullets(p, top, [
        f"Pay {money(total, dollar=True)} by {long_date(pay_by)}. Pay online at www.irs.gov/payments, or "
        "send your payment with the payment stub below.",
        "If you can't pay the full amount, visit www.irs.gov/paymentplan or call us to apply for a payment plan.",
    ])
    top += 4
    top = section(p, top, "If we don't hear from you")
    top = p.para(36, top, 540, "If you don't pay the amount due or contact us by the date above, we may file a "
                 "Notice of Federal Tax Lien, which is a public notice to your creditors that the government has a "
                 "right to your current and future property. Penalties and interest will continue to accrue.")
    payment_stub(p, 636, "CP503", ndate, tp, year, pay_by, total)
    notice_footer(p, "CP503", year, ndate, tp["ssn4"], 3)
    p.save()

    return assemble("CP503", {
        "notice_type": "CP503",
        "tax_year": year,
        "notice_date": ndate.isoformat(),
        "taxpayer_ssn_last4": tp["ssn4"],
        "taxpayer_name": tp["name"],
        "pay_by_date": pay_by.isoformat(),
        "amount_due": canon(total),
        "caller_id": caller,
    })


# =========================================================================== driver
DOCS = [
    (13, "K1_1065", lambda n, path: gen_k1_1065(n, path, "GENERAL", loss=False)),
    (14, "K1_1065", lambda n, path: gen_k1_1065(n, path, "LIMITED", loss=True)),
    (15, "K1_1120S", gen_k1_1120s),
    (16, "IRS_WAGE_INCOME_TRANSCRIPT", gen_wage_income_transcript),
    (17, "IRS_ACCOUNT_TRANSCRIPT", gen_account_transcript),
    (18, "CP2000", gen_cp2000),
    (19, "CP14", gen_cp14),
    (20, "CP503", gen_cp503),
]


def main(scan=True):
    os.makedirs(OUT_DIR, exist_ok=True)
    gt = {}
    for n, dtype, fn in DOCS:
        doc_id = f"doc{n:02d}"
        pdf_name = f"{doc_id}_{dtype}.pdf"
        png_name = f"{doc_id}_{dtype}.png"
        pdf_path = os.path.join(OUT_DIR, pdf_name)
        png_path = os.path.join(OUT_DIR, png_name)
        fields = fn(n, pdf_path)
        profile = PROFILES[(n - 1) % 4]
        if scan:
            pdf_to_scan(pdf_path, png_path, profile, seed=n)
        gt[doc_id] = {"doc_id": doc_id, "doc_type": dtype, "png": png_name, "pdf": pdf_name,
                      "scan_profile": profile, "fields": fields}
        print(f"{doc_id} {dtype:<28} {profile:<10} {len(fields)} fields")
    with open(GT_PATH, "w") as fh:
        json.dump(gt, fh, indent=2)
    print(f"wrote {GT_PATH}")
    return gt


if __name__ == "__main__":
    main(scan="--no-scan" not in sys.argv)
