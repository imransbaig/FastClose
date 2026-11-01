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
    ("GREGORY LANDIS DBA LANDIS PROPERTY MANAGEMENT", "3318 OAKMONT DR", "COLUMBUS", "OH", "43221"),
]
INT_PAYERS = [
    ("FIRST MERIDIAN BANK NA", "PO BOX 41900", "CHARLOTTE", "NC", "28201"),
    ("LAKESHORE COMMUNITY FEDERAL CREDIT UNION", "6150 LAKESHORE AVE", "CLEVELAND", "OH", "44102"),
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
    c.setFont(vfont, vsize)
    ty = y + h - CAP_SIZE - 4 - vsize - 2
    for ln in lines:
        c.drawString(x + 5, ty, ln)
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
