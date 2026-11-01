#!/usr/bin/env python3
"""Merge data/docs/ground_truth_forms.json + ground_truth_k1_irs.json into data/docs/ground_truth.json and validate
every entry against eval/fields.json (exact key set, canonical formats). Exits non-zero on any problem."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = json.load(open(ROOT / "eval/fields.json"))
D = ROOT / "data/docs"
PROFILES = ["clean", "skew_noise", "fax", "photo"]
RX = {"money": r"^-?\d+\.\d{2}$", "date": r"^\d{4}-\d{2}-\d{2}$", "ein": r"^\d{2}-\d{7}$", "last4": r"^\d{4}$", "year": r"^20\d{2}$", "pct": r"^\d+\.\d{2}$"}
DATE_F = {"notice_date", "pay_by_date", "response_deadline", "tc150_date", "tc846_date"}
CODE_F = {"notice_type", "filing_status", "partner_type", "box15_state", "box7_distribution_code", "box12a_code", "caller_id"}


def kind(f):
    if f in DATE_F: return "date"
    if f in ("tax_year",): return "year"
    if f in CODE_F: return "code"
    if f == "form_count": return "int"
    if f.endswith("_checkbox"): return "bool"
    if f.endswith("_pct") or f.endswith("_pct_ending"): return "pct"
    if f.endswith("_last4"): return "last4"
    if f.endswith("_ein") or f.endswith("_tin"): return "ein"
    if f.endswith("_name"): return "name"
    return "money"


merged, problems = {}, []
for part in ["ground_truth_forms.json", "ground_truth_k1_irs.json"]:
    p = D / part
    if not p.exists():
        problems.append(f"missing {part}")
        continue
    merged.update(json.load(open(p)))
for did, e in sorted(merged.items()):
    n = int(did[3:])
    t = e.get("doc_type")
    exp = FIELDS["doc_types"].get(t)
    if not exp:
        problems.append(f"{did}: unknown doc_type {t}"); continue
    got = list(e.get("fields", {}).keys())
    if set(got) != set(exp):
        problems.append(f"{did}: field keys differ: missing {sorted(set(exp)-set(got))} extra {sorted(set(got)-set(exp))}")
    if e.get("scan_profile") != PROFILES[(n - 1) % 4]:
        problems.append(f"{did}: scan_profile {e.get('scan_profile')} != {PROFILES[(n-1)%4]}")
    for f in ("png", "pdf"):
        if not (D / e.get(f, "")).exists():
            problems.append(f"{did}: {f} file missing: {e.get(f)}")
    for f, v in e.get("fields", {}).items():
        k = kind(f)
        if k == "bool":
            if not isinstance(v, bool): problems.append(f"{did}.{f}: not a JSON boolean: {v!r}")
            continue
        if not isinstance(v, str):
            problems.append(f"{did}.{f}: not a string: {v!r}"); continue
        if k in RX and not re.match(RX[k], v): problems.append(f"{did}.{f}: bad {k} format: {v!r}")
        if k in ("name", "code") and v != v.upper(): problems.append(f"{did}.{f}: not uppercase: {v!r}")
        if k == "int" and not v.isdigit(): problems.append(f"{did}.{f}: not an int: {v!r}")
if len(merged) != 20:
    problems.append(f"expected 20 docs, got {len(merged)}")
if problems:
    print("\n".join(problems)); sys.exit(1)
(D / "ground_truth.json").write_text(json.dumps(merged, indent=1))
from collections import Counter
print(f"OK: wrote {D/'ground_truth.json'} with {len(merged)} docs; types: {dict(Counter(e['doc_type'] for e in merged.values()))}; fields total: {sum(len(e['fields']) for e in merged.values())}")
