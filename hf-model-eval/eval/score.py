#!/usr/bin/env python3
"""Score model predictions against ground truth for the three areas.

    python eval/score.py --area docs    --pred results/pred_docs_<name>.json    --name <name>
    python eval/score.py --area notices --pred results/pred_notices_<name>.json --name <name>
    python eval/score.py --area calls   --pred results/pred_calls_<name>.json   --name <name>
    python eval/score.py --compile            # merge every results/score_*.json into results/eval_sheet.csv/.xlsx

Prediction file formats (JSON object keyed by item id):
  docs    {"doc01": {"fields": {...}, "raw_output": "...", "latency_sec": 1.2}}
  notices {"notice01": {"fields": {...} or null, "json_valid": true, "raw_output": "..."}}
  calls   {"call01": {"transcript": "...", "summary": {...} or null, "asr_model": "...", "summary_model": "..."}}
Exact match is computed after normalizing both sides with the rules below (same rules for every model).
"""
import argparse
import datetime as dt
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import os
HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("EVAL_ROOT", HERE.parent))  # override for test fixtures
FIELDS = json.load(open(HERE / "fields.json"))

DATE_FIELDS = {"notice_date", "pay_by_date", "response_deadline", "key_date", "tc150_date", "tc846_date"}
YEAR_FIELDS = {"tax_year", "tax_year_discussed"}
CODE_FIELDS = {"notice_type", "entity_type", "topic_code", "follow_up_action_code", "action_required", "urgency",
               "filing_status", "partner_type", "box15_state", "box7_distribution_code", "box12a_code", "caller_id"}
INT_FIELDS = {"form_count"}
FILING = {"MFJ": "MARRIED FILING JOINT", "MARRIED FILING JOINTLY": "MARRIED FILING JOINT", "JOINT": "MARRIED FILING JOINT",
          "MARRIED JOINT": "MARRIED FILING JOINT", "HOH": "HEAD OF HOUSEHOLD", "S": "SINGLE", "MFS": "MARRIED FILING SEPARATE",
          "MARRIED FILING SEPARATELY": "MARRIED FILING SEPARATE"}
PARTNER = {"LIMITED PARTNER": "LIMITED", "LP": "LIMITED", "GENERAL PARTNER": "GENERAL", "GP": "GENERAL",
           "LIMITED PARTNER OR OTHER LLC MEMBER": "LIMITED", "LLC MEMBER": "LIMITED"}
NONE_WORDS = {"", "none", "null", "n/a", "na", "not stated", "not applicable", "-", "--", "nan"}


def _s(v):
    return "" if v is None else str(v).strip()


def norm_money(v):
    s = _s(v).lower()
    if s in NONE_WORDS or s == "blank":
        return "0.00"
    neg = s.startswith("(") and s.endswith(")") or s.startswith("-") or s.startswith("$-") or s.startswith("-$")
    s = re.sub(r"[^\d.]", "", s)
    if s in ("", "."):
        return _s(v).upper()
    try:
        x = float(s)
    except ValueError:
        return _s(v).upper()
    return f"{-x if neg else x:.2f}".replace("-0.00", "0.00")


def norm_date(v):
    s = _s(v)
    if s.lower() in NONE_WORDS:
        return "NONE"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s
    if not re.search(r"(19|20)\d{2}", s) and not re.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", s):
        return s.upper()  # no year present: cannot be a full date, keep raw so it scores wrong
    try:
        from dateutil import parser
        return parser.parse(s, fuzzy=True).strftime("%Y-%m-%d")
    except Exception:
        return s.upper()


def norm_name(v):
    s = _s(v).upper()
    s = s.replace("&AMP;", "&").replace("’", "'")
    s = re.sub(r"[.,'\"]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def norm_ein(v):
    d = re.sub(r"\D", "", _s(v))
    return f"{d[:2]}-{d[2:]}" if len(d) == 9 else _s(v).upper()


def norm_last4(v):
    d = re.sub(r"\D", "", _s(v))
    return d[-4:] if len(d) >= 4 else _s(v)


def norm_code(field, v):
    s = re.sub(r"\s+", " ", _s(v).upper().replace("_", "_")).strip()
    s = re.sub(r"[.]", "", s)
    if field == "filing_status":
        return FILING.get(s, s)
    if field == "partner_type":
        return PARTNER.get(s, s)
    if field == "notice_type":
        s = s.replace(" ", "").replace("NOTICE", "").replace("LETTER", "")
    if field in ("entity_type", "topic_code", "follow_up_action_code", "action_required", "urgency"):
        s = s.replace(" ", "_").replace("-", "_")
    if field == "box7_distribution_code":
        s = s.replace(" ", "")
    return s


def norm_bool(v):
    s = _s(v).lower()
    if isinstance(v, bool):
        return "true" if v else "false"
    return "true" if s in {"true", "yes", "x", "checked", "1", "y", "✓", "☒"} else "false"


def norm_pct(v):
    s = re.sub(r"[^\d.]", "", _s(v))
    try:
        return f"{float(s):.2f}"
    except ValueError:
        return _s(v).upper()


def normalize(field, v):
    if field in DATE_FIELDS:
        return norm_date(v)
    if field in YEAR_FIELDS:
        d = re.findall(r"20\d{2}", _s(v))
        return d[0] if d else _s(v)
    if field in CODE_FIELDS:
        return norm_code(field, v)
    if field in INT_FIELDS:
        d = re.sub(r"\D", "", _s(v))
        return str(int(d)) if d else _s(v)
    if field.endswith("_checkbox"):
        return norm_bool(v)
    if field.endswith("_pct") or field.endswith("_pct_ending"):
        return norm_pct(v)
    if field.endswith("_last4"):
        return norm_last4(v)
    if field.endswith("_ein") or field.endswith("_tin"):
        return norm_ein(v)
    if field.endswith("_name"):
        return norm_name(v)
    return norm_money(v)


def load_gt(area):
    if area == "docs":
        p = ROOT / "data/docs/ground_truth.json"
        gt = json.load(open(p))
        return {k: {"type": v["doc_type"], "fields": v["fields"]} for k, v in gt.items()}
    if area == "notices":
        gt = json.load(open(ROOT / "data/notices/ground_truth.json"))
        return {k: {"type": "NOTICE", "fields": {f: v[f] for f in FIELDS["notice_fields"]}} for k, v in gt.items()}
    if area == "calls":
        gt = json.load(open(ROOT / "data/calls/ground_truth.json"))
        out = {}
        for k, v in gt.items():
            ref = (ROOT / "data/calls" / f"{k}.ref.txt").read_text()
            out[k] = {"type": "CALL", "fields": {f: v[f] for f in FIELDS["call_summary_fields"]}, "ref": ref}
        return out
    raise ValueError(area)


def score_fields(gt_fields, pred_fields, ftype=None):
    rows = []
    pred_fields = pred_fields or {}
    pred_l = {k.lower(): v for k, v in pred_fields.items()} if isinstance(pred_fields, dict) else {}
    for f, g in gt_fields.items():
        p = pred_l.get(f.lower())
        gn, pn = normalize(f, g), normalize(f, p) if p is not None else None
        rows.append({"field": f, "gt": gn, "pred": pn, "pred_raw": p, "correct": pn == gn})
    return rows


def wer(ref, hyp):
    import jiwer
    from whisper_normalizer.english import EnglishTextNormalizer
    n = EnglishTextNormalizer()
    r, h = n(ref), n(hyp or "")
    return jiwer.wer(r, h), r, h


def score(area, pred, name):
    gt = load_gt(area)
    items, per_type, per_field = {}, defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    tot_c = tot_n = 0
    wers = []
    json_invalid = 0
    for k, g in gt.items():
        p = pred.get(k, {}) or {}
        pf = p.get("fields") if area != "calls" else p.get("summary")
        if area == "notices" and (pf is None or p.get("json_valid") is False):
            json_invalid += 1
        rows = score_fields(g["fields"], pf)
        c = sum(r["correct"] for r in rows)
        items[k] = {"type": g["type"], "n": len(rows), "correct": c, "acc": round(c / len(rows), 3), "rows": rows}
        tot_c += c
        tot_n += len(rows)
        per_type[g["type"]][0] += c
        per_type[g["type"]][1] += len(rows)
        for r in rows:
            per_field[r["field"]][0] += r["correct"]
            per_field[r["field"]][1] += 1
        if area == "calls":
            w, rn, hn = wer(g["ref"], p.get("transcript", ""))
            items[k]["wer"] = round(w, 4)
            items[k]["ref_words"] = len(rn.split())
            wers.append(w)
    out = {"name": name, "area": area, "scored_at": dt.datetime.now(dt.timezone.utc).isoformat(),
           "items": len(gt), "items_with_predictions": sum(1 for k in gt if k in pred),
           "fields_total": tot_n, "fields_correct": tot_c, "field_exact_match": round(tot_c / tot_n, 4) if tot_n else None,
           "per_type": {t: {"correct": c, "n": n, "acc": round(c / n, 3)} for t, (c, n) in per_type.items()},
           "per_field": {f: {"correct": c, "n": n, "acc": round(c / n, 3)} for f, (c, n) in per_field.items()},
           "per_item": items}
    if area == "notices":
        out["json_invalid_items"] = json_invalid
    if area == "calls":
        out["wer_mean"] = round(sum(wers) / len(wers), 4) if wers else None
        out["wer_per_call"] = {k: v["wer"] for k, v in items.items()}
    return out


def compile_sheet():
    rows = []
    for p in sorted((ROOT / "results").glob("score_*.json")):
        s = json.load(open(p))
        rows.append({"area": s["area"], "model": s["name"], "items": s["items"], "items_predicted": s["items_with_predictions"],
                     "fields_total": s["fields_total"], "fields_correct": s["fields_correct"],
                     "field_exact_match": s["field_exact_match"], "wer_mean": s.get("wer_mean"),
                     "json_invalid_items": s.get("json_invalid_items"), "scored_at": s["scored_at"],
                     "per_type": json.dumps({t: v["acc"] for t, v in s["per_type"].items()})})
    import pandas as pd
    df = pd.DataFrame(rows).sort_values(["area", "field_exact_match"], ascending=[True, False])
    df.to_csv(ROOT / "results/eval_sheet.csv", index=False)
    try:
        df.to_excel(ROOT / "results/eval_sheet.xlsx", index=False)
    except Exception as e:
        print("xlsx skipped:", e)
    print(df.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area", choices=["docs", "notices", "calls"])
    ap.add_argument("--pred")
    ap.add_argument("--name")
    ap.add_argument("--compile", action="store_true")
    a = ap.parse_args()
    if a.compile:
        compile_sheet()
        return
    pred = json.load(open(a.pred))
    s = score(a.area, pred, a.name)
    outp = ROOT / "results" / f"score_{a.area}_{a.name}.json"
    outp.write_text(json.dumps(s, indent=1))
    print(f"{a.area} / {a.name}: field exact match {s['field_exact_match']} ({s['fields_correct']}/{s['fields_total']})"
          + (f", mean WER {s['wer_mean']}" if a.area == "calls" else "")
          + (f", invalid JSON items {s['json_invalid_items']}" if a.area == "notices" else ""))
    for t, v in s["per_type"].items():
        print(f"  {t:28s} {v['acc']:.3f} ({v['correct']}/{v['n']})")
    print("wrote", outp)


if __name__ == "__main__":
    main()
