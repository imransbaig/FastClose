#!/usr/bin/env python3
"""Pull candidate open models from the Hugging Face Hub API for three tax-firm task areas.

    python hf_pull.py --out pulls/2026-11-15 [--limit 40] [--areas A B C] [--include-derivatives]

Area A  doc field extraction      pipeline_tag image-text-to-text, image-to-text
Area B  speech-to-text            pipeline_tag automatic-speech-recognition
Area C  small JSON instruct       pipeline_tag text-generation, num_parameters max:9B

For every area the script pulls two lists (sort=trending_score and sort=downloads), unions them,
then fetches per-model details: downloads (30 day), all-time downloads, likes, created date,
parameter count, license tag, gated flag, LICENSE file text classification, finetune / adapter /
quantized / merge counts, Spaces using the model, inference-provider support.
Downloads are age-normalized two ways: downloads_30d_per_day and all_time_downloads_per_day.

Needs network access to huggingface.co (API) and, for LICENSE file reads, the Hub file CDN
(cas-bridge.xethub.hf.co, cdn-lfs.hf.co, cdn-lfs-us-1.hf.co). Set HF_TOKEN to read gated repos
whose terms you have accepted. Output: <out>/area_<X>_candidates.json, area_<X>_ranked.csv,
summary.xlsx, pull_meta.json. Re-run in two weeks and diff the CSVs.
"""
import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import (EntryNotFoundError, GatedRepoError, HfHubHTTPError,
                                    RepositoryNotFoundError)

AREAS = {
    "A": {"name": "doc_field_extraction", "pipeline_tags": ["image-text-to-text", "image-to-text"], "num_parameters": None},
    "B": {"name": "speech_to_text", "pipeline_tags": ["automatic-speech-recognition"], "num_parameters": None},
    "C": {"name": "small_json_instruct", "pipeline_tags": ["text-generation"], "num_parameters": "max:9B"},
}
SORTS = ["trending_score", "downloads"]
DETAIL_EXPAND = ["author", "cardData", "createdAt", "downloads", "downloadsAllTime", "gated", "likes",
                 "safetensors", "trendingScore", "pipeline_tag", "tags", "inferenceProviderMapping",
                 "spaces", "library_name", "lastModified", "childrenModelCount", "gguf", "siblings"]
WEIGHT_EXT = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".onnx", ".ckpt", ".msgpack", ".h5",
              ".nemo", ".tflite", ".mlmodel", ".npz", ".mlpackage", ".params")

# --- license rubric -------------------------------------------------------------------------
PERMISSIVE = {"apache-2.0", "mit", "bsd", "bsd-2-clause", "bsd-3-clause", "bsd-3-clause-clear", "cc0-1.0",
              "cc-by-4.0", "cc-by-3.0", "cc-by-sa-4.0", "cc-by-sa-3.0", "mpl-2.0", "gpl-3.0", "gpl-2.0",
              "lgpl-3.0", "lgpl-2.1", "agpl-3.0", "unlicense", "isc", "wtfpl", "artistic-2.0", "pddl",
              "odc-by", "odbl", "eupl-1.1", "etalab-2.0", "zlib", "postgresql", "ms-pl", "ncsa", "ofl-1.1",
              "epl-2.0", "epl-1.0", "cdla-permissive-2.0", "cdla-sharing-1.0", "osl-3.0", "ecl-2.0",
              "afl-3.0", "bsl-1.0", "lppl-1.3c", "cc-by-2.5", "cc-by-2.0"}
CONDITIONAL = {  # commercial use allowed, with conditions you must read
    "llama2": "Meta Llama 2 Community License (AUP; >700M MAU needs a separate license)",
    "llama3": "Meta Llama 3 Community License (AUP, attribution 'Built with Llama'; >700M MAU)",
    "llama3.1": "Meta Llama 3.1 Community License (same conditions as Llama 3)",
    "llama3.2": "Meta Llama 3.2 Community License (same; EU multimodal restriction)",
    "llama3.3": "Meta Llama 3.3 Community License (same conditions)",
    "llama4": "Meta Llama 4 Community License (same; EU restriction)",
    "gemma": "Gemma Terms of Use (prohibited-use policy; commercial use allowed)",
    "deepseek": "DeepSeek Model License (use-based restrictions; commercial use allowed)",
    "qwen": "Tongyi Qianwen License (commercial allowed; >100M MAU needs a license)",
    "tongyi-qianwen": "Tongyi Qianwen License (commercial allowed; >100M MAU needs a license)",
    "nvidia-open-model-license": "NVIDIA Open Model License (commercial allowed; guardrail/attribution terms)",
    "openrail": "OpenRAIL (commercial allowed; use-based restrictions in Attachment A)",
    "openrail++": "OpenRAIL++ (commercial allowed; use-based restrictions)",
    "creativeml-openrail-m": "CreativeML OpenRAIL-M (commercial allowed; use-based restrictions)",
    "bigscience-openrail-m": "BigScience OpenRAIL-M (commercial allowed; use-based restrictions)",
    "bigcode-openrail-m": "BigCode OpenRAIL-M (commercial allowed; use-based restrictions)",
    "bigscience-bloom-rail-1.0": "BLOOM RAIL (commercial allowed; use-based restrictions)",
    "stabilityai-ai-community": "Stability AI Community License (free commercial use below $1M annual revenue)",
    "cc-by-nd-4.0": "CC BY-ND (commercial allowed; no derivatives, so no fine-tuning)",
    "apple-amlr": "Apple ML Research license (READ: often research-only)",
    "intel-research": "Intel research license (READ)",
}
NOT_ALLOWED = {"cc-by-nc-4.0", "cc-by-nc-sa-4.0", "cc-by-nc-nd-4.0", "cc-by-nc-3.0", "cc-by-nc-sa-3.0",
               "cc-by-nc-2.0", "cc-by-nc-nd-3.0", "qwen-research", "tongyi-qianwen-research",
               "microsoft-research-license", "mistral-research", "mrl", "c-uda", "deepfloyd-if-license",
               "llama-research", "bsd-3-clause-non-commercial"}
NC_PATTERNS = [r"non-?\s?commercial", r"not\s+(be\s+)?used?\s+for\s+(any\s+)?commercial", r"research\s+(purposes?|use)\s+only",
               r"solely\s+for\s+(non-?commercial|research|academic)", r"evaluation\s+purposes?\s+only",
               r"personal,?\s+non-?commercial", r"academic\s+(and|or)\s+research\s+purposes?\s+only",
               r"may\s+not\s+be\s+used\s+commercially", r"\bnoncommercial\b"]
COND_PATTERNS = [r"monthly\s+active\s+users", r"acceptable\s+use\s+policy", r"use-?\s?based\s+restrictions",
                 r"attachment\s+a", r"annual\s+(gross\s+)?revenue", r"\$1,?000,?000", r"prohibited\s+use",
                 r"must\s+(prominently\s+)?display", r"built\s+with"]
# patterns that explicitly carve out commercial use even if NC-like words appear elsewhere
COMMERCIAL_OK_PATTERNS = [r"commercial\s+(use|purposes?)\s+(is|are)\s+(permitted|allowed)", r"for\s+any\s+purpose",
                          r"including\s+commercial", r"commercial\s+or\s+non-?commercial"]


def classify_license(tag, text):
    """Return (status, reason). status in ALLOWED / ALLOWED_WITH_CONDITIONS / NOT_ALLOWED / UNKNOWN_REVIEW."""
    tag = (tag or "").lower().strip()
    reasons = []
    if tag in NOT_ALLOWED:
        return "NOT_ALLOWED", f"license tag '{tag}' is non-commercial"
    text_l = (text or "").lower()
    nc_hits = [p for p in NC_PATTERNS if re.search(p, text_l)]
    ok_hits = [p for p in COMMERCIAL_OK_PATTERNS if re.search(p, text_l)]
    cond_hits = [p for p in COND_PATTERNS if re.search(p, text_l)]
    if nc_hits and not ok_hits:
        return "NOT_ALLOWED", f"LICENSE text matches non-commercial clause(s): {nc_hits[:2]}"
    if tag in PERMISSIVE:
        if nc_hits:
            return "ALLOWED_WITH_CONDITIONS", f"permissive tag '{tag}' but LICENSE text mentions {nc_hits[:1]} alongside a commercial carve-out; read it"
        return "ALLOWED", f"permissive license '{tag}'"
    if tag in CONDITIONAL:
        return "ALLOWED_WITH_CONDITIONS", CONDITIONAL[tag]
    if tag.startswith("llama"):
        return "ALLOWED_WITH_CONDITIONS", "Meta Llama community license (read AUP / MAU cap)"
    if tag.startswith("cc-by-nc"):
        return "NOT_ALLOWED", "Creative Commons non-commercial"
    if tag.startswith("cc-by"):
        return "ALLOWED", f"Creative Commons attribution license '{tag}'"
    if text:
        if cond_hits:
            return "ALLOWED_WITH_CONDITIONS", f"custom license; conditions found: {cond_hits[:3]}"
        if ok_hits:
            return "ALLOWED", "custom license text explicitly permits commercial use"
        return "UNKNOWN_REVIEW", "custom license text has no obvious NC clause; human review required"
    return "UNKNOWN_REVIEW", f"no LICENSE file readable and tag '{tag or '(none)'}' not in rubric"


# --- helpers ----------------------------------------------------------------------------------
def params_from(info):
    try:
        if info.safetensors and info.safetensors.total:
            return int(info.safetensors.total), "safetensors"
    except Exception:
        pass
    try:
        if info.gguf and info.gguf.get("total"):
            return int(info.gguf["total"]), "gguf"
    except Exception:
        pass
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])", info.id.split("/")[-1])
    if m:
        return int(float(m.group(1)) * 1e9), "from_name"
    m = re.search(r"(\d+(?:\.\d+)?)\s*[mM](?![a-zA-Z])", info.id.split("/")[-1])
    if m:
        return int(float(m.group(1)) * 1e6), "from_name"
    return None, "unknown"


def license_tag_from(info):
    cd = info.card_data or {}
    try:
        cd = cd.to_dict() if hasattr(cd, "to_dict") else dict(cd)
    except Exception:
        cd = {}
    lic = cd.get("license")
    if isinstance(lic, list):
        lic = ",".join(str(x) for x in lic)
    if not lic:
        for t in info.tags or []:
            if t.startswith("license:"):
                lic = t.split(":", 1)[1]
                break
    return (lic or ""), cd.get("license_name") or "", cd.get("license_link") or ""


def read_license_file(api, repo_id, siblings, token):
    names = [s.rfilename for s in siblings or []]
    cands = [n for n in names if re.match(r"^(LICENSE|LICENCE|License|license)(\.[A-Za-z0-9_.-]+)?$", n)]
    cands += [n for n in names if n not in cands and re.search(r"(?i)license", n) and n.lower().endswith((".md", ".txt"))]
    for n in cands[:3]:
        try:
            p = hf_hub_download(repo_id, n, token=token)
            return n, Path(p).read_text(errors="replace")[:200000]
        except GatedRepoError:
            return n, "GATED_NOT_ACCESSIBLE"
        except Exception as e:  # noqa
            continue
    return "", ""


def count_filter(api, flt, cap=2000):
    n = 0
    try:
        for _ in api.list_models(filter=flt, limit=cap):
            n += 1
            if n >= cap:
                break
    except Exception:
        return None
    return n


def age_days(created, now):
    if not created:
        return None
    if created.tzinfo is None:
        created = created.replace(tzinfo=dt.timezone.utc)
    return max((now - created).days, 1)


def detail(api, repo_id, now, token, skip_finetune_counts=False):
    info = api.model_info(repo_id, expand=DETAIL_EXPAND, token=token)
    lic_tag, lic_name, lic_link = license_tag_from(info)
    lic_file, lic_text = read_license_file(api, repo_id, info.siblings, token)
    status, reason = classify_license(lic_tag, lic_text if lic_text != "GATED_NOT_ACCESSIBLE" else "")
    if lic_text == "GATED_NOT_ACCESSIBLE" and status == "UNKNOWN_REVIEW":
        reason += " (gated: LICENSE not readable without accepting terms)"
    files = [s.rfilename for s in info.siblings or []]
    has_weights = any(f.lower().endswith(WEIGHT_EXT) for f in files)
    params, params_src = params_from(info)
    n_days = age_days(info.created_at, now)
    dl30 = info.downloads or 0
    dl_all = info.downloads_all_time or 0
    tags = info.tags or []
    is_derivative = any(t.startswith(("base_model:quantized:", "base_model:adapter:", "base_model:merge:")) for t in tags)
    is_finetune = any(t.startswith("base_model:finetune:") for t in tags)
    providers = []
    try:
        for m in info.inference_provider_mapping or []:
            providers.append(f"{m.provider}:{m.status}")
    except Exception:
        pass
    spaces = list(info.spaces or [])
    rec = {
        "repo_id": info.id, "author": info.author, "pipeline_tag": info.pipeline_tag, "library": info.library_name,
        "created_at": info.created_at.isoformat() if info.created_at else None,
        "last_modified": info.last_modified.isoformat() if info.last_modified else None,
        "age_days": n_days, "downloads_30d": dl30, "downloads_all_time": dl_all,
        "downloads_30d_per_day": round(dl30 / 30, 1),
        "all_time_downloads_per_day": round(dl_all / n_days, 1) if n_days else None,
        "likes": info.likes or 0, "trending_score": info.trending_score,
        "params": params, "params_billions": round(params / 1e9, 2) if params else None, "params_source": params_src,
        "gated": info.gated if info.gated is not None else False,
        "license_tag": lic_tag, "license_name": lic_name, "license_link": lic_link, "license_file": lic_file,
        "license_status": status, "license_reason": reason,
        "self_hostable_weights": has_weights, "weight_formats": sorted({os.path.splitext(f)[1] for f in files if f.lower().endswith(WEIGHT_EXT)}),
        "is_derivative_copy": is_derivative, "is_finetune_of_other": is_finetune,
        "children_model_count": info.children_model_count,
        "spaces_count": len(spaces), "spaces_sample": spaces[:10],
        "inference_providers": providers, "inference_providers_count": len([p for p in providers if p.endswith(":live")]),
        "tags": [t for t in tags if ":" in t][:40],
    }
    if not skip_finetune_counts:
        rec["finetunes"] = count_filter(api, f"base_model:finetune:{info.id}")
        rec["adapters"] = count_filter(api, f"base_model:adapter:{info.id}")
        rec["quantized"] = count_filter(api, f"base_model:quantized:{info.id}")
        rec["merges"] = count_filter(api, f"base_model:merge:{info.id}")
    return rec


def borda_rank(recs, keys):
    for k in keys:
        order = sorted(recs, key=lambda r: (r.get(k) is None, -(r.get(k) or 0)))
        for i, r in enumerate(order):
            r.setdefault("_ranks", {})[k] = i + 1
    for r in recs:
        r["rank_sum"] = sum(r["_ranks"].values())
    recs.sort(key=lambda r: r["rank_sum"])
    for i, r in enumerate(recs):
        r["rank"] = i + 1
    return recs


def run_area(api, key, cfg, limit, now, token, include_derivatives, skip_finetune_counts):
    print(f"\n=== Area {key}: {cfg['name']} ===", flush=True)
    positions = {}
    for tag in cfg["pipeline_tags"]:
        for sort in SORTS:
            kw = dict(pipeline_tag=tag, sort=sort, limit=limit, expand=["trendingScore", "downloads", "createdAt", "tags"], token=token)
            if cfg["num_parameters"]:
                kw["num_parameters"] = cfg["num_parameters"]
            try:
                lst = list(api.list_models(**kw))
            except Exception as e:
                print(f"  list_models({tag},{sort}) failed: {e}", file=sys.stderr)
                continue
            print(f"  {tag:32s} sort={sort:15s} -> {len(lst)} models")
            for i, m in enumerate(lst):
                positions.setdefault(m.id, {})[f"pos_{sort}_{tag}"] = i + 1
    recs = []
    for i, (rid, pos) in enumerate(positions.items()):
        try:
            rec = detail(api, rid, now, token, skip_finetune_counts)
        except Exception as e:
            print(f"  detail({rid}) failed: {e}", file=sys.stderr)
            rec = {"repo_id": rid, "error": str(e)}
        rec.update(pos)
        recs.append(rec)
        if (i + 1) % 10 == 0:
            print(f"  details {i+1}/{len(positions)}", flush=True)
        time.sleep(0.2)
    ranked = [r for r in recs if "error" not in r and (include_derivatives or not r["is_derivative_copy"])]
    dropped = [r for r in recs if "error" not in r and not include_derivatives and r["is_derivative_copy"]]
    borda_rank(ranked, ["downloads_30d", "all_time_downloads_per_day", "trending_score", "likes", "children_model_count", "spaces_count", "inference_providers_count"])
    return recs, ranked, dropped


CSV_COLS = ["rank", "repo_id", "license_status", "license_tag", "gated", "self_hostable_weights", "params_billions",
            "created_at", "age_days", "downloads_30d", "downloads_30d_per_day", "downloads_all_time",
            "all_time_downloads_per_day", "likes", "trending_score", "finetunes", "adapters", "quantized", "merges",
            "children_model_count", "spaces_count", "inference_providers_count", "inference_providers", "pipeline_tag",
            "library", "weight_formats", "license_reason", "rank_sum"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=40, help="models per (sort x pipeline_tag) list")
    ap.add_argument("--areas", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--include-derivatives", action="store_true", help="keep quantized/adapter/merge copies in the ranking")
    ap.add_argument("--skip-finetune-counts", action="store_true", help="faster: skip the 4 base_model filter counts per model")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=a.token)
    now = dt.datetime.now(dt.timezone.utc)
    import huggingface_hub
    meta = {"pulled_at": now.isoformat(), "huggingface_hub": huggingface_hub.__version__, "limit": a.limit,
            "areas": a.areas, "sorts": SORTS, "include_derivatives": a.include_derivatives}
    xlsx = {}
    for key in a.areas:
        cfg = AREAS[key]
        recs, ranked, dropped = run_area(api, key, cfg, a.limit, now, a.token, a.include_derivatives, a.skip_finetune_counts)
        (out / f"area_{key}_candidates.json").write_text(json.dumps(recs, indent=1, default=str))
        with (out / f"area_{key}_ranked.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
            w.writeheader()
            for r in ranked:
                w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})
        xlsx[key] = ranked
        meta[f"area_{key}"] = {"candidates": len(recs), "ranked": len(ranked), "derivative_copies_dropped": len(dropped),
                               "not_allowed": sum(r["license_status"] == "NOT_ALLOWED" for r in ranked),
                               "needs_review": sum(r["license_status"] == "UNKNOWN_REVIEW" for r in ranked)}
        print(f"  ranked {len(ranked)} (dropped {len(dropped)} derivative copies). Top 10:")
        for r in ranked[:10]:
            print(f"   {r['rank']:2d} {r['repo_id']:50s} dl30={r['downloads_30d']:>9} dl/day(all)={r['all_time_downloads_per_day']:>8} trend={r['trending_score']} lic={r['license_status']}")
    (out / "pull_meta.json").write_text(json.dumps(meta, indent=1))
    try:
        import pandas as pd
        with pd.ExcelWriter(out / "summary.xlsx") as xw:
            for key, ranked in xlsx.items():
                pd.DataFrame(ranked)[[c for c in CSV_COLS if ranked and c in ranked[0]]].to_excel(xw, sheet_name=f"area_{key}", index=False)
    except Exception as e:
        print(f"xlsx skipped: {e}", file=sys.stderr)
    print(f"\nwrote {out}/")


if __name__ == "__main__":
    main()
