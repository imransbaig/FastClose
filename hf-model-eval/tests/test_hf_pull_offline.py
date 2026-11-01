"""Offline test of hf_pull.py: exercises detail(), ranking and CSV writing with a fake HfApi (no network).
    python tests/test_hf_pull_offline.py
"""
import datetime as dt
import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hf_pull  # noqa: E402


class Obj(types.SimpleNamespace):
    pass


def fake_info(rid, days_old, dl30, dl_all, lic, gated=False, params=7_000_000_000, files=("model.safetensors", "LICENSE", "README.md"), tags=()):
    now = dt.datetime.now(dt.timezone.utc)
    return Obj(id=rid, author=rid.split("/")[0], pipeline_tag="text-generation", library_name="transformers",
               created_at=now - dt.timedelta(days=days_old), last_modified=now, downloads=dl30, downloads_all_time=dl_all,
               likes=10, trending_score=5, safetensors=Obj(total=params), gguf=None, gated=gated,
               card_data={"license": lic}, tags=list(tags) + [f"license:{lic}"], siblings=[Obj(rfilename=f) for f in files],
               inference_provider_mapping=[Obj(provider="together", status="live")], spaces=["a/s1", "b/s2"], children_model_count=12)


class FakeApi:
    def __init__(self):
        self.infos = {
            "org/good-7B": fake_info("org/good-7B", 400, 300000, 4000000, "apache-2.0"),
            "org/nc-3B": fake_info("org/nc-3B", 30, 50000, 50000, "cc-by-nc-4.0", params=3e9),
            "org/gated-8B": fake_info("org/gated-8B", 100, 120000, 900000, "llama3.1", gated="auto", params=8e9),
            "quant/good-7B-GGUF": fake_info("quant/good-7B-GGUF", 10, 90000, 90000, "apache-2.0", files=("q4.gguf",), tags=["base_model:quantized:org/good-7B"]),
        }

    def list_models(self, **kw):
        if "filter" in kw:
            return iter([1, 2, 3])
        return iter([Obj(id=k) for k in self.infos])

    def model_info(self, rid, **kw):
        return self.infos[rid]


def fake_download(repo_id, filename, token=None):
    p = Path("/tmp/claude-0/-home-user-FastClose/334d624a-ab4a-5c18-a804-35185029be6c/scratchpad/fake_LICENSE")
    p.write_text("Apache License Version 2.0 ... you may use for any purpose")
    return str(p)


hf_pull.hf_hub_download = fake_download
api = FakeApi()
now = dt.datetime.now(dt.timezone.utc)
recs, ranked, dropped = hf_pull.run_area(api, "C", hf_pull.AREAS["C"], 10, now, None, False, False)
assert len(recs) == 4 and len(dropped) == 1 and dropped[0]["repo_id"] == "quant/good-7B-GGUF", (len(recs), dropped)
byid = {r["repo_id"]: r for r in ranked}
assert byid["org/good-7B"]["license_status"] == "ALLOWED"
assert byid["org/nc-3B"]["license_status"] == "NOT_ALLOWED"
assert byid["org/gated-8B"]["license_status"] == "ALLOWED_WITH_CONDITIONS" and byid["org/gated-8B"]["gated"]
assert byid["org/good-7B"]["all_time_downloads_per_day"] == round(4000000 / 400, 1)
assert byid["org/good-7B"]["finetunes"] == 3 and byid["org/good-7B"]["params_billions"] == 7.0
assert ranked[0]["rank"] == 1 and all("rank_sum" in r for r in ranked)
print("ranking:", [(r["rank"], r["repo_id"], r["rank_sum"]) for r in ranked])
# csv writing path
import csv, io
buf = io.StringIO()
w = csv.DictWriter(buf, fieldnames=hf_pull.CSV_COLS, extrasaction="ignore")
w.writeheader()
for r in ranked:
    w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})
assert "org/good-7B" in buf.getvalue()
print("OK: hf_pull offline test passed")
