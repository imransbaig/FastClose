#!/usr/bin/env python3
"""Area A runner: extract fields from scanned tax-document images (W-2, 1099s, K-1s, IRS transcripts, notices).

    python eval/run_docs.py --dry-run                                   # build every prompt, print the first, call nothing
    python eval/run_docs.py --runner openai_compat --model Qwen/Qwen2.5-VL-7B-Instruct --name qwen25vl7b \
        --base-url http://localhost:8000/v1 [--json-mode]               # vLLM / llama.cpp server / Ollama /v1 / LM Studio / SGLang
    python eval/run_docs.py --runner transformers  --model Qwen/Qwen2.5-VL-7B-Instruct --name qwen25vl7b
    python eval/run_docs.py --runner manual        --model claude-desktop --name baseline_claude
    python eval/score.py --area docs --pred results/pred_docs_<name>.json --name <name>

Items come from {data-root}/docs/ground_truth.json:
    {"doc01": {"doc_type": "W2", "png": "doc01.png", "fields": {...}}, ...}
Every runner sends the SAME prompt, prompts.doc_prompt(doc_type), with the image, so models are compared fairly.
The PNG is downscaled with Pillow so its longest side is <= --image-max-side (default 1600, 0 = never) before it is
base64-encoded / handed to the processor; the size actually sent is recorded as image_size_sent.

Output (the "docs" format documented in score.py), written after EVERY item so a crash keeps progress:
    {"doc01": {"fields": {...} | null, "raw_output": "...", "latency_sec": 1.2, "json_valid": true, "model": "...",
               "runner": "...", "doc_type": "W2", "image_size_sent": [w, h], "retry": false, ...}}
If the first answer is not a JSON object the item is re-asked ONCE with the prompt plus
"Your previous answer was not valid JSON. Output only the JSON object." and recorded with retry=true.
latency_sec is the wall time spent in model calls for the item (both attempts when a retry happened); null for manual.
Resume: ids already present in the output file are skipped unless --overwrite, which re-runs the selected ids (all, or
--ids) and keeps the rest of the file; items that errored are always re-run.
A sidecar <out>.meta.json records the arguments and the run summary.

Manual runner: writes results/manual_docs_<name>/<id>.prompt.txt (first line = path of the image as sent, then the
prompt) and <id>.png; when <id>.response.txt exists it is read back as the model output. An invalid response gets
<id>.retry.prompt.txt, answered in <id>.retry.response.txt. Re-run the same command after answering prompts.

Only Pillow + requests are hard requirements; torch / transformers are imported lazily by the transformers runner.
"""
import argparse
import base64
import datetime as dt
import io
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from prompts import FIELDS, doc_prompt, parse_json_object  # noqa: E402

AREA = "docs"
RUNNERS = ("openai_compat", "transformers", "llama_cpp", "manual")
RETRY_SUFFIX = "\nYour previous answer was not valid JSON. Output only the JSON object."


# --- images ---------------------------------------------------------------------------------
def load_image(path, max_side):
    """Open the PNG with Pillow, convert to RGB and shrink so max(w, h) <= max_side (keep aspect; 0 = no limit).
    Returns (PIL.Image, [w, h])."""
    from PIL import Image
    img = Image.open(path)
    img.load()
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max_side and max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    return img, list(img.size)


def png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def data_url(img):
    return "data:image/png;base64," + base64.b64encode(png_bytes(img)).decode("ascii")


# --- backends -------------------------------------------------------------------------------
class Backend:
    """generate() returns the model's text, or None when the answer is not available yet (manual runner)."""
    name = "base"

    def __init__(self, args):
        self.args = args

    def generate(self, prompt, img, item_id, retry=False):
        raise NotImplementedError


def _message_text(resp):
    msg = resp["choices"][0]["message"]
    content = msg.get("content")
    if isinstance(content, list):  # some servers return content parts
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


class OpenAICompat(Backend):
    """POST {base_url}/chat/completions with the image as a data URL. Works with vLLM, llama.cpp server
    (llama-server --mmproj ...), Ollama /v1, LM Studio and SGLang. One retry on HTTP error / timeout."""
    name = "openai_compat"

    def __init__(self, args):
        super().__init__(args)
        import requests
        self.requests = requests
        self.url = args.base_url.rstrip("/") + "/chat/completions"
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {args.api_key}"}

    def generate(self, prompt, img, item_id, retry=False):
        body = {"model": self.args.model,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_url(img)}},
                    {"type": "text", "text": prompt}]}],
                "max_tokens": self.args.max_tokens, "temperature": self.args.temperature}
        if self.args.json_mode:
            body["response_format"] = {"type": "json_object"}
        last = None
        for attempt in (1, 2):
            try:
                r = self.requests.post(self.url, json=body, headers=self.headers, timeout=self.args.timeout)
                if r.status_code >= 400:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
                return _message_text(r.json())
            except (self.requests.RequestException, RuntimeError, ValueError, KeyError, IndexError, TypeError) as e:
                last = e
                if attempt == 1:
                    print(f"    {item_id}: request failed ({type(e).__name__}: {str(e)[:160]}); retrying once", flush=True)
                    time.sleep(2)
        raise RuntimeError(f"openai_compat request failed twice: {type(last).__name__}: {last}")


class Transformers(Backend):
    """Local inference: AutoProcessor + AutoModelForImageTextToText, chat template with image + text, decode new tokens."""
    name = "transformers"

    def __init__(self, args):
        super().__init__(args)
        import torch
        import transformers
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        dev = args.device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = {"auto": "auto", "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
        kw = {"trust_remote_code": args.trust_remote_code}
        ver = tuple(int(x) for x in re.findall(r"\d+", transformers.__version__)[:2])
        kw["dtype" if ver >= (4, 56) else "torch_dtype"] = dtype  # kwarg renamed in transformers 4.56
        if dev == "cuda":
            kw["device_map"] = "auto"
        print(f"loading {args.model} on {dev} (dtype={args.dtype}, transformers {transformers.__version__})", flush=True)
        self.processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
        self.model = AutoModelForImageTextToText.from_pretrained(args.model, **kw)
        if dev != "cuda":
            self.model.to(dev)
        self.model.eval()

    def generate(self, prompt, img, item_id, retry=False):
        torch, proc, model = self.torch, self.processor, self.model
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
        try:
            inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                              return_dict=True, return_tensors="pt")
        except Exception:  # older transformers: template renders text, the processor binds the image
            msgs_txt = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            text = proc.apply_chat_template(msgs_txt, add_generation_prompt=True, tokenize=False)
            inputs = proc(text=[text], images=[img], return_tensors="pt")
        inputs = inputs.to(model.device)
        if "pixel_values" in inputs and model.dtype in (torch.bfloat16, torch.float16):
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
        gen = {"max_new_tokens": self.args.max_tokens, "do_sample": self.args.temperature > 0}
        if self.args.temperature > 0:
            gen["temperature"] = self.args.temperature
        with torch.inference_mode():
            out = model.generate(**inputs, **gen)
        new = out[:, inputs["input_ids"].shape[-1]:]  # only the generated tokens
        return proc.batch_decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


class Manual(Backend):
    """Writes prompt files for a human (or Claude in the desktop app) and reads answers back from response files."""
    name = "manual"

    def __init__(self, args):
        super().__init__(args)
        self.dir = ROOT / "results" / f"manual_{AREA}_{args.name}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pending = []

    def generate(self, prompt, img, item_id, retry=False):
        stem = f"{item_id}.retry" if retry else item_id
        img_path = self.dir / f"{item_id}.png"
        if not retry:
            img_path.write_bytes(png_bytes(img))  # the image exactly as it would be sent to a model
        (self.dir / f"{stem}.prompt.txt").write_text(f"{img_path}\n\n{prompt}\n", encoding="utf-8")
        resp = self.dir / f"{stem}.response.txt"
        if resp.exists():
            return resp.read_text(encoding="utf-8")
        self.pending.append(stem)
        return None


def make_backend(args):
    if args.runner == "openai_compat":
        return OpenAICompat(args)
    if args.runner == "transformers":
        return Transformers(args)
    if args.runner == "manual":
        return Manual(args)
    if args.runner == "llama_cpp":
        sys.exit("--runner llama_cpp is text-only (notices / calls). For images run a llama.cpp server with --mmproj "
                 "and use --runner openai_compat --base-url http://localhost:8080/v1")
    raise ValueError(args.runner)


# --- items ----------------------------------------------------------------------------------
def load_items(args):
    gt_path = Path(args.data_root) / "docs" / "ground_truth.json"
    if not gt_path.exists():
        sys.exit(f"ground truth not found: {gt_path} (expected {{id: {{doc_type, png, fields}}}}; generate the docs first)")
    gt = json.load(open(gt_path, encoding="utf-8"))
    ids = list(gt)
    if args.ids:
        want = [s.strip() for s in args.ids.split(",") if s.strip()]
        unknown = [i for i in want if i not in gt]
        if unknown:
            sys.exit(f"--ids not in ground truth: {unknown}")
        ids = [i for i in ids if i in want]
    items = []
    for i in ids:
        e = gt[i]
        if e.get("doc_type") not in FIELDS["doc_types"]:
            sys.exit(f"{i}: unknown doc_type {e.get('doc_type')!r}; known: {list(FIELDS['doc_types'])}")
        items.append({"id": i, "doc_type": e["doc_type"], "png": Path(args.data_root) / "docs" / e["png"],
                      "prompt": doc_prompt(e["doc_type"])})
    return items


def run_item(backend, item, args):
    """Returns the record for the output file, or None when the manual answer is not written yet."""
    doc_type, prompt = item["doc_type"], item["prompt"]
    img, size = load_image(item["png"], args.image_max_side)
    rec = {"id": item["id"], "doc_type": doc_type, "model": args.model, "runner": args.runner,
           "image_path": str(item["png"]), "image_size_sent": size, "image_max_side": args.image_max_side,
           "json_mode": bool(args.json_mode), "retry": False}
    t0 = time.perf_counter()
    raw = backend.generate(prompt, img, item["id"])
    latency = time.perf_counter() - t0
    if raw is None:
        return None
    fields, ok = parse_json_object(raw)
    if not ok:
        rec["retry"] = True
        rec["raw_output_attempt1"] = raw
        t1 = time.perf_counter()
        raw2 = backend.generate(prompt + RETRY_SUFFIX, img, item["id"], retry=True)
        latency += time.perf_counter() - t1
        if raw2 is not None:
            raw, (fields, ok) = raw2, parse_json_object(raw2)
    expected = FIELDS["doc_types"][doc_type]
    rec.update({"raw_output": raw, "fields": fields if ok else None, "json_valid": bool(ok),
                "latency_sec": None if args.runner == "manual" else round(latency, 3),
                "keys_present": sum(1 for k in expected if k in fields) if ok else 0, "keys_expected": len(expected),
                "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
    return rec


def write_json(path, data):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def dry_run(items, args):
    print(f"dry run: {len(items)} item(s) from {Path(args.data_root) / 'docs' / 'ground_truth.json'}; no model called\n")
    for it in items:
        if it["png"].exists():
            try:
                _, size = load_image(it["png"], args.image_max_side)
                note = f"image_size_sent={size[0]}x{size[1]}"
            except Exception as e:  # noqa: BLE001
                note = f"IMAGE UNREADABLE: {e}"
        else:
            note = "IMAGE MISSING"
        print(f"  {it['id']:10s} {it['doc_type']:28s} {len(FIELDS['doc_types'][it['doc_type']]):2d} fields "
              f"{len(it['prompt']):5d} chars  {note}")
    print("\n--- first prompt (" + items[0]["id"] + ", " + items[0]["doc_type"] + ") ---")
    print(items[0]["prompt"])
    print("--- end prompt ---")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runner", choices=RUNNERS, default="openai_compat")
    ap.add_argument("--model", help="HF repo id, local path, or the served model name for openai_compat")
    ap.add_argument("--name", help="short label used in the output filename (default: derived from --model)")
    ap.add_argument("--data-root", default=str(ROOT / "data"), help=f"default {ROOT / 'data'}")
    ap.add_argument("--out", help=f"default {ROOT / 'results'}/pred_{AREA}_<name>.json")
    ap.add_argument("--ids", help="comma list of doc ids to run (default: all)")
    ap.add_argument("--overwrite", action="store_true", help="re-run ids already present in the output file")
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY") or "none")
    ap.add_argument("--json-mode", action="store_true", help='send response_format={"type":"json_object"} (openai_compat)')
    ap.add_argument("--timeout", type=float, default=600, help="seconds per openai_compat request")
    ap.add_argument("--image-max-side", type=int, default=1600, help="downscale longest side to this many px (0 = never)")
    ap.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto", help="transformers runner")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto", help="transformers runner")
    ap.add_argument("--trust-remote-code", action="store_true", help="transformers runner")
    ap.add_argument("--dry-run", action="store_true", help="build every prompt, print the first one, call no model")
    args = ap.parse_args(argv)

    items = load_items(args)
    if not items:
        sys.exit("no items to run")
    if args.dry_run:
        dry_run(items, args)
        return 0
    if not args.model:
        ap.error("--model is required (unless --dry-run)")
    if not args.name:
        args.name = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(args.model.rstrip("/")).name) or "model"
        print(f"--name not given, using {args.name!r}")
    out_path = Path(args.out) if args.out else ROOT / "results" / f"pred_{AREA}_{args.name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = json.load(open(out_path, encoding="utf-8")) if out_path.exists() else {}
    # --overwrite re-runs the selected ids (all, or --ids) and replaces their records; other ids in the file are kept
    todo = [it for it in items if args.overwrite or it["id"] not in out or "error" in out[it["id"]]]
    print(f"{AREA} / {args.name}: {len(items)} item(s), {len(items) - len(todo)} already done, {len(todo)} to run "
          f"-> {out_path}")
    if not todo:
        return 0

    backend = make_backend(args)
    meta_path = out_path.with_name(out_path.stem + ".meta.json")
    meta = {"area": AREA, "name": args.name, "args": vars(args), "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "prompt_module": str(HERE / "prompts.py")}
    write_json(meta_path, meta)
    errors, pending, latencies = 0, 0, []
    for n, it in enumerate(todo, 1):
        tag = f"[{n}/{len(todo)}] {it['id']} {it['doc_type']}"
        try:
            rec = run_item(backend, it, args)
        except KeyboardInterrupt:
            print("\ninterrupted; progress is saved in", out_path)
            return 130
        except Exception as e:  # noqa: BLE001  record the failure and keep going; the id is re-run on the next invocation
            errors += 1
            out[it["id"]] = {"id": it["id"], "doc_type": it["doc_type"], "model": args.model, "runner": args.runner,
                             "error": f"{type(e).__name__}: {e}", "fields": None, "json_valid": False, "raw_output": "",
                             "latency_sec": None}
            write_json(out_path, out)
            print(f"{tag} ERROR {type(e).__name__}: {str(e)[:200]}", flush=True)
            continue
        if rec is None:
            pending += 1
            print(f"{tag} pending: answer {backend.dir / (it['id'] + '.response.txt')}", flush=True)
            continue
        out[it["id"]] = rec
        write_json(out_path, out)
        if rec["latency_sec"] is not None:
            latencies.append(rec["latency_sec"])
        lat = f"{rec['latency_sec']:.1f}s" if rec["latency_sec"] is not None else "n/a"
        w, h = rec["image_size_sent"]
        print(f"{tag} json_valid={rec['json_valid']} keys={rec['keys_present']}/{rec['keys_expected']} "
              f"retry={rec['retry']} latency={lat} img={w}x{h}", flush=True)
        if args.runner == "manual" and not rec["json_valid"]:
            print(f"    invalid JSON; answer {backend.dir / (it['id'] + '.retry.response.txt')} and re-run with "
                  f"--overwrite --ids {it['id']}", flush=True)

    done = [r for r in out.values() if "error" not in r]
    valid = sum(1 for r in done if r.get("json_valid"))
    retried = sum(1 for r in done if r.get("retry"))
    summary = {"items_total": len(items), "items_done": len(done), "json_valid": valid, "retried": retried,
               "errors_this_run": errors, "pending_manual": pending,
               "mean_latency_sec": round(sum(latencies) / len(latencies), 3) if latencies else None,
               "finished_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    meta["summary"] = summary
    write_json(meta_path, meta)
    print(f"\nsummary: {len(done)}/{len(items)} items in {out_path}; json_valid {valid}/{len(done)}; retried {retried}; "
          f"errors this run {errors}; pending manual {pending}"
          + (f"; mean latency {summary['mean_latency_sec']}s" if latencies else ""))
    if pending:
        print(f"answer the pending prompts in {backend.dir} and re-run the same command (done ids are skipped)")
    if len(done) == len(items):
        print(f"next: python {HERE / 'score.py'} --area {AREA} --pred {out_path} --name {args.name}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
