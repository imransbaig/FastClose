#!/usr/bin/env python3
"""Area B stage 2: turn a client-call transcript into the firm's call-log JSON (client_name / entity_type /
tax_year_discussed / topic_code / key_amount / key_date / follow_up_action_code) with any open text-generation model,
so score.py can compare every field with the ground truth and with the baseline.

Pipeline:  data/calls/callNN.wav --eval/run_asr.py--> results/asr_<asrname>.json
                                 --THIS SCRIPT-----> results/pred_calls_<name>.json
                                 --eval/score.py---> field exact match + WER of the transcript used

Every model gets the IDENTICAL prompt from eval/prompts.py (call_summary_prompt) - never edit the prompt here.

Transcript source (exactly one):
  --transcripts results/asr_<asrname>.json   output of eval/run_asr.py: {"call01": {"transcript": "...", "model": "...", ...}}
  --use-reference                            {data-root}/calls/<id>.ref.txt (the ground-truth transcript) for every id in
                                             {data-root}/calls/ground_truth.json, labelled asr_model="reference". Use it to
                                             measure the summarizer alone, with no ASR errors (upper bound for stage 2).

Output (the "calls" format documented in eval/score.py), one record per id, rewritten after EVERY item so a crash keeps
progress; rerunning the same command resumes (ids that hold an answer are skipped; pending / errored ids are redone).
--overwrite redoes the selected ids (with --ids only those records are replaced and the rest of the file is kept;
without --ids the file starts fresh):

    {"call01": {"transcript": "<the transcript used>", "summary": {...} | null, "json_valid": true, "raw_output": "...",
                "asr_model": "models/sherpa-onnx-whisper-large-v3", "asr_name": "whisper-large-v3", "asr_runner": "...",
                "summary_model": "Qwen/Qwen3-4B-Instruct-2507", "latency_sec": 1.23, "runner": "openai_compat",
                "id": "call01", "fields": <same object as summary>, "model": <same as summary_model>,
                "retry": false, "pending": false, "error": null, "schema_issues": [], "usage": {...}, "ts": "..."}}

If the first answer is not a JSON object the item is re-asked ONCE with
"Your previous answer was not valid JSON. Output only the JSON object." and recorded with retry=true.
latency_sec = wall time inside model calls for the item (both attempts when a retry happened); null for the manual runner.
A sidecar <out>.meta.json records the arguments (incl. which ASR file fed the run) and the run summary.

Examples (run from the repo root):

  # summarize the Whisper transcripts with a model served by vLLM / Ollama / llama.cpp server / LM Studio / SGLang:
  python eval/summarize_calls.py --transcripts results/asr_whisper-large-v3.json --runner openai_compat \\
      --base-url http://localhost:8000/v1 --model Qwen/Qwen3-4B-Instruct-2507 --name qwen3-4b_whisper-large-v3 --json-mode
  python eval/summarize_calls.py --transcripts results/asr_parakeet-v2.json --runner openai_compat \\
      --base-url http://localhost:11434/v1 --model llama3.2:3b --json-mode             # Ollama

  # summarizer-only upper bound on the reference transcripts:
  python eval/summarize_calls.py --use-reference --runner openai_compat --model Qwen/Qwen3-4B-Instruct-2507 --json-mode

  # transformers (needs torch + transformers; downloads the model from the Hub):
  python eval/summarize_calls.py --use-reference --runner transformers --model microsoft/Phi-4-mini-instruct

  # llama.cpp python binding on a local GGUF:
  python eval/summarize_calls.py --use-reference --runner llama_cpp --model models/qwen3-4b-q4_k_m.gguf --json-mode

  # manual (baseline: a human, or Claude in the desktop app, answers the exact same prompts):
  python eval/summarize_calls.py --use-reference --runner manual --name claude-baseline
      -> writes results/manual_calls_claude-baseline/<id>.prompt.txt ; put each answer in <id>.response.txt and rerun
         the same command to collect them (unanswered ids stay "pending" and are retried on rerun).

  # see what would run (ids, transcript source, word counts, first prompt) without touching any model:
  python eval/summarize_calls.py --use-reference --runner openai_compat --model x --dry-run

Then:  python eval/score.py --area calls --pred results/pred_calls_<name>.json --name <name>

Only the standard library is needed to import this file; requests / torch / transformers / llama_cpp are imported
lazily by the runner that uses them.
"""
import argparse
import datetime as dt
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from prompts import FIELDS, call_summary_prompt, parse_json_object  # noqa: E402  (shared prompts - do not duplicate)

AREA = "calls"
CALL_FIELDS = list(FIELDS["call_summary_fields"])
CALL_ENUMS = {k: set(v) for k, v in FIELDS["call_enums"].items()}
RETRY_SUFFIX = "\nYour previous answer was not valid JSON. Output only the JSON object."


# ----------------------------------------------------------------------------------------------------------------
# Runner backends. Each exposes  generate(messages, max_tokens, temperature, item_id, attempt) -> Gen
# Heavy deps (requests / torch / transformers / llama_cpp) are imported lazily inside the constructors so this
# module imports and --dry-run works on a machine with none of them installed.
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class Gen:
    text: str = ""
    pending: bool = False          # manual runner only: no response file yet
    usage: dict | None = None      # token counts when the backend reports them


class OpenAICompatRunner:
    """POST {base_url}/chat/completions with plain requests. Works with vLLM, SGLang, llama.cpp server, LM Studio,
    Ollama (/v1). Retries once on HTTP error / timeout / connection error (timeout 600 s by default)."""
    name = "openai_compat"

    def __init__(self, base_url, api_key, model, json_mode=False, timeout=600, extra_body=None):
        import requests
        self.requests = requests
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.json_mode = json_mode
        self.timeout = timeout
        self.extra_body = extra_body or {}
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    def generate(self, messages, max_tokens, temperature, **_):
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        body.update(self.extra_body)
        data = None
        for attempt in (1, 2):
            try:
                resp = self.requests.post(self.url, headers=self.headers, json=body, timeout=self.timeout)
                if resp.status_code >= 400:
                    raise RuntimeError(f"HTTP {resp.status_code} from {self.url}: {resp.text[:300]}")
                data = resp.json()
                if "choices" not in data:
                    raise RuntimeError(f"no 'choices' in response: {json.dumps(data)[:300]}")
                break
            except (self.requests.RequestException, RuntimeError, ValueError) as e:
                if attempt == 2:
                    raise
                print(f"    request failed ({type(e).__name__}: {str(e)[:160]}); retrying once", flush=True)
                time.sleep(2)
        msg = data["choices"][0].get("message") or {}
        content = msg.get("content")
        if isinstance(content, list):  # some servers return content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        return Gen(text=content or "", usage=data.get("usage"))


class TransformersRunner:
    """AutoTokenizer + AutoModelForCausalLM, apply_chat_template(add_generation_prompt=True), decode new tokens only."""
    name = "transformers"

    def __init__(self, model_id, dtype="auto", device="auto", trust_remote_code=False, json_mode=False, chat_kwargs=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.chat_kwargs = chat_kwargs or {}
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        td = "auto" if dtype == "auto" else getattr(torch, dtype)
        kw = {"trust_remote_code": trust_remote_code}
        if self.device == "cuda":
            kw["device_map"] = "auto"
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=td, **kw)
        except TypeError:  # older transformers: kwarg is torch_dtype
            self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=td, **kw)
        if self.device != "cuda":
            self.model.to(self.device)
        self.model.eval()
        if json_mode:
            print("note: --json-mode has no effect on the transformers runner (no constrained decoding); ignored", flush=True)
        print(f"loaded {model_id} on {self.model.device} dtype={self.model.dtype}", flush=True)

    def generate(self, messages, max_tokens, temperature, **_):
        tok = self.tok
        if getattr(tok, "chat_template", None):
            enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
                                          **self.chat_kwargs)
            if not hasattr(enc, "keys"):  # very old transformers returns a bare tensor
                enc = {"input_ids": enc}
        else:  # base model without a chat template: plain prompt
            enc = tok("\n\n".join(m["content"] for m in messages) + "\n", return_tensors="pt")
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        n_in = enc["input_ids"].shape[1]
        gen_kw = {"max_new_tokens": max_tokens, "do_sample": temperature > 0,
                  "pad_token_id": tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id}
        if temperature > 0:
            gen_kw["temperature"] = temperature
        with self.torch.inference_mode():
            out = self.model.generate(**enc, **gen_kw)
        text = tok.decode(out[0, n_in:], skip_special_tokens=True)
        return Gen(text=text, usage={"prompt_tokens": int(n_in), "completion_tokens": int(out.shape[1] - n_in)})


class LlamaCppRunner:
    """llama_cpp.Llama(model_path, n_ctx=8192, verbose=False).create_chat_completion(...); JSON grammar when --json-mode.
    A 10-minute call is roughly 1,500 words (~2,500 tokens) plus the prompt, so the default n_ctx=8192 is enough."""
    name = "llama_cpp"

    def __init__(self, model_path, n_ctx=8192, n_gpu_layers=-1, json_mode=False):
        from llama_cpp import Llama
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False)
        self.json_mode = json_mode

    def generate(self, messages, max_tokens, temperature, **_):
        kw = {"response_format": {"type": "json_object"}} if self.json_mode else {}
        r = self.llm.create_chat_completion(messages=messages, temperature=temperature, max_tokens=max_tokens, **kw)
        text = (r["choices"][0].get("message") or {}).get("content") or ""
        return Gen(text=text, usage=r.get("usage"))


class ManualRunner:
    """Writes results/manual_calls_<name>/<id>.prompt.txt (and <id>.retry.prompt.txt for the JSON retry) and reads
    <id>.response.txt / <id>.retry.response.txt back when present. Lets a human, or Claude in the desktop app,
    answer the exact same prompts offline (this is how the baseline row is produced)."""
    name = "manual"

    def __init__(self, root, name, area=AREA):
        self.dir = Path(root) / "results" / f"manual_{area}_{name}"
        self.dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def render(messages):
        if len(messages) == 1:
            return messages[0]["content"]
        parts = []
        for m in messages:
            tag = "ASSISTANT (your previous answer)" if m["role"] == "assistant" else m["role"].upper()
            parts.append(f"=== {tag} ===\n{m['content']}")
        return "\n\n".join(parts) + "\n"

    def generate(self, messages, item_id, attempt=1, **_):
        stem = item_id if attempt == 1 else f"{item_id}.retry"
        (self.dir / f"{stem}.prompt.txt").write_text(self.render(messages), encoding="utf-8")
        resp = self.dir / f"{stem}.response.txt"
        if resp.exists():
            return Gen(text=resp.read_text(encoding="utf-8", errors="replace"))
        return Gen(text="", pending=True)


def make_runner(a):
    if a.runner == "openai_compat":
        return OpenAICompatRunner(a.base_url, a.api_key, a.model, json_mode=a.json_mode, timeout=a.timeout,
                                  extra_body=_load_json_arg(a.extra_body, "--extra-body"))
    if a.runner == "transformers":
        return TransformersRunner(a.model, dtype=a.dtype, device=a.device, trust_remote_code=a.trust_remote_code,
                                  json_mode=a.json_mode, chat_kwargs=_load_json_arg(a.chat_kwargs, "--chat-kwargs"))
    if a.runner == "llama_cpp":
        return LlamaCppRunner(a.model, n_ctx=a.n_ctx, n_gpu_layers=a.n_gpu_layers, json_mode=a.json_mode)
    if a.runner == "manual":
        if a.json_mode:
            print("note: --json-mode has no effect on the manual runner; ignored", flush=True)
        return ManualRunner(ROOT, a.name)
    raise ValueError(a.runner)


def _load_json_arg(s, flag):
    if not s:
        return {}
    try:
        o = json.loads(s)
    except json.JSONDecodeError as e:
        sys.exit(f"{flag} must be a JSON object: {e}")
    if not isinstance(o, dict):
        sys.exit(f"{flag} must be a JSON object")
    return o


# ----------------------------------------------------------------------------------------------------------------
# Transcript sources
# ----------------------------------------------------------------------------------------------------------------
def _gt_ids(data_root):
    """Ids of data/calls/ground_truth.json in file order ([] when the file is absent)."""
    p = Path(data_root) / "calls" / "ground_truth.json"
    if not p.exists():
        return [], p
    gt = json.load(open(p, encoding="utf-8"))
    return [k for k, v in gt.items() if isinstance(v, dict) and not k.startswith("_")], p


def load_sources(a):
    """Returns (label, {id: {"transcript", "model", "runner", "name", "file"}}) in ground-truth order.
    label is the ASR label used in the default output name ("reference" or the asr_<label>.json stem)."""
    calls_dir = Path(a.data_root) / "calls"
    gt_ids, gt_path = _gt_ids(a.data_root)
    src = {}
    if a.use_reference:
        if not gt_ids:
            sys.exit(f"--use-reference needs {gt_path} (ids + call_summary fields); generate the calls first")
        label, missing = "reference", []
        for i in gt_ids:
            p = calls_dir / f"{i}.ref.txt"
            if not p.exists():
                missing.append(str(p))
                continue
            src[i] = {"transcript": p.read_text(encoding="utf-8", errors="replace").strip(), "model": "reference",
                      "runner": "reference", "name": label, "file": str(p)}
        if missing:
            print(f"warning: {len(missing)} reference transcript(s) missing, skipped: {missing[:5]}", flush=True)
    else:
        asr_path = Path(a.transcripts)
        if not asr_path.exists():
            sys.exit(f"ASR file not found: {asr_path} (produce it with eval/run_asr.py, or use --use-reference)")
        try:
            asr = json.load(open(asr_path, encoding="utf-8"))
        except json.JSONDecodeError as e:
            sys.exit(f"{asr_path} is not valid JSON: {e}")
        label = re.sub(r"^asr_", "", asr_path.stem)
        recs = {k: v for k, v in asr.items() if isinstance(v, dict) and not k.startswith("_")}
        ids = gt_ids or sorted(recs)
        if gt_ids:
            extra = [k for k in recs if k not in gt_ids]
            if extra:
                print(f"warning: {len(extra)} id(s) in {asr_path.name} not in ground truth, ignored: {extra[:5]}", flush=True)
        missing = [i for i in ids if i not in recs]
        if missing:
            print(f"warning: {len(missing)} id(s) have no transcript in {asr_path.name}, skipped: {missing[:5]} "
                  f"(finish eval/run_asr.py first)", flush=True)
        for i in ids:
            if i in recs:
                r = recs[i]
                src[i] = {"transcript": (r.get("transcript") or "").strip(), "model": r.get("model") or label,
                          "runner": r.get("runner"), "name": label, "file": str(asr_path)}
    if a.ids:
        want = [s.strip() for s in a.ids.split(",") if s.strip()]
        unknown = [w for w in want if w not in src]
        if unknown:
            print(f"warning: --ids without a transcript / not in ground truth, skipped: {unknown}", flush=True)
        src = {w: src[w] for w in want if w in src}
    return label, src


# ----------------------------------------------------------------------------------------------------------------
# Item loop
# ----------------------------------------------------------------------------------------------------------------
def retry_messages(prompt, raw):
    """Second attempt after invalid JSON: show the model its own answer and the fixed correction line. If the first
    answer was empty (nothing to show), append the correction line to the prompt itself."""
    if raw and raw.strip():
        return [{"role": "user", "content": prompt},
                {"role": "assistant", "content": raw},
                {"role": "user", "content": RETRY_SUFFIX.strip()}]
    return [{"role": "user", "content": prompt + RETRY_SUFFIX}]


def schema_issues(fields):
    """Cheap 'clean JSON' signal on top of json_valid: missing / extra keys and out-of-enum values. Scoring itself is
    done by score.py; this only helps rank models on JSON discipline."""
    if not isinstance(fields, dict):
        return ["not_an_object"]
    issues = [f"missing:{k}" for k in CALL_FIELDS if k not in fields]
    issues += [f"extra:{k}" for k in fields if k not in CALL_FIELDS]
    for k, allowed in CALL_ENUMS.items():
        v = fields.get(k)
        if v is not None and str(v).strip().upper().replace(" ", "_").replace("-", "_") not in allowed:
            issues.append(f"enum:{k}={v!r}"[:80])
    return issues


def run_item(runner, a, item_id, src, prompt):
    base = {"id": item_id, "transcript": src["transcript"], "asr_model": src["model"], "asr_name": src["name"],
            "asr_runner": src.get("runner"), "summary_model": a.model, "model": a.model, "runner": a.runner,
            "json_mode": bool(a.json_mode)}
    raw, fields, valid, retry, pending, err, usage, spent = "", None, False, False, False, None, None, 0.0
    if not src["transcript"]:
        err = "empty transcript: nothing sent to the model (scores as all-wrong, WER 1.0)"
    else:
        t0 = time.perf_counter()
        try:
            g = runner.generate([{"role": "user", "content": prompt}], max_tokens=a.max_tokens,
                                temperature=a.temperature, item_id=item_id, attempt=1)
            raw, pending, usage = g.text, g.pending, g.usage
            if not pending:
                fields, valid = parse_json_object(raw)
                if not valid:
                    retry = True
                    g2 = runner.generate(retry_messages(prompt, raw), max_tokens=a.max_tokens,
                                         temperature=a.temperature, item_id=item_id, attempt=2)
                    if g2.pending:
                        pending = True  # manual: retry prompt written, keep first raw until the retry answer arrives
                    else:
                        raw, usage = g2.text, g2.usage or usage
                        fields, valid = parse_json_object(raw)
        except KeyboardInterrupt:
            raise
        except Exception as e:  # record and move on; the summary counts errors
            err = f"{type(e).__name__}: {e}"[:500]
        finally:
            spent = time.perf_counter() - t0  # both attempts (and a failed request's retry) count
    summary = fields if valid else None
    rec = dict(base)
    rec.update({"summary": summary, "fields": summary, "json_valid": bool(valid), "raw_output": raw or "",
                "latency_sec": None if a.runner == "manual" else round(spent, 3),
                "retry": retry, "pending": pending, "error": err,
                "schema_issues": schema_issues(fields) if valid else None,
                "usage": usage, "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
    return rec


def write_json(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def summarize(fields):
    if not isinstance(fields, dict):
        return "-"
    return " ".join(f"{k.split('_')[0]}={str(fields.get(k))[:16]}"
                    for k in ("client_name", "topic_code", "key_amount", "key_date", "follow_up_action_code"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runner", required=True, choices=["openai_compat", "transformers", "llama_cpp", "manual"])
    ap.add_argument("--model", help="repo id (openai_compat/transformers), GGUF path (llama_cpp) or label (manual)")
    ap.add_argument("--name", help="short label used in the output filename (default: <model>_<asr label>)")
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    ap.add_argument("--out", help=f"default results/pred_{AREA}_<name>.json")
    ap.add_argument("--ids", help="comma list of call ids to run (default: every id with a transcript)")
    ap.add_argument("--overwrite", action="store_true",
                    help="redo ids even if already in the output file (with --ids only those ids are replaced and the "
                         "rest of the file is kept; without --ids the file starts fresh)")
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--json-mode", action="store_true",
                    help='send response_format={"type":"json_object"} (openai_compat / llama_cpp)')
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would run and show the first prompt; no model calls, no writes")
    s = ap.add_argument_group("transcript source (exactly one)")
    s.add_argument("--transcripts", metavar="ASR_JSON", help="results/asr_<asrname>.json written by eval/run_asr.py")
    s.add_argument("--use-reference", action="store_true",
                   help="use {data-root}/calls/<id>.ref.txt (ground-truth transcripts); asr_model='reference'")
    g = ap.add_argument_group("openai_compat")
    g.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    g.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "none"))
    g.add_argument("--timeout", type=float, default=600, help="per-request timeout in seconds")
    g.add_argument("--extra-body", help='JSON merged into the request body, e.g. \'{"chat_template_kwargs":{"enable_thinking":false}}\'')
    g = ap.add_argument_group("transformers")
    g.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    g.add_argument("--device", default="auto", help="auto | cuda | cpu | mps")
    g.add_argument("--trust-remote-code", action="store_true")
    g.add_argument("--chat-kwargs", help='JSON passed to apply_chat_template, e.g. \'{"enable_thinking": false}\'')
    g = ap.add_argument_group("llama_cpp")
    g.add_argument("--n-ctx", type=int, default=8192)
    g.add_argument("--n-gpu-layers", type=int, default=-1, help="-1 = offload all layers when the build has GPU support")
    a = ap.parse_args(argv)

    if bool(a.transcripts) == bool(a.use_reference):
        ap.error("give exactly one transcript source: --transcripts <results/asr_<name>.json> or --use-reference")
    if a.runner != "manual" and not a.model:
        ap.error(f"--model is required for runner {a.runner}")
    if not a.model:
        a.model = "manual"

    asr_label, sources = load_sources(a)
    if not a.name:
        model_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(a.model).name).strip("_") or "model"
        a.name = f"{model_label}_{re.sub(r'[^A-Za-z0-9_.-]+', '_', asr_label)}"
    out_path = Path(a.out) if a.out else ROOT / "results" / f"pred_{AREA}_{a.name}.json"
    if not sources:
        sys.exit("nothing to run (no transcripts)")
    items = [(i, src, call_summary_prompt(src["transcript"])) for i, src in sources.items()]

    existing = {}
    if out_path.exists():
        try:
            existing = json.load(open(out_path, encoding="utf-8"))
            if not isinstance(existing, dict):
                raise ValueError("top level is not a JSON object keyed by id")
        except (json.JSONDecodeError, ValueError) as e:
            if not a.overwrite:
                sys.exit(f"cannot resume: {out_path} is not valid ({e}); use --overwrite or fix the file")
            existing = {}
    if a.overwrite:
        # --overwrite redoes the SELECTED ids: with --ids the other records in the file are kept (so redoing one
        # call does not wipe the rest); without --ids the whole file starts fresh.  Same rule as run_notices.py.
        selected = {it[0] for it in items}
        existing = {k: v for k, v in existing.items() if k not in selected} if a.ids else {}
    # A record counts as done when it holds a real answer (valid JSON or not). Records still pending (manual runner,
    # no response file yet) or that ended in a backend error (server down, timeout twice...) are redone on resume.
    done = {k for k, v in existing.items() if isinstance(v, dict) and not v.get("pending") and not v.get("error")}
    todo = [it for it in items if it[0] not in done]

    print(f"area={AREA} runner={a.runner} model={a.model} name={a.name} json_mode={a.json_mode} "
          f"max_tokens={a.max_tokens} temperature={a.temperature}\n"
          f"transcripts={'reference (' + a.data_root + '/calls/*.ref.txt)' if a.use_reference else a.transcripts} "
          f"asr_label={asr_label}\nout={out_path}  items={len(items)} already_done={len(items) - len(todo)} to_run={len(todo)}",
          flush=True)

    if a.dry_run:
        for i, src, prompt in items:
            prev = existing.get(i) if isinstance(existing.get(i), dict) else None
            st = "done" if i in done else ("pending" if prev and prev.get("pending") else ("error" if prev else "todo"))
            words = len(src["transcript"].split())
            print(f"  {i:10s} {st:8s} asr_model={str(src['model'])[:40]:40s} transcript={words:5d} words "
                  f"prompt={len(prompt):6d} chars{'  EMPTY TRANSCRIPT' if not words else ''}", flush=True)
        if a.runner == "manual":
            print(f"manual prompts would go to {ROOT / 'results' / f'manual_{AREA}_{a.name}'}")
        p0 = items[0][2]
        print("\n--- prompt preview (first item) ---\n" + p0[:1200] + ("\n..." if len(p0) > 1200 else ""))
        print("\ndry run: no model calls, nothing written")
        return 0

    if not todo:
        print("nothing to do (all ids present; use --overwrite to redo)")
        return 0

    runner = make_runner(a)
    results = dict(existing)  # already filtered for --overwrite / --ids above
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out_path.with_name(out_path.name.replace(".json", "") + ".meta.json")
    meta = {"area": AREA, "args": vars(a), "asr_label": asr_label, "started": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    new = []
    try:
        for n, (item_id, src, prompt) in enumerate(todo, 1):
            rec = run_item(runner, a, item_id, src, prompt)
            results[item_id] = rec
            new.append(rec)
            write_json(out_path, results)
            status = ("ERROR " + rec["error"][:80]) if rec["error"] else ("pending" if rec["pending"] else summarize(rec["summary"]))
            lat = "   n/a " if rec["latency_sec"] is None else f"{rec['latency_sec']:7.2f}s"
            print(f"[{n}/{len(todo)}] {item_id:10s} json_valid={str(rec['json_valid']):5s} retry={str(rec['retry']):5s} "
                  f"{lat}  {status}", flush=True)
    except KeyboardInterrupt:
        print(f"\ninterrupted; {len(new)} new item(s) saved to {out_path}; rerun the same command to resume", flush=True)
        return 130

    # summary over everything now in the file (resumed + new)
    recs = [v for v in results.values() if isinstance(v, dict)]
    lat = sorted(r["latency_sec"] for r in new if r.get("latency_sec") is not None) or [0.0]
    n_valid = sum(1 for r in recs if r.get("json_valid"))
    n_schema_ok = sum(1 for r in recs if r.get("json_valid") and not r.get("schema_issues"))
    n_retry = sum(1 for r in recs if r.get("retry"))
    n_err = sum(1 for r in recs if r.get("error"))
    n_pending = sum(1 for r in recs if r.get("pending"))
    summary = {"items_in_file": len(recs), "new": len(new), "json_valid": n_valid, "schema_ok": n_schema_ok,
               "retried": n_retry, "errors": n_err, "pending": n_pending,
               "latency_mean_new": round(sum(lat) / len(lat), 3), "latency_max_new": round(lat[-1], 3),
               "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    meta["summary"] = summary
    write_json(meta_path, meta)
    print(f"\n{AREA} / {a.name}: {len(recs)} items in file ({len(new)} new) | json_valid {n_valid}/{len(recs)} | "
          f"all-keys-and-enums-ok {n_schema_ok}/{len(recs)} | retried {n_retry} | errors {n_err} | pending {n_pending}\n"
          f"latency (new items): mean {summary['latency_mean_new']:.2f}s  median {lat[len(lat) // 2]:.2f}s  max {lat[-1]:.2f}s\n"
          f"wrote {out_path}  (+ {meta_path.name})")
    if n_pending:
        print(f"answer the pending prompts in {runner.dir} (<id>.response.txt / <id>.retry.response.txt) and rerun the same command")
    print(f"next: python eval/score.py --area {AREA} --pred {out_path} --name {a.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
