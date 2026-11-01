# hf-model-eval: does an open Hugging Face model beat our current method?

Three task areas, one harness. Every model (open or the baseline) gets the **same prompt** (`eval/prompts.py`),
the **same ground truth** (`data/*/ground_truth.json`) and the **same normalization + exact-match scoring** (`eval/score.py`).
Nothing leaves your machine: models run locally (vLLM / Ollama / llama.cpp / LM Studio / transformers), which is what
IRC §7216 requires for client data.

| Area | Task | Runner(s) | Prediction file |
|---|---|---|---|
| A | field extraction from scanned tax docs (W-2, 1099s, K-1s, IRS transcripts, notices) | `eval/run_docs.py` | `results/pred_docs_<name>.json` |
| B | speech-to-text, then call-log summary | `eval/run_asr.py` -> `eval/summarize_calls.py` | `results/asr_<name>.json` -> `results/pred_calls_<name>.json` |
| C | IRS notice -> strict JSON (type, deadline, action, urgency) | `eval/run_notices.py` | `results/pred_notices_<name>.json` |

`data/` ships synthetic, redacted-format items (20 scanned docs, 5 calls with audio + reference transcripts, notices) built by `gen/`.
Drop your own redacted files in the same layout (`<id>.png` / `<id>.wav` + `<id>.ref.txt` / `<id>.txt`, plus `ground_truth.json`) to score on real work.

## 0. Install

```bash
cd hf-model-eval
python -m venv .venv && source .venv/bin/activate        # Python 3.11+ (built on 3.13)
pip install -r requirements.txt                           # Hub API, scoring, sherpa-onnx (CPU ASR), Pillow, pandas
pip install torch transformers accelerate                 # optional: --runner transformers (GPU recommended)
pip install llama-cpp-python                              # optional: --runner llama_cpp on local GGUF files
export HF_TOKEN=hf_...                                    # optional: read gated repos you have accepted
```

## 1. Network: domains to allow if you run inside a sandbox / egress filter

| Purpose | Domains |
|---|---|
| Hub API + model cards + LICENSE files (`hf_pull.py`, `transformers`, `huggingface-cli`) | `huggingface.co` |
| Model weight downloads (Hub CDN; files are redirected here) | `cas-bridge.xethub.hf.co`, `cdn-lfs.hf.co`, `cdn-lfs-us-1.hf.co` |
| pip | `pypi.org`, `files.pythonhosted.org` |
| ONNX ASR/TTS exports used by `models/download_models.sh` (sherpa-onnx releases; optional) | `github.com`, `objects.githubusercontent.com` |
| Ollama model pulls (optional) | `registry.ollama.ai` |

The eval runs themselves need **no** network once models are local.

## 2. Pull the Hub data (trending + downloads, age-normalized, license, adoption signals)

```bash
python hf_pull.py --out pulls/$(date +%F)                 # all three areas, 40 models per (sort x pipeline_tag)
python hf_pull.py --out pulls/$(date +%F) --areas A --limit 60 --skip-finetune-counts   # faster, one area
```

Writes `pulls/<date>/area_<A|B|C>_candidates.json` (raw API rows), `area_<X>_ranked.csv` (downloads, downloads/day since
created, likes, created, params, license tag + LICENSE-file verdict, gated, finetunes, Spaces, inference providers),
`summary.xlsx`, `pull_meta.json`. **Re-run in two weeks and `diff` the CSVs** to see who is actually gaining adoption.

## 3. Serve a candidate model locally (any OpenAI-compatible server works with `--runner openai_compat`)

```bash
# vLLM (GPU) - vision model for area A, or a text model for B/C
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --max-model-len 16384            # -> --base-url http://localhost:8000/v1
vllm serve Qwen/Qwen3-4B-Instruct-2507 --max-model-len 16384

# Ollama (Mac / CPU / GPU)
ollama run qwen2.5vl:7b                                                   # -> --base-url http://localhost:11434/v1 --model qwen2.5vl:7b
ollama run qwen3:4b

# llama.cpp server on a GGUF (add --mmproj mmproj.gguf for a vision model)
llama-server -m model.gguf --port 8080                                    # -> --base-url http://localhost:8080/v1 --model model

# LM Studio: load the model, Developer tab -> Start Server                # -> --base-url http://localhost:1234/v1 --model <id shown>
```

`--model` is whatever name the server exposes (vLLM: the repo id; Ollama: the tag; llama.cpp: anything).
Add `--json-mode` to send `response_format={"type":"json_object"}` (vLLM, llama.cpp, LM Studio, Ollama support it).
Alternatives without a server: `--runner transformers --model <repo id>` (downloads from the Hub, runs in-process) and,
for text areas, `--runner llama_cpp --model path/to/model.gguf`.

## 4. Run the evals

All runners share the same flags: `--runner {openai_compat,transformers,llama_cpp,manual} --model <id|path> --name <label>
[--ids doc01,doc02] [--overwrite] [--max-tokens 800] [--temperature 0] [--base-url ... --api-key ...] [--json-mode] [--dry-run]`.
Output is written after **every** item; rerun the same command to resume. `--dry-run` shows the exact prompts and calls nothing.

```bash
# A. docs: image + prompt -> JSON fields
python eval/run_docs.py --runner openai_compat --base-url http://localhost:8000/v1 \
    --model Qwen/Qwen2.5-VL-7B-Instruct --name qwen25vl-7b --json-mode

# B. stage 1: ASR  (sherpa-onnx = CPU ONNX exports; get them with: bash models/download_models.sh)
python eval/run_asr.py --runner sherpa_onnx  --model models/sherpa-onnx-whisper-large-v3            --name whisper-large-v3
python eval/run_asr.py --runner sherpa_onnx  --model models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8 --name parakeet-v3
python eval/run_asr.py --runner transformers --model openai/whisper-large-v3                        --name whisper-large-v3-hf
python eval/run_asr.py --runner openai_compat --base-url http://localhost:8000/v1 --model whisper-1 --name server-asr
#    -> results/asr_<name>.json  {"call01": {"transcript": ..., "model": ..., "audio_sec": ..., "rtf": ...}}

# B. stage 2: transcript -> call-log JSON (text model); the ASR transcript or the reference transcript
python eval/summarize_calls.py --transcripts results/asr_whisper-large-v3.json --runner openai_compat \
    --base-url http://localhost:8000/v1 --model Qwen/Qwen3-4B-Instruct-2507 --name qwen3-4b_whisper-large-v3 --json-mode
python eval/summarize_calls.py --use-reference --runner openai_compat --model Qwen/Qwen3-4B-Instruct-2507 --json-mode
#    (--use-reference = ground-truth transcripts, asr_model="reference": the summarizer's ceiling with zero ASR errors)

# C. notices: text -> JSON classification
python eval/run_notices.py --runner openai_compat --base-url http://localhost:8000/v1 \
    --model Qwen/Qwen3-4B-Instruct-2507 --name qwen3-4b --json-mode
python eval/run_notices.py --runner llama_cpp --model models/qwen3-4b-q4_k_m.gguf --name qwen3-4b-gguf --json-mode
```

## 5. Score one run (field-level exact match after normalization; WER for calls)

```bash
python eval/score.py --area docs    --pred results/pred_docs_qwen25vl-7b.json            --name qwen25vl-7b
python eval/score.py --area calls   --pred results/pred_calls_qwen3-4b_whisper-large-v3.json --name qwen3-4b_whisper-large-v3
python eval/score.py --area notices --pred results/pred_notices_qwen3-4b.json            --name qwen3-4b
#    -> results/score_<area>_<name>.json  (per item, per field, per doc type; calls also get WER per call)
```

## 6. Compile the eval sheet (every `results/score_*.json`, ranked per area)

```bash
python eval/score.py --compile        # -> results/eval_sheet.csv and results/eval_sheet.xlsx
```

Columns: area, model, items, fields_total, fields_correct, field_exact_match, wer_mean, json_invalid_items, per_type accuracy.
The baseline row (section 7) sits in the same table; a model is worth switching to only if its `field_exact_match` beats it.

## 7. Baseline row with the manual runner (your current method: a human, or Claude in the desktop app)

The manual runner writes the exact prompt each model sees and reads your answer back, so the baseline is compared fairly.

```bash
python eval/run_docs.py        --runner manual --name baseline_claude            # results/manual_docs_baseline_claude/<id>.prompt.txt (+ <id>.png)
python eval/summarize_calls.py --runner manual --name baseline_claude --use-reference
python eval/run_notices.py     --runner manual --name baseline_claude
# 1. open each <id>.prompt.txt (docs: first line is the image path - attach that image), get the answer
# 2. save the answer as <id>.response.txt in the same folder (an invalid-JSON answer gets <id>.retry.prompt.txt -> <id>.retry.response.txt)
# 3. rerun the SAME command: answered ids are collected, unanswered stay "pending"
python eval/score.py --area docs --pred results/pred_docs_baseline_claude.json --name baseline_claude   # then --compile
```

Use the same recipe for any hosted model you want in the table (paste prompts into its UI), or for a second human rater.

## Layout

```
hf_pull.py               Hub API pull (areas A/B/C, trending + downloads, license + adoption signals)
eval/fields.json         field schema per doc type, call-log / notice enums, normalization rules
eval/prompts.py          doc_prompt / notice_prompt / call_summary_prompt / parse_json_object  (shared by every runner)
eval/run_docs.py         area A runner        eval/run_asr.py + eval/summarize_calls.py   area B      eval/run_notices.py   area C
eval/score.py            scoring (--area ...) and the compiled sheet (--compile)
data/{docs,calls,notices}/   items + ground_truth.json          gen/   generators for the synthetic set
models/                  local ASR/TTS ONNX models (download_models.sh)        results/   asr_*, pred_*, score_*, eval_sheet.*
pulls/<date>/            dated Hub pulls - rerun and diff
```

## Rerun checklist (every two weeks)

```bash
python hf_pull.py --out pulls/$(date +%F) && diff pulls/<old>/area_A_ranked.csv pulls/$(date +%F)/area_A_ranked.csv
# for each new top-2 candidate: serve it (3), run its area (4), score (5); then
python eval/score.py --compile
```
