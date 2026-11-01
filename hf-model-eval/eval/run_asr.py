#!/usr/bin/env python3
"""Area B stage 1: transcribe data/calls/*.wav with an open ASR model.

    python eval/run_asr.py --runner sherpa_onnx --model models/sherpa-onnx-whisper-large-v3 --name whisper-large-v3
    python eval/run_asr.py --runner transformers --model openai/whisper-large-v3 --name whisper-large-v3-hf
    python eval/run_asr.py --runner openai_compat --base-url http://localhost:8000/v1 --model whisper-1 --name server

Output results/asr_<name>.json: {"call01": {"transcript": "...", "model": "...", "runner": "...", "audio_sec": .., "latency_sec": .., "rtf": .., "segments": n}}
sherpa_onnx runner: auto-detects whisper / parakeet (nemo transducer) / canary / moonshine from the model directory,
and splits the call with Silero VAD into <= 20 s speech segments (Whisper ONNX exports only accept 30 s windows).
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def load_audio(path, sr=16000):
    x, s = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if s != sr:
        from scipy import signal
        x = signal.resample_poly(x, sr, s).astype(np.float32)
    return x, sr


# ---------------- sherpa-onnx ----------------
def _first(d, *pats):
    for p in pats:
        m = sorted(glob.glob(str(Path(d) / p)))
        if m:
            return m[0]
    return None


def sherpa_recognizer(model_dir, threads):
    import sherpa_onnx as so
    d = Path(model_dir)
    tokens = _first(d, "*tokens.txt")
    if _first(d, "preprocess.onnx"):
        kind = "moonshine"
        rec = so.OfflineRecognizer.from_moonshine(preprocessor=_first(d, "preprocess.onnx"), encoder=_first(d, "encode.int8.onnx", "encode.onnx"),
                                                  uncached_decoder=_first(d, "uncached_decode.int8.onnx", "uncached_decode.onnx"),
                                                  cached_decoder=_first(d, "cached_decode.int8.onnx", "cached_decode.onnx"), tokens=tokens, num_threads=threads)
    elif _first(d, "*joiner*.onnx"):
        kind = "nemo_transducer"
        rec = so.OfflineRecognizer.from_transducer(encoder=_first(d, "encoder.int8.onnx", "encoder.onnx"), decoder=_first(d, "decoder.int8.onnx", "decoder.onnx"),
                                                   joiner=_first(d, "joiner.int8.onnx", "joiner.onnx"), tokens=tokens, num_threads=threads, model_type="nemo_transducer")
    elif "canary" in d.name:
        kind = "canary"
        rec = so.OfflineRecognizer.from_nemo_canary(encoder=_first(d, "encoder.int8.onnx", "encoder.onnx"), decoder=_first(d, "decoder.int8.onnx", "decoder.onnx"),
                                                    tokens=tokens, num_threads=threads, src_lang="en", tgt_lang="en", use_pnc=True)
    else:
        kind = "whisper"
        rec = so.OfflineRecognizer.from_whisper(encoder=_first(d, "*encoder.int8.onnx", "*encoder.onnx"), decoder=_first(d, "*decoder.int8.onnx", "*decoder.onnx"),
                                                tokens=tokens, language="en", task="transcribe", num_threads=threads)
    return rec, kind


def vad_segments(x, sr, vad_model, max_speech=20.0, threads=2):
    import sherpa_onnx as so
    cfg = so.VadModelConfig(silero_vad=so.SileroVadModelConfig(model=vad_model, threshold=0.5, min_silence_duration=0.4,
                                                                 min_speech_duration=0.25, max_speech_duration=max_speech),
                            sample_rate=sr, num_threads=threads)
    vad = so.VoiceActivityDetector(cfg, buffer_size_in_seconds=120)
    win = cfg.silero_vad.window_size
    segs = []
    for i in range(0, len(x), win):
        chunk = x[i:i + win]
        if len(chunk) < win:
            chunk = np.pad(chunk, (0, win - len(chunk)))
        vad.accept_waveform(chunk)
        while not vad.empty():
            s = vad.front
            segs.append((s.start / sr, np.asarray(s.samples, dtype=np.float32)))
            vad.pop()
    vad.flush()
    while not vad.empty():
        s = vad.front
        segs.append((s.start / sr, np.asarray(s.samples, dtype=np.float32)))
        vad.pop()
    return segs


def pack_segments(segs, sr, max_chunk=28.0, gap=0.3):
    """Greedily concatenate consecutive VAD segments (with a short silence between) into chunks <= max_chunk s.
    Whisper ONNX exports pad every input to 30 s, so decoding many tiny segments wastes a full encoder pass each."""
    chunks, cur, cur_len, cur_start = [], [], 0.0, None
    sil = np.zeros(int(gap * sr), dtype=np.float32)
    for start, seg in segs:
        d = len(seg) / sr
        if cur and cur_len + gap + d > max_chunk:
            chunks.append((cur_start, np.concatenate(cur)))
            cur, cur_len, cur_start = [], 0.0, None
        if cur_start is None:
            cur_start = start
        if cur:
            cur.append(sil)
            cur_len += gap
        if d > max_chunk:  # single over-long segment: hard split
            n = int(np.ceil(d / max_chunk))
            for part in np.array_split(seg, n):
                chunks.append((start, part))
            cur, cur_len, cur_start = [], 0.0, None
            continue
        cur.append(seg)
        cur_len += d
    if cur:
        chunks.append((cur_start, np.concatenate(cur)))
    return chunks


def run_sherpa(files, a):
    rec, kind = sherpa_recognizer(a.model, a.threads)
    out = {}
    for cid, path in files:
        x, sr = load_audio(path)
        t0 = time.time()
        if a.no_vad:
            segs = [(0.0, x)]
        else:
            segs = pack_segments(vad_segments(x, sr, a.vad_model, a.max_segment), sr, a.max_chunk)
        texts = []
        for start, seg in segs:
            st = rec.create_stream()
            st.accept_waveform(sr, seg)
            rec.decode_stream(st)
            texts.append(st.result.text.strip())
        lat = time.time() - t0
        out[cid] = {"transcript": " ".join(t for t in texts if t), "model": a.model, "runner": f"sherpa_onnx:{kind}", "audio_sec": round(len(x) / sr, 1),
                    "latency_sec": round(lat, 1), "rtf": round(lat / (len(x) / sr), 3), "segments": len(segs), "vad": not a.no_vad}
        yield cid, out[cid]


# ---------------- transformers ----------------
def run_transformers(files, a):
    import torch
    from transformers import pipeline
    device = 0 if torch.cuda.is_available() else -1
    asr = pipeline("automatic-speech-recognition", model=a.model, device=device, torch_dtype=torch.float16 if device == 0 else torch.float32)
    for cid, path in files:
        x, sr = load_audio(path)
        t0 = time.time()
        kw = {"chunk_length_s": 30, "batch_size": 8} if "whisper" in a.model.lower() else {}
        res = asr({"raw": x, "sampling_rate": sr}, return_timestamps="whisper" in a.model.lower(), generate_kwargs={"language": "en", "task": "transcribe"} if "whisper" in a.model.lower() else None, **kw)
        lat = time.time() - t0
        yield cid, {"transcript": res["text"].strip(), "model": a.model, "runner": "transformers", "audio_sec": round(len(x) / sr, 1), "latency_sec": round(lat, 1), "rtf": round(lat / (len(x) / sr), 3), "segments": None}


# ---------------- OpenAI-compatible /audio/transcriptions ----------------
def run_openai(files, a):
    import requests
    for cid, path in files:
        x, sr = load_audio(path)
        t0 = time.time()
        with open(path, "rb") as f:
            r = requests.post(f"{a.base_url.rstrip('/')}/audio/transcriptions", headers={"Authorization": f"Bearer {a.api_key}"},
                              files={"file": (Path(path).name, f, "audio/wav")}, data={"model": a.model, "language": "en", "response_format": "json"}, timeout=1800)
        r.raise_for_status()
        lat = time.time() - t0
        yield cid, {"transcript": r.json().get("text", "").strip(), "model": a.model, "runner": "openai_compat", "audio_sec": round(len(x) / sr, 1), "latency_sec": round(lat, 1), "rtf": round(lat / (len(x) / sr), 3), "segments": None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runner", choices=["sherpa_onnx", "transformers", "openai_compat"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--ids", default="")
    ap.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--vad-model", default=str(ROOT / "models/silero_vad.onnx"))
    ap.add_argument("--max-segment", type=float, default=20.0, help="VAD max speech segment (s)")
    ap.add_argument("--max-chunk", type=float, default=28.0, help="pack VAD segments into chunks up to this many seconds")
    ap.add_argument("--no-vad", action="store_true")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "none"))
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    calls = Path(a.data_root) / "calls"
    ids = a.ids.split(",") if a.ids else sorted(p.stem for p in calls.glob("call*.wav"))
    outp = Path(a.out) if a.out else ROOT / "results" / f"asr_{a.name}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    out = json.load(open(outp)) if outp.exists() and not a.overwrite else {}
    files = [(cid, str(calls / f"{cid}.wav")) for cid in ids if cid not in out]
    runner = {"sherpa_onnx": run_sherpa, "transformers": run_transformers, "openai_compat": run_openai}[a.runner]
    for cid, rec in runner(files, a):
        out[cid] = rec
        outp.write_text(json.dumps(out, indent=1))
        print(f"{cid}: {rec['audio_sec']}s audio in {rec['latency_sec']}s (RTF {rec['rtf']}), {len(rec['transcript'].split())} words", flush=True)
    print("wrote", outp)


if __name__ == "__main__":
    main()
