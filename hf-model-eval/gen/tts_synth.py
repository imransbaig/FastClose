#!/usr/bin/env python3
"""Synthesize two-speaker phone-call audio for data/calls/callNN.script.json using Kokoro-82M (ONNX, via sherpa-onnx).

    python gen/tts_synth.py --model-dir models/kokoro-en-v0_19 [--ids call01,call02] [--threads 4]

Output: data/calls/callNN.wav  (16 kHz mono 16-bit, telephone-band-limited, light line noise)
        data/calls/callNN.audio.json (per-turn timing, voices used, duration)
Telephone effect: 24 kHz TTS -> 8 kHz -> 300-3400 Hz band-pass -> soft clipping -> noise (~27 dB SNR) -> 16 kHz.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
# kokoro-en-v0_19 speaker ids (sherpa-onnx packaging): 0 af, 1 af_bella, 2 af_nicole, 3 af_sarah, 4 af_sky,
# 5 am_adam, 6 am_michael, 7 bf_emma, 8 bf_isabella, 9 bm_george, 10 bm_lewis
VOICES = {  # call -> (CPA sid, CLIENT sid, cpa speed, client speed)
    "call01": (6, 3, 1.02, 0.98), "call02": (9, 1, 0.98, 1.05), "call03": (5, 7, 1.05, 0.97),
    "call04": (6, 10, 1.0, 1.0), "call05": (5, 4, 1.0, 1.03),
}
VOICE_NAMES = ["af", "af_bella", "af_nicole", "af_sarah", "af_sky", "am_adam", "am_michael", "bf_emma", "bf_isabella", "bm_george", "bm_lewis"]


def load_tts(model_dir, threads):
    import sherpa_onnx as so
    d = Path(model_dir)
    cfg = so.OfflineTtsConfig(
        model=so.OfflineTtsModelConfig(
            kokoro=so.OfflineTtsKokoroModelConfig(model=str(d / "model.onnx"), voices=str(d / "voices.bin"),
                                                   tokens=str(d / "tokens.txt"), data_dir=str(d / "espeak-ng-data")),
            num_threads=threads, provider="cpu"),
        max_num_sentences=1)
    return so.OfflineTts(cfg)


def telephone(x, sr_in, rng, snr_db=27.0, sr_out=16000):
    x = x.astype(np.float32)
    x8 = signal.resample_poly(x, 8000, sr_in)
    sos = signal.butter(4, [300, 3400], btype="band", fs=8000, output="sos")
    x8 = signal.sosfiltfilt(sos, x8)
    x8 = np.tanh(x8 * 1.6) / 1.6  # mild line compression
    rms = np.sqrt(np.mean(x8 ** 2)) + 1e-9
    noise = rng.standard_normal(len(x8)).astype(np.float32)
    noise = signal.sosfiltfilt(signal.butter(2, [200, 3600], btype="band", fs=8000, output="sos"), noise)
    noise *= rms / (np.sqrt(np.mean(noise ** 2)) + 1e-9) / (10 ** (snr_db / 20))
    x8 = x8 + noise
    y = signal.resample_poly(x8, sr_out, 8000)
    y = y / (np.max(np.abs(y)) + 1e-9) * 0.9
    return y.astype(np.float32)


def synth_call(tts, call_id, script, out_dir, rng):
    cpa_sid, cli_sid, cpa_speed, cli_speed = VOICES[call_id]
    sr = None
    pieces, timeline, t = [], [], 0.0
    for i, turn in enumerate(script["turns"]):
        sid, speed = (cpa_sid, cpa_speed) if turn["speaker"] == "CPA" else (cli_sid, cli_speed)
        audio = tts.generate(turn["text"], sid=sid, speed=speed)
        sr = audio.sample_rate
        samples = np.asarray(audio.samples, dtype=np.float32)
        gap = rng.uniform(0.25, 0.7)
        pieces.append(samples)
        pieces.append(np.zeros(int(gap * sr), dtype=np.float32))
        timeline.append({"turn": i, "speaker": turn["speaker"], "start": round(t, 2), "end": round(t + len(samples) / sr, 2), "voice": VOICE_NAMES[sid]})
        t += len(samples) / sr + gap
    clean = np.concatenate(pieces)
    y = telephone(clean, sr, rng)
    sf.write(out_dir / f"{call_id}.wav", y, 16000, subtype="PCM_16")
    meta = {"call_id": call_id, "duration_sec": round(len(y) / 16000, 1), "sample_rate": 16000,
            "tts": "hexgrad/Kokoro-82M v0.19 (ONNX via sherpa-onnx)", "cpa_voice": VOICE_NAMES[cpa_sid], "client_voice": VOICE_NAMES[cli_sid],
            "post": "8 kHz telephone band-pass 300-3400 Hz, soft clip, 27 dB SNR line noise, 16 kHz output", "turns": timeline}
    (out_dir / f"{call_id}.audio.json").write_text(json.dumps(meta, indent=1))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ROOT / "models/kokoro-en-v0_19"))
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    ap.add_argument("--ids", default="")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    calls_dir = Path(a.data_root) / "calls"
    ids = a.ids.split(",") if a.ids else sorted(p.name.split(".")[0] for p in calls_dir.glob("call*.script.json"))
    tts = load_tts(a.model_dir, a.threads)
    for cid in ids:
        script = json.load(open(calls_dir / f"{cid}.script.json"))
        meta = synth_call(tts, cid, script, calls_dir, np.random.default_rng(int(cid[-2:])))
        print(f"{cid}: {meta['duration_sec']} s, {len(script['turns'])} turns, voices {meta['cpa_voice']}/{meta['client_voice']}", flush=True)


if __name__ == "__main__":
    main()
