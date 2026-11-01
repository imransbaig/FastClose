#!/bin/bash
# Transcribe every call with every downloaded sherpa-onnx model, then score. Run from hf-model-eval/.
set -u
cd "$(dirname "$0")/.."
run() { name=$1; dir=$2; shift 2; echo "=== $name ==="; python3 eval/run_asr.py --runner sherpa_onnx --model "$dir" --name "$name" "$@" || echo "FAILED $name"; }
run whisper-large-v3           models/sherpa-onnx-whisper-large-v3
run whisper-large-v3-turbo     models/sherpa-onnx-whisper-turbo
run parakeet-tdt-0.6b-v2       models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8
run parakeet-tdt-0.6b-v3       models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8
run moonshine-base             models/sherpa-onnx-moonshine-base-en-int8
run canary-180m-flash          models/sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8
run distil-whisper-large-v2    models/sherpa-onnx-whisper-distil-large-v2
echo ALL_ASR_DONE
