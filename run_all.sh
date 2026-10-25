#!/usr/bin/env bash
# Full pipeline: zero-shot baselines -> two LoRA runs -> robustness eval of each.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.45}
export PYTORCH_MPS_LOW_WATERMARK_RATIO=${PYTORCH_MPS_LOW_WATERMARK_RATIO:-0.3}
mkdir -p logs

$PY transcribe.py --model openai/whisper-small                       2>&1 | tee -a logs/eval.log
[ -d runs/lora_clean/epoch1 ] || $PY finetune.py --out runs/lora_clean ${FT_ARGS:-}           2>&1 | tee logs/lora_clean.log
[ -d runs/lora_aug/epoch1 ]   || $PY finetune.py --out runs/lora_aug --augment ${FT_ARGS:-} 2>&1 | tee logs/lora_aug.log
# One process per condition: MPS non-pool allocations grow across generate() calls.
CONDS="clean white_10dB white_0dB babble_10dB babble_0dB reverb_0.6s telephony"
for t in lora_clean lora_aug; do
  for c in $CONDS; do
    $PY transcribe.py --model openai/whisper-small --adapter runs/$t/epoch1 --tag $t --conditions $c --bs ${EVAL_BS:-4} 2>&1 | tee -a logs/eval.log
  done
done
for m in tiny base; do
  $PY transcribe.py --model openai/whisper-$m --conditions clean      2>&1 | tee -a logs/eval.log
done
$PY score.py
