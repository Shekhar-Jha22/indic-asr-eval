"""Transcribe FLEURS-hi test under each perturbation; cache hypotheses as JSONL.

python transcribe.py --model openai/whisper-small --conditions clean white_10dB ...
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from asr_eval import data, perturb

CORE = ["clean", "white_10dB", "white_0dB", "babble_10dB", "babble_0dB", "reverb_0.6s", "telephony"]
OUT = Path(__file__).parent / "results" / "hyps"


def load_model(name, adapter=None, device="mps", dtype=torch.float32):
    proc = WhisperProcessor.from_pretrained(name)
    model = WhisperForConditionalGeneration.from_pretrained(name, torch_dtype=dtype)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    return proc, model.to(device).eval()


@torch.inference_mode()
def transcribe(proc, model, wavs, bs=16, beams=1, device="mps"):
    out = []
    # Sort by length so padding waste inside a batch is minimal.
    order = np.argsort([len(w) for w in wavs])[::-1]
    for i in range(0, len(order), bs):
        out += _decode(proc, model, order[i: i + bs], wavs, beams, device)
    return [t for _, t in sorted(out)]


def _decode(proc, model, idx, wavs, beams, device):
    """Decode one batch; on OOM free the cache and split it in half (capped-memory runs)."""
    try:
        feats = proc.feature_extractor([wavs[j] for j in idx], sampling_rate=16000,
                                       return_tensors="pt").input_features
        ids = model.generate(feats.to(device, model.dtype), language="hi", task="transcribe",
                             num_beams=beams, max_new_tokens=225, use_cache=True)
        return list(zip(idx, proc.batch_decode(ids, skip_special_tokens=True)))
    except RuntimeError as e:
        if "out of memory" not in str(e) or len(idx) == 1:
            raise
        if device == "mps":
            torch.mps.empty_cache()
        h = len(idx) // 2
        return _decode(proc, model, idx[:h], wavs, beams, device) + _decode(proc, model, idx[h:], wavs, beams, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--tag", help="name under results/hyps (default: model basename)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--conditions", nargs="+", default=CORE)
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--fp16", action="store_true", help="fp16 overflows to NaN on MPS (outputs '!!!!'); CUDA only")
    a = ap.parse_args()

    tag = a.tag or a.model.split("/")[-1]
    utts = data.load(a.split)
    pool = [u["audio"] for u in data.load("validation", limit=200)]  # babble source, disjoint from test
    proc, model = load_model(a.model, a.adapter, dtype=torch.float16 if a.fp16 else torch.float32)

    for cond in a.conditions:
        path = OUT / tag / f"{a.split}_{cond}.jsonl"
        if path.exists():
            continue
        wavs = [perturb.CONDITIONS[cond](u["audio"], np.random.default_rng(u["id"]), pool) for u in utts]
        t0 = time.time()
        hyps = transcribe(proc, model, wavs, bs=a.bs, beams=a.beams)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for u, h in zip(utts, hyps):
                f.write(json.dumps(dict(id=u["id"], ref=u["text"], hyp=h, gender=u["gender"],
                                        dur=len(u["audio"]) / 16000), ensure_ascii=False) + "\n")
        print(f"{tag} {cond}: {len(utts)} utts in {time.time() - t0:.0f}s", flush=True)
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
