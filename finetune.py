"""LoRA fine-tuning of Whisper on FLEURS-hi train, optionally with on-the-fly
synthetic corruption (asr_eval.perturb.random_augment).

python finetune.py --out runs/lora_clean
python finetune.py --out runs/lora_aug --augment
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from asr_eval import data, metrics, normalize, perturb
from transcribe import transcribe


def collate(proc, batch, rng, pool, augment):
    wavs = [u["audio"].astype(np.float32) for u in batch]
    wavs = [perturb.random_augment(w, rng, pool) if augment else w for w in wavs]
    feats = proc.feature_extractor(wavs, sampling_rate=16000, return_tensors="pt").input_features
    tok = proc.tokenizer([u["text"] for u in batch], padding=True, return_tensors="pt")
    labels = tok.input_ids.masked_fill(tok.attention_mask == 0, -100)
    # The model prepends decoder_start_token itself when shifting labels.
    if (labels[:, 0] == proc.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")).all():
        labels = labels[:, 1:]
    return feats, labels


def evaluate(proc, model, utts, device):
    model.eval()
    hyps = transcribe(proc, model, [u["audio"] for u in utts], bs=4, device=device)
    refs = [normalize.indic(u["text"]) for u in utts]
    hyps = [normalize.indic(h) for h in hyps]
    model.train()
    if device == "mps":
        torch.mps.empty_cache()
    return metrics.corpus_rate(refs, hyps, n_boot=200).value, metrics.corpus_rate(refs, hyps, "char", n_boot=200).value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-small")
    ap.add_argument("--out", required=True)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--grad_ckpt", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--enc_layers", type=int, default=4, help="top-k encoder layers to adapt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_n", type=int, default=120, help="validation utts decoded per epoch")
    a = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    proc = WhisperProcessor.from_pretrained(a.model, language="hi", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(a.model)
    model.generation_config.forced_decoder_ids = None
    if a.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # Adapt the full decoder and only the top encoder layers: autograd then keeps
    # activations for 4/12 encoder blocks, which is what fits on a 16 GB laptop GPU.
    proj = "(q_proj|k_proj|v_proj|out_proj|fc1|fc2)"
    top = "|".join(str(i) for i in range(model.config.encoder_layers - a.enc_layers, model.config.encoder_layers))
    model = get_peft_model(model, LoraConfig(
        r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
        target_modules=rf".*decoder\.layers\.\d+\..*{proj}|.*encoder\.layers\.({top})\..*{proj}"))
    model.print_trainable_parameters()
    model.to(device).train()

    train = [u for u in data.load("train") if len(u["audio"]) <= 30 * 16000]
    for u in train:  # halve host RAM; upcast per batch in collate
        u["audio"] = u["audio"].astype(np.float16)
    val = data.load("validation", limit=a.val_n)
    pool = [u["audio"].astype(np.float32) for u in train[:100]]  # babble source drawn from train only
    print(f"train={len(train)} val={len(val)}", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.01)
    steps = a.epochs * math.ceil(len(train) / (a.bs * a.accum))
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, steps - warm))))))

    log, step, t0 = [], 0, time.time()
    for ep in range(a.epochs):
        order = rng.permutation(len(train))
        running = []
        for i in range(0, len(order), a.bs):
            feats, labels = collate(proc, [train[j] for j in order[i: i + a.bs]], rng, pool, a.augment)
            loss = model(input_features=feats.to(device), labels=labels.to(device)).loss
            (loss / a.accum).backward()
            running.append(loss.item())
            if (i // a.bs + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(), sched.step(), opt.zero_grad(set_to_none=True)
                step += 1
                if step % 25 == 0:
                    print(f"ep{ep} step {step}/{steps} loss {np.mean(running[-50:]):.3f} "
                          f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
        wer, cer = evaluate(proc, model, val, device)
        log.append(dict(epoch=ep, step=step, train_loss=float(np.mean(running)), val_wer=wer, val_cer=cer))
        print(json.dumps(log[-1]), flush=True)
        model.save_pretrained(out / f"epoch{ep}")
    (out / "log.json").write_text(json.dumps(dict(args=vars(a), log=log), indent=1))


if __name__ == "__main__":
    main()
