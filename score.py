"""Score cached hypotheses -> results/scores.json, results/REPORT.md, results/robustness.png."""
import json
import re
from pathlib import Path

import numpy as np

from asr_eval import metrics, normalize

R = Path(__file__).parent / "results"
CONDS = ["clean", "white_10dB", "white_0dB", "babble_10dB", "babble_0dB", "reverb_0.6s", "telephony"]
LATIN = re.compile(r"[A-Za-z]")
DEVA = re.compile(r"[ऀ-ॿ]")


def load(tag, cond):
    p = R / "hyps" / tag / f"test_{cond}.jsonl"
    return [json.loads(l) for l in p.open()] if p.exists() else None


def diagnostics(rows):
    """Failure modes that WER alone hides."""
    off_script = np.mean([len(LATIN.findall(r["hyp"])) > len(DEVA.findall(r["hyp"])) for r in rows])
    ratio = np.array([len(normalize.indic(r["hyp"]).split()) / max(1, len(normalize.indic(r["ref"]).split())) for r in rows])
    return dict(off_script=float(off_script), hallucination=float(np.mean(ratio > 1.5)),
                truncation=float(np.mean(ratio < 0.5)))


def score(rows, norm="indic"):
    f = normalize.NORMALISERS[norm]
    refs, hyps = [f(r["ref"]) for r in rows], [f(r["hyp"]) for r in rows]
    return metrics.corpus_rate(refs, hyps, "word"), metrics.corpus_rate(refs, hyps, "char")


def main():
    tags = sorted(p.name for p in (R / "hyps").iterdir())
    out, md = {}, ["# Results (FLEURS hi_in test, 418 utts)\n",
                   "WER/CER in %, 95% bootstrap CI over utterances; `indic` normaliser.\n"]

    md += ["## Model x condition (WER)\n", "| model | " + " | ".join(CONDS) + " |",
           "|---" * (len(CONDS) + 1) + "|"]
    for t in tags:
        out[t] = {}
        cells = []
        for c in CONDS:
            rows = load(t, c)
            if rows is None:
                cells.append("–")
                continue
            w, ch = score(rows)
            out[t][c] = dict(wer=w.value, wer_ci=[w.lo, w.hi], cer=ch.value, cer_ci=[ch.lo, ch.hi], **diagnostics(rows))
            cells.append(f"{100 * w.value:.1f}")
        md.append(f"| {t} | " + " | ".join(cells) + " |")

    md += ["\n## Clean-set detail\n", "| model | WER | CER | off-script | halluc. | trunc. |", "|---|---|---|---|---|---|"]
    for t in tags:
        if "clean" in out[t]:
            s = out[t]["clean"]
            md.append(f"| {t} | {100 * s['wer']:.1f} [{100 * s['wer_ci'][0]:.1f}, {100 * s['wer_ci'][1]:.1f}] | "
                      f"{100 * s['cer']:.1f} | {100 * s['off_script']:.1f}% | {100 * s['hallucination']:.1f}% | {100 * s['truncation']:.1f}% |")

    # Normaliser sensitivity: same hypotheses, different scoring conventions.
    md += ["\n## Normaliser sensitivity (clean)\n", "| model | " + " | ".join(normalize.NORMALISERS) + " |",
           "|---" * (len(normalize.NORMALISERS) + 1) + "|"]
    out["_normaliser"] = {}
    for t in tags:
        rows = load(t, "clean")
        if rows is None:
            continue
        ws = {n: score(rows, n)[0].value for n in normalize.NORMALISERS}
        out["_normaliser"][t] = ws
        md.append(f"| {t} | " + " | ".join(f"{100 * v:.1f}" for v in ws.values()) + " |")

    # Paired significance: augmented vs clean fine-tuning, and each vs zero-shot.
    pairs = [("whisper-small", "lora_clean"), ("whisper-small", "lora_aug"), ("lora_clean", "lora_aug")]
    md += ["\n## Paired bootstrap (ΔWER = B − A, p = P(B not better))\n", "| A → B | condition | ΔWER | p |", "|---|---|---|---|"]
    out["_paired"] = {}
    for a, b in pairs:
        for c in CONDS:
            ra, rb = load(a, c), load(b, c)
            if ra is None or rb is None:
                continue
            f = normalize.indic
            p, d = metrics.paired_bootstrap([f(r["ref"]) for r in ra], [f(r["hyp"]) for r in ra], [f(r["hyp"]) for r in rb])
            out["_paired"][f"{a}->{b}:{c}"] = dict(delta=d, p=p)
            md.append(f"| {a} → {b} | {c} | {100 * d:+.1f} | {p:.3f} |")

    (R / "scores.json").write_text(json.dumps(out, indent=1))
    (R / "REPORT.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    plot(out, tags)


def plot(out, tags):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    shown = [t for t in tags if len(out[t]) == len(CONDS)]
    if not shown:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(CONDS))
    w = 0.8 / len(shown)
    for i, t in enumerate(shown):
        v = [100 * out[t][c]["wer"] for c in CONDS]
        e = np.array([[100 * (out[t][c]["wer"] - out[t][c]["wer_ci"][0]), 100 * (out[t][c]["wer_ci"][1] - out[t][c]["wer"])] for c in CONDS]).T
        ax.bar(x + (i - (len(shown) - 1) / 2) * w, v, w, yerr=e, capsize=2, label=t)
    ax.set_xticks(x, CONDS, rotation=20)
    ax.set_ylabel("WER (%)")
    ax.set_title("Whisper-small on FLEURS-hi under synthetic corruption")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(R / "robustness.png", dpi=150)


if __name__ == "__main__":
    main()
