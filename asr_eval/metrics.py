"""Corpus WER/CER with utterance-level bootstrap confidence intervals."""
from dataclasses import dataclass

import jiwer
import numpy as np


def _edit_counts(refs, hyps, unit):
    """Per-utterance (errors, ref_len) so corpus metrics can be resampled."""
    errs, lens = np.zeros(len(refs)), np.zeros(len(refs))
    for i, (r, h) in enumerate(zip(refs, hyps)):
        if unit == "char":
            r, h = " ".join(r.replace(" ", "▁")), " ".join(h.replace(" ", "▁"))
        rt = r.split()
        if not rt:
            continue
        o = jiwer.process_words(r, h if h.strip() else "<empty>")
        errs[i] = o.substitutions + o.deletions + o.insertions
        lens[i] = len(rt)
    return errs, lens


@dataclass
class Score:
    value: float
    lo: float
    hi: float

    def __str__(self):
        return f"{100 * self.value:.2f} [{100 * self.lo:.2f}, {100 * self.hi:.2f}]"


def corpus_rate(refs, hyps, unit="word", n_boot=1000, seed=0) -> Score:
    errs, lens = _edit_counts(refs, hyps, unit)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(refs), size=(n_boot, len(refs)))
    boot = errs[idx].sum(1) / lens[idx].sum(1)
    return Score(errs.sum() / lens.sum(), *np.percentile(boot, [2.5, 97.5]))


def paired_bootstrap(refs, hyps_a, hyps_b, unit="word", n_boot=2000, seed=0):
    """P(system B is not better than A) under utterance resampling."""
    ea, la = _edit_counts(refs, hyps_a, unit)
    eb, _ = _edit_counts(refs, hyps_b, unit)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(refs), size=(n_boot, len(refs)))
    delta = (eb[idx].sum(1) - ea[idx].sum(1)) / la[idx].sum(1)
    return float((delta >= 0).mean()), float((eb.sum() - ea.sum()) / la.sum())
