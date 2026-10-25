import numpy as np

from asr_eval import metrics, normalize, perturb


def test_snr_is_exact():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(16000).astype(np.float32)
    for snr in (0, 10, 20):
        n = perturb.white(x, snr, np.random.default_rng(1)) - x
        assert abs(10 * np.log10(np.mean(x ** 2) / np.mean(n ** 2)) - snr) < 1e-3


def test_perturbations_preserve_length_and_are_deterministic():
    x = np.random.default_rng(0).standard_normal(32000).astype(np.float32)
    pool = [np.random.default_rng(i).standard_normal(20000) for i in range(6)]
    for name, f in perturb.CONDITIONS.items():
        a, b = f(x, np.random.default_rng(7), pool), f(x, np.random.default_rng(7), pool)
        assert np.array_equal(a, b), name
        if not name.startswith("speed"):
            assert len(a) == len(x), name


def test_whisper_basic_destroys_matras():
    # The bug this repo measures: vowel signs become spaces, so one word is scored
    # as several consonant fragments and a wrong matra costs nothing.
    assert normalize.whisper_basic("किताब") == "क त ब"
    assert normalize.whisper_basic("कीताब") == normalize.whisper_basic("किताब")
    assert normalize.indic("कीताब") != normalize.indic("किताब")


def test_indic_normaliser_equivalences():
    assert normalize.indic("ज़मीन") == normalize.indic("ज़मीन")  # precomposed vs decomposed nukta
    assert normalize.indic("२०२४ में।") == "2024 में"
    assert normalize.NORMALISERS["indic_lenient"]("ज़") == normalize.NORMALISERS["indic_lenient"]("ज")


def test_wer_and_ci():
    refs = ["a b c d"] * 50
    hyps = ["a b c x"] * 50
    s = metrics.corpus_rate(refs, hyps)
    assert abs(s.value - 0.25) < 1e-9 and s.lo <= 0.25 <= s.hi
    p, d = metrics.paired_bootstrap(refs, hyps, refs)
    assert d < 0 and p == 0.0
