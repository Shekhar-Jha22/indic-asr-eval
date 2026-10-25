"""Synthetic acoustic perturbations for robustness benchmarks and augmentation.

Everything operates on float32 mono numpy arrays at 16 kHz and is seeded, so a
perturbed test set is reproducible bit-for-bit from (utterance id, condition).
"""
import numpy as np
from scipy.signal import butter, fftconvolve, resample_poly, sosfilt

SR = 16000


def _rms(x):
    return np.sqrt(np.mean(x ** 2) + 1e-12)


def add_noise(x, noise, snr_db):
    if len(noise) < len(x):
        noise = np.tile(noise, int(np.ceil(len(x) / len(noise))))
    noise = noise[: len(x)]
    scale = _rms(x) / (_rms(noise) * 10 ** (snr_db / 20))
    return (x + scale * noise).astype(np.float32)


def white(x, snr_db, rng):
    return add_noise(x, rng.standard_normal(len(x)), snr_db)


def babble(x, snr_db, rng, pool, n_talkers=4):
    """Sum of n other utterances from `pool` as competing speech."""
    picks = rng.choice(len(pool), n_talkers, replace=False)
    b = np.zeros(len(x))
    for k in picks:
        s = pool[k]
        off = rng.integers(0, max(1, len(s) - len(x))) if len(s) > len(x) else 0
        s = np.tile(s, int(np.ceil(len(x) / len(s))) + 1)[off: off + len(x)]
        b += s / _rms(s)
    return add_noise(x, b, snr_db)


def synthetic_rir(rt60, rng, sr=SR):
    """Exponentially decaying noise tail; decay set so energy drops 60 dB at rt60."""
    n = int(rt60 * sr)
    t = np.arange(n) / sr
    h = rng.standard_normal(n) * np.exp(-6.9078 * t / rt60)
    h[0] = 1.0
    return h / np.sqrt(np.sum(h ** 2))


def reverb(x, rt60, rng):
    y = fftconvolve(x, synthetic_rir(rt60, rng))[: len(x)]
    return (y * _rms(x) / _rms(y)).astype(np.float32)


_TEL_SOS = butter(6, [300, 3400], btype="band", fs=8000, output="sos")


def telephony(x, rng=None):
    """8 kHz narrowband + 300-3400 Hz channel + 8-bit mu-law, back to 16 kHz."""
    y = sosfilt(_TEL_SOS, resample_poly(x, 1, 2))
    y = np.clip(y / (np.abs(y).max() + 1e-9), -1, 1)
    mu = 255.0
    q = np.round((np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu) + 1) * 127.5)
    y = np.sign(q / 127.5 - 1) * ((1 + mu) ** np.abs(q / 127.5 - 1) - 1) / mu
    return resample_poly(y, 2, 1).astype(np.float32)


def speed(x, factor, rng=None):
    """Kaldi-style speed perturbation (changes tempo and pitch)."""
    up, down = int(round(100 / factor)), 100
    return resample_poly(x, up, down).astype(np.float32)


CONDITIONS = {
    "clean": lambda x, rng, pool: x,
    "white_20dB": lambda x, rng, pool: white(x, 20, rng),
    "white_10dB": lambda x, rng, pool: white(x, 10, rng),
    "white_5dB": lambda x, rng, pool: white(x, 5, rng),
    "white_0dB": lambda x, rng, pool: white(x, 0, rng),
    "babble_10dB": lambda x, rng, pool: babble(x, 10, rng, pool),
    "babble_5dB": lambda x, rng, pool: babble(x, 5, rng, pool),
    "babble_0dB": lambda x, rng, pool: babble(x, 0, rng, pool),
    "reverb_0.6s": lambda x, rng, pool: reverb(x, 0.6, rng),
    "reverb_1.2s": lambda x, rng, pool: reverb(x, 1.2, rng),
    "telephony": lambda x, rng, pool: telephony(x),
    "speed_0.9": lambda x, rng, pool: speed(x, 0.9),
    "speed_1.1": lambda x, rng, pool: speed(x, 1.1),
}


def random_augment(x, rng, pool):
    """Training-time augmentation: one random corruption with p=0.8."""
    r = rng.random()
    if r < 0.2:
        return x
    if r < 0.4:
        return white(x, rng.uniform(0, 20), rng)
    if r < 0.6:
        return babble(x, rng.uniform(0, 15), rng, pool, n_talkers=int(rng.integers(2, 6)))
    if r < 0.75:
        return reverb(x, rng.uniform(0.3, 1.0), rng)
    if r < 0.85:
        return telephony(x)
    return speed(x, rng.choice([0.9, 1.1]))
