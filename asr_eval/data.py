"""FLEURS Hindi loader (parquet mirror on the HF hub) -> list of dicts with 16 kHz audio."""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent / "data" / "parquet-data" / "hi_in"


def _decode(blob):
    x, sr = sf.read(io.BytesIO(blob["bytes"]), dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if sr != 16000:
        x = resample_poly(x, 16000, sr).astype(np.float32)
    return x


def load(split, limit=None):
    df = pd.read_parquet(ROOT / f"{split}-00000-of-00001.parquet")
    if limit:
        df = df.iloc[:limit]
    return [
        dict(id=int(r.id), audio=_decode(r.audio), text=r.raw_transcription,
             gender=int(r.gender))
        for r in df.itertuples()
    ]
