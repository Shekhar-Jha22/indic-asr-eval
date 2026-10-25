"""Text normalisation for Indic ASR scoring.

Whisper's BasicTextNormalizer replaces every Unicode mark (category M*) with a
space. For Devanagari that removes matras, virama and nukta and splits each word
into consonant fragments: "किताब" -> "क त ब", and "कीताब" scores as identical. `whisper_basic` reproduces that behaviour so its effect on
reported WER can be measured; `indic` is the normaliser we actually score with.
"""
import re
import unicodedata

_PUNCT = re.compile(r"[।॥॰\"'“”‘’`.,!?;:()\[\]{}<>«»…—–\-_/\\|@#$%^&*+=~]")
_WS = re.compile(r"\s+")

# Precomposed nukta letters -> base + nukta, so both spellings compare equal.
_NUKTA_DECOMP = {
    "ऩ": "ऩ", "ऱ": "ऱ", "ऴ": "ऴ",
    "क़": "क़", "ख़": "ख़", "ग़": "ग़",
    "ज़": "ज़", "ड़": "ड़", "ढ़": "ढ़",
    "फ़": "फ़", "य़": "य़",
}
_NUKTA_RE = re.compile("|".join(_NUKTA_DECOMP))
_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_ZW = re.compile("[​‌‍﻿]")


def indic(text: str, strip_nukta: bool = False, chandrabindu_to_anusvara: bool = False) -> str:
    t = unicodedata.normalize("NFC", text)
    t = _NUKTA_RE.sub(lambda m: _NUKTA_DECOMP[m.group(0)], t)
    t = _ZW.sub("", t).translate(_DEV_DIGITS).lower()
    if strip_nukta:
        t = t.replace("़", "")
    if chandrabindu_to_anusvara:
        t = t.replace("ँ", "ं")
    t = _PUNCT.sub(" ", t)
    return _WS.sub(" ", t).strip()


def whisper_basic(text: str) -> str:
    """Behaviour of transformers' BasicTextNormalizer (remove_diacritics=False)."""
    t = re.sub(r"[<\[][^>\]]*[>\]]", "", text.lower())
    t = re.sub(r"\(([^)]+?)\)", "", t)
    t = "".join(" " if unicodedata.category(c)[0] in "MSP" else c
                for c in unicodedata.normalize("NFKC", t))
    return _WS.sub(" ", t).strip()


def raw(text: str) -> str:
    return _WS.sub(" ", text).strip()


NORMALISERS = {
    "raw": raw,
    "whisper_basic": whisper_basic,
    "indic": indic,
    "indic_lenient": lambda s: indic(s, strip_nukta=True, chandrabindu_to_anusvara=True),
}
