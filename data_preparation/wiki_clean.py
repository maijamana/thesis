"""Wikipedia markup and encoding cleanup (thesis: -LRB- tokens, etc.)."""

from __future__ import annotations

import html
import re
import unicodedata

import pandas as pd

BRACKET_REPLACEMENTS = [
    ("-LRB-", "("),
    ("-RRB-", ")"),
    ("-LSB-", "["),
    ("-RSB-", "]"),
    ("-LCB-", "{"),
    ("-RCB-", "}"),
]

MOJIBAKE_REPLACEMENTS = [
    ("â\x80\x93", "\u2013"),
    ("â\x80\x94", "\u2014"),
    ("â\x80\x99", "'"),
    ("â\x80\x9c", "\u201c"),
    ("â\x80\x9d", "\u201d"),
    ("â\x80\x98", "'"),
    ("Ã©", "é"),
    ("Ã¨", "è"),
    ("Ã ", "à"),
    ("Ã¡", "á"),
    ("Ã±", "ñ"),
]


def clean_text(s: str) -> str:
    if not isinstance(s, str) or pd.isna(s):
        return s
    s = html.unescape(s)
    for token, repl in BRACKET_REPLACEMENTS:
        s = s.replace(token, repl)
    for bad, good in MOJIBAKE_REPLACEMENTS:
        s = s.replace(bad, good)
    s = s.replace("``", '"').replace("''", '"')
    for punct in ",.;:?!":
        s = re.sub(r"\s+" + re.escape(punct), punct, s)
    return unicodedata.normalize("NFC", s)


def clean_pairs_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("original", "simplified"):
        if col in out.columns:
            out[col] = out[col].astype(str).apply(clean_text)
    return out
