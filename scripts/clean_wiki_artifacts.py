#!/usr/bin/env python3
"""
Пайплайн очищення Wikipedia-артефактів у CSV (формат aligned_pairs_meaning80_*.csv).

Порядок: HTML unescape → Bracket tokens → Mojibake → Нормалізація лапок → Пробіли → NFC.
"""

from __future__ import annotations

import argparse
import html
import re
import unicodedata
from pathlib import Path
from typing import List

import pandas as pd


# 1. Bracket placeholder tokens (exact string match)
BRACKET_REPLACEMENTS = [
    ("-LRB-", "("),
    ("-RRB-", ")"),
    ("-LSB-", "["),
    ("-RSB-", "]"),
    ("-LCB-", "{"),
    ("-RCB-", "}"),
]

# 3. Common mojibake / encoding corruption (replace with correct Unicode)
# Order: longer sequences first where applicable
MOJIBAKE_REPLACEMENTS = [
    ("â\x80\x93", "\u2013"),   # en-dash
    ("â\x80\x94", "\u2014"),   # em-dash
    ("â\x80\x99", "'"),        # right single quote
    ("â\x80\x9c", "\u201c"),   # left double quote
    ("â\x80\x9d", "\u201d"),   # right double quote
    ("â\x80\x98", "'"),        # left single quote
    ("â", "\u2013"),
    ("â", "\u2014"),
    ("â", "'"),
    ("â", "\u201c"),
    ("â", "\u201d"),
    ("â", "'"),
    ("â ``", " - "),            # malformed hyphen "1910 â `` 11"
    ("Ã©", "é"),
    ("Ã¨", "è"),
    ("Ã ", "à"),
    ("Ã¡", "á"),
    ("Ã±", "ñ"),
    ("Ã­", "í"),
    ("Ã³", "ó"),
    ("Ãº", "ú"),
    ("Ã¶", "ö"),
    ("Ã¼", "ü"),
    ("Ã§", "ç"),
    ("Å", "ō"),
    ("Å ", "ō"),
]

# Columns to clean (must exist in CSV)
TEXT_COLUMNS = ["original", "simplified", "topic"]


def html_unescape(s: str) -> str:
    """Decode HTML entities."""
    if not isinstance(s, str) or pd.isna(s):
        return s
    return html.unescape(s)


def replace_bracket_tokens(s: str) -> str:
    """Replace -LRB-, -RRB-, etc. with literal brackets. Preserve spacing."""
    if not isinstance(s, str) or pd.isna(s):
        return s
    for token, replacement in BRACKET_REPLACEMENTS:
        s = s.replace(token, replacement)
    return s


def fix_mojibake(s: str) -> str:
    """Replace known mojibake sequences. Preserve diacritics."""
    if not isinstance(s, str) or pd.isna(s):
        return s
    for bad, good in MOJIBAKE_REPLACEMENTS:
        s = s.replace(bad, good)
    return s


def normalize_quotes(s: str) -> str:
    """Replace double backticks with ", normalize quotation marks."""
    if not isinstance(s, str) or pd.isna(s):
        return s
    s = s.replace("``", '"')
    s = s.replace("''", '"')   # two single quotes used as double quote
    # Normalize curly/smart quotes to straight
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    return s


def fix_spacing(s: str) -> str:
    """Remove space before , . ; : ? !"""
    if not isinstance(s, str) or pd.isna(s):
        return s
    for punct in [",", ".", ";", ":", "?", "!"]:
        s = re.sub(r"\s+" + re.escape(punct), punct, s)
    return s


def unicode_nfc(s: str) -> str:
    """Unicode NFC normalization."""
    if not isinstance(s, str) or pd.isna(s):
        return s
    return unicodedata.normalize("NFC", s)


def clean_text(s: str) -> str:
    """Full pipeline: order as recommended."""
    s = html_unescape(s)
    s = replace_bracket_tokens(s)
    s = fix_mojibake(s)
    s = normalize_quotes(s)
    s = fix_spacing(s)
    s = unicode_nfc(s)
    return s


def clean_dataframe(df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
    """Apply cleaning to specified text columns."""
    out = df.copy()
    for col in text_columns:
        if col not in out.columns:
            continue
        out[col] = out[col].astype(str).apply(clean_text)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Очищення Wikipedia-артефактів у CSV (aligned_pairs_meaning80_*.csv)"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Вхідний CSV (наприклад data/aligned_pairs_meaning80_2.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Вихідний CSV (default: <input_stem>_cleaned.csv)",
    )
    parser.add_argument(
        "--columns",
        type=str,
        default=None,
        help="Колонки для очищення, через кому (default: original,simplified,topic)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Файл не знайдено: {args.input}")

    text_cols = [c.strip() for c in args.columns.split(",")] if args.columns else TEXT_COLUMNS

    df = pd.read_csv(args.input, encoding="utf-8")
    existing = [c for c in text_cols if c in df.columns]
    if not existing:
        raise SystemExit(f"Жодної з колонок {text_cols} не знайдено у файлі.")

    print(f"Очищення колонок: {existing}")
    df_cleaned = clean_dataframe(df, existing)

    out = args.output or args.input.parent / f"{args.input.stem}_cleaned.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df_cleaned.to_csv(out, index=False, encoding="utf-8")

    print(f"Збережено: {out} ({len(df_cleaned)} рядків)")


if __name__ == "__main__":
    main()
