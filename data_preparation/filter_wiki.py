#!/usr/bin/env python3
"""
Filter WikiLarge pairs by CEFR difference and MeaningBERT (thesis Run 1 / Run 3 wiki leg).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from pairs_io import normalize_pairs
from wiki_clean import clean_pairs_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path, help="CSV with cefr_diff and meaning_score")
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--cefr-min", type=float, default=0.8)
    parser.add_argument("--meaning-min", type=float, default=0.65)
    parser.add_argument("--no-clean", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if not args.no_clean:
        df = clean_pairs_df(df)
    df = df[df["cefr_diff"] >= args.cefr_min]
    if args.meaning_min is not None:
        df = df[df["meaning_score"] > args.meaning_min]
    out = normalize_pairs(df)
    out.to_csv(args.output_csv, index=False)
    print(f"Kept {len(out)} / {len(pd.read_csv(args.input_csv))} → {args.output_csv}")


if __name__ == "__main__":
    main()
