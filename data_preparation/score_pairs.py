#!/usr/bin/env python3
"""
Add cefr_orig, cefr_simp, cefr_diff to a pairs CSV (for Wiki/ASSET filtering pipeline).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from cefr_scoring import add_cefr_columns, load_cefr_labeler
from pairs_io import load_pairs_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    df = load_pairs_csv(args.input_csv)
    print(f"Scoring {len(df)} pairs...")
    labeler = load_cefr_labeler()
    scored = add_cefr_columns(df, labeler, batch_size=args.batch_size)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output_csv, index=False)
    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
