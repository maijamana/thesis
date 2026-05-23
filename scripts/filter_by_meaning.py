#!/usr/bin/env python3
"""
Фільтрує CSV за колонкою meaning_score.
Залишає лише рядки з meaning_score > поріг (за замовчуванням 0.75).
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Фільтрація рядків за meaning_score > поріг"
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("data/aligned_pairs_filtered_meaning.csv"),
        help="Вхідний CSV з колонкою meaning_score (default: data/aligned_pairs_filtered_meaning.csv)",
    )
    parser.add_argument(
        "--min-meaning",
        type=float,
        default=0.75,
        help="Залишити рядки з meaning_score > цього значення (default: 0.75)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Вихідний CSV (default: <input>_meaning_gt_<min>.csv)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Файл не знайдено: {args.input}")

    df = pd.read_csv(args.input, encoding="utf-8")

    if "meaning_score" not in df.columns:
        raise SystemExit("У файлі немає колонки 'meaning_score'.")

    before = len(df)
    filtered = df[df["meaning_score"] > args.min_meaning].reset_index(drop=True)
    after = len(filtered)

    out = args.output
    if out is None:
        min_str = str(args.min_meaning).replace(".", "_")
        out = args.input.parent / f"{args.input.stem}_meaning_gt_{min_str}.csv"

    out.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(out, index=False, encoding="utf-8")

    print(f"Вхід: {args.input} ({before} рядків)")
    print(f"Поріг: meaning_score > {args.min_meaning}")
    print(f"Вихід: {out} ({after} рядків)")
    if before > 0:
        print(f"Залишено: {after / before:.1%}")


if __name__ == "__main__":
    main()
