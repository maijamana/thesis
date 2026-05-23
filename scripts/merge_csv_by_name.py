#!/usr/bin/env python3
"""
Об'єднує кілька CSV з однаковою структурою в один файл.

Можна передати glob-шаблон (наприклад data/aligned_pairs_meaning80_*.csv)
або список файлів. Файли сортуються за назвою перед об'єднанням.
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import pandas as pd


def _sort_key(path: str) -> tuple:
    """Сортування за назвою: _0, _1, _2 перед _10."""
    p = Path(path)
    stem = p.stem
    m = re.search(r"_(\d+)$", stem)
    if m:
        return (stem[: m.start()], int(m.group(1)))
    return (stem, 0)


def main():
    parser = argparse.ArgumentParser(
        description="Об'єднання CSV за шаблоном імені або списком файлів"
    )
    parser.add_argument(
        "input",
        nargs="+",
        help="Glob-шаблон або список файлів, напр. data/aligned_pairs_meaning80_*.csv або file1.csv file2.csv",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Вихідний CSV (default: з першого входу _merged.csv)",
    )
    parser.add_argument(
        "--drop-duplicates",
        action="store_true",
        help="Видалити дублікати рядків після об'єднання",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Колонки для визначення дублікатів, через кому (default: всі рядки порівнюються)",
    )
    args = parser.parse_args()

    # Розгортаємо glob, якщо передано один аргумент-шаблон
    paths = []
    for p in args.input:
        if "*" in p or "?" in p:
            paths.extend(glob.glob(p))
        else:
            paths.append(p)

    paths = sorted(set(paths), key=_sort_key)

    if not paths:
        raise SystemExit("Не знайдено жодного файлу.")

    for p in paths:
        if not Path(p).exists():
            raise SystemExit(f"Файл не знайдено: {p}")

    print(f"Читаю {len(paths)} файлів:")
    for p in paths:
        print(f"  {p}")

    dfs = []
    for p in paths:
        df = pd.read_csv(p, encoding="utf-8")
        dfs.append(df)
        print(f"  -> {len(df)} рядків")

    merged = pd.concat(dfs, ignore_index=True)
    print(f"\nРазом до дедуплікації: {len(merged)} рядків")

    if args.drop_duplicates:
        subset = [c.strip() for c in args.subset.split(",")] if args.subset else None
        if subset:
            for c in subset:
                if c not in merged.columns:
                    raise SystemExit(f"Колонки '{c}' немає в даних.")
        before = len(merged)
        merged = merged.drop_duplicates(subset=subset).reset_index(drop=True)
        print(f"Після видалення дублікатів: {len(merged)} рядків (видалено {before - len(merged)})")

    out = args.output
    if out is None:
        first = Path(paths[0]).stem
        # Прибираємо суфікс _0, _1 тощо для імені merged
        base = first.rstrip("_0123456789")
        out = Path(paths[0]).parent / f"{base}_merged.csv"

    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False, encoding="utf-8")
    print(f"\nЗбережено: {out} ({len(merged)} рядків)")


if __name__ == "__main__":
    main()
