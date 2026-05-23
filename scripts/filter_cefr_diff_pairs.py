#!/usr/bin/env python3
"""
Filter sentence pairs by CEFR expected-level difference.

Inputs (line-aligned files):
- original:   data/asset.valid.orig
- simplified: data/asset.valid.simp.0  (or .1)

Outputs:
- CSV with columns: original, simplified, cefr_orig, cefr_simp, cefr_diff
  filtered by cefr_diff >= threshold (default 0.5)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from transformers import pipeline


CEFR_ORDER = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}


def read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def build_pairs_df(original_path: Path, simplified_path: Path) -> pd.DataFrame:
    orig_lines = read_lines(original_path)
    simp_lines = read_lines(simplified_path)

    if len(orig_lines) != len(simp_lines):
        raise ValueError(
            f"Files are not line-aligned: {original_path} has {len(orig_lines)} lines, "
            f"{simplified_path} has {len(simp_lines)} lines."
        )

    return pd.DataFrame({"original": orig_lines, "simplified": simp_lines})


def cefr_expected_level(
    labeler,
    texts: List[str],
    batch_size: int = 32,
) -> List[float]:
    results: List[List[Dict]] = labeler(texts, truncation=True, batch_size=batch_size)
    return [
        float(sum(CEFR_ORDER[p["label"]] * p["score"] for p in preds))
        for preds in results
    ]


def filter_cefr_diff(
    df: pd.DataFrame,
    threshold: float = 0.8,
    sample_size: int | None = None,
    first_n: int | None = None,
    seed: int = 42,
    batch_size: int = 32,
) -> pd.DataFrame:
    data = df[["original", "simplified"]].replace("", np.nan).dropna().reset_index(drop=True)

    if first_n is not None and first_n > 0:
        data = data.head(first_n).reset_index(drop=True)

    if sample_size is not None and sample_size > 0 and sample_size < len(data):
        data = data.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    print(f"Аналізуємо {len(data)} пар...\n")

    cefr_labeler = pipeline(
        task="text-classification",
        model="AbdullahBarayan/ModernBERT-base-doc_sent_en-Cefr",
        top_k=None,
    )

    print("    -> Рахуємо CEFR рівні оригіналів...")
    orig_levels = cefr_expected_level(cefr_labeler, data["original"].tolist(), batch_size=batch_size)

    print("    -> Рахуємо CEFR рівні спрощень...")
    simp_levels = cefr_expected_level(cefr_labeler, data["simplified"].tolist(), batch_size=batch_size)

    data["cefr_orig"] = orig_levels
    data["cefr_simp"] = simp_levels
    data["cefr_diff"] = data["cefr_orig"] - data["cefr_simp"]

    filtered = data[data["cefr_diff"] >= threshold].reset_index(drop=True)
    kept = len(filtered)
    total = len(data)
    print(f"\nПісля фільтрації (cefr_diff >= {threshold}) залишилось: {kept} пар ({kept/total:.1%})")

    return filtered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original",
        type=Path,
        default=Path("data/asset.valid.orig"),
        help="Path to original lines file",
    )
    parser.add_argument(
        "--simplified",
        type=Path,
        required=True,
        help="Path to simplified lines file (e.g. data/asset.valid.simp.0)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: data/filtered_<simp_filename>.csv)",
    )
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument(
        "--first-n",
        type=int,
        default=None,
        help="Take the first N pairs (deterministic). Cannot be used together with --sample-size.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()

    if args.first_n is not None and args.sample_size is not None:
        raise SystemExit("Помилка: використовуйте або --first-n, або --sample-size (не разом).")

    if args.out is None:
        out_name = f"filtered_{args.simplified.name}.csv"
        args.out = Path("data") / out_name

    df_pairs = build_pairs_df(args.original, args.simplified)
    filtered = filter_cefr_diff(
        df_pairs,
        threshold=args.threshold,
        sample_size=args.sample_size,
        first_n=args.first_n,
        seed=args.seed,
        batch_size=args.batch_size,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(args.out, index=False)
    print(f"\nЗбережено: {args.out}")


if __name__ == "__main__":
    main()

