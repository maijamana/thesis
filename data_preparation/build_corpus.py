#!/usr/bin/env python3
"""
Assemble T5 training corpora (Run 1–3) per thesis filtering rules.

Canonical files (already in repo):
  data/data_for_training/data_wiki_asset_gpt.csv   — Run 1
  data/data_for_training/data_asset_gpt.csv        — Run 2
  data/data_for_training/data_simpwiki_asset_gpt.csv — Run 3

This script rebuilds a corpus from intermediate scored CSVs when you pass paths.
By default it writes to a NEW path unless --output matches your intent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import CORPUS_CONFIGS, DEFAULT_PATHS
from pairs_io import load_pairs_csv, normalize_pairs
from wiki_clean import clean_pairs_df


def _filter_wiki(df: pd.DataFrame, cefr_min: float, meaning_min: float | None) -> pd.DataFrame:
    out = df.copy()
    if "cefr_diff" not in out.columns:
        raise ValueError("Wiki source must include cefr_diff (pre-score with cefr_scoring.py or use full_meaning_filtered_wiki.csv)")
    out = out[out["cefr_diff"] >= cefr_min]
    if meaning_min is not None:
        if "meaning_score" not in out.columns:
            raise ValueError("Wiki source must include meaning_score for MeaningBERT filtering")
        out = out[out["meaning_score"] > meaning_min]
    return normalize_pairs(out)


def _filter_asset(df: pd.DataFrame, cefr_min: float) -> pd.DataFrame:
    out = df.copy()
    if "cefr_diff" in out.columns:
        out = out[out["cefr_diff"] >= cefr_min]
    return normalize_pairs(out)


def build_corpus(
    cfg_name: str,
    *,
    wiki_path: Path | None,
    asset_path: Path | None,
    gpt_path: Path | None,
    clean_wiki: bool = True,
) -> pd.DataFrame:
    cfg = CORPUS_CONFIGS[cfg_name]
    parts: list[pd.DataFrame] = []

    if cfg.include_wiki:
        if wiki_path is None or not wiki_path.exists():
            raise FileNotFoundError(f"Wiki scored CSV required for {cfg_name}: {wiki_path}")
        wiki = pd.read_csv(wiki_path)
        if clean_wiki:
            wiki = clean_pairs_df(wiki)
        wiki = _filter_wiki(wiki, cfg.wiki_cefr_min, cfg.wiki_meaning_min)
        parts.append(wiki)
        print(f"  Wiki: {len(wiki)} pairs (cefr_diff >= {cfg.wiki_cefr_min}" + (
            f", meaning > {cfg.wiki_meaning_min}" if cfg.wiki_meaning_min else ""
        ) + ")")

    if cfg.include_asset:
        if asset_path is None or not asset_path.exists():
            raise FileNotFoundError(f"ASSET scored CSV required for {cfg_name}: {asset_path}")
        asset = _filter_asset(pd.read_csv(asset_path), cfg.asset_cefr_min)
        parts.append(asset)
        print(f"  ASSET: {len(asset)} pairs (cefr_diff >= {cfg.asset_cefr_min})")

    if cfg.include_gpt:
        if gpt_path is None or not gpt_path.exists():
            raise FileNotFoundError(f"GPT CSV required for {cfg_name}: {gpt_path}")
        gpt = normalize_pairs(load_pairs_csv(gpt_path))
        parts.append(gpt)
        print(f"  GPT: {len(gpt)} pairs (no additional filtering per thesis)")

    if not parts:
        raise ValueError("No sources selected")

    merged = pd.concat(parts, ignore_index=True)
    merged = normalize_pairs(merged)
    print(f"  Combined (deduplicated): {len(merged)} pairs — expected {cfg.expected_pairs_note}")
    return merged


def verify_against_canonical(merged: pd.DataFrame, canonical: Path) -> None:
    if not canonical.exists():
        print(f"  Canonical file not found for verification: {canonical}")
        return
    ref = load_pairs_csv(canonical)
    print(f"  Verify: built={len(merged)}, canonical={len(ref)} (±{abs(len(merged)-len(ref))})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build T5 training corpus (thesis Run 1–3)")
    parser.add_argument("--run", choices=list(CORPUS_CONFIGS.keys()), required=True)
    parser.add_argument("--output", type=Path, default=None, help="Output CSV (default: ./_rebuilt_<run>.csv)")
    parser.add_argument("--wiki-csv", type=Path, default=None, help="Wiki pairs with cefr_diff [, meaning_score]")
    parser.add_argument("--asset-csv", type=Path, default=None, help="ASSET pairs with cefr_diff")
    parser.add_argument("--gpt-csv", type=Path, default=None, help="GPT synthetic pairs")
    parser.add_argument("--no-wiki-clean", action="store_true")
    parser.add_argument("--verify", action="store_true", help="Compare row count to canonical file")
    args = parser.parse_args()

    cfg = CORPUS_CONFIGS[args.run]
    wiki = args.wiki_csv or DEFAULT_PATHS["wiki_scored_with_meaning"]
    asset = args.asset_csv or DEFAULT_PATHS["asset_scored"]
    gpt = args.gpt_csv or DEFAULT_PATHS["gpt_pairs"]

    print(f"Building {cfg.name}: {cfg.description}\n")
    merged = build_corpus(
        args.run,
        wiki_path=wiki if cfg.include_wiki else None,
        asset_path=asset if cfg.include_asset else None,
        gpt_path=gpt if cfg.include_gpt else None,
        clean_wiki=not args.no_wiki_clean,
    )

    out = args.output or Path(f"_rebuilt_{args.run}.csv")
    merged.to_csv(out, index=False)
    print(f"\nSaved: {out}")

    if args.verify:
        verify_against_canonical(merged, cfg.canonical_output)


if __name__ == "__main__":
    main()
