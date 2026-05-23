"""Load and normalize sentence-pair tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def normalize_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Keep original/simplified only; strip; drop empty and duplicate pairs."""
    if "orig" in df.columns and "original" not in df.columns:
        df = df.rename(columns={"orig": "original", "simp": "simplified"})
    if not {"original", "simplified"}.issubset(df.columns):
        raise ValueError("Need columns original and simplified (or orig/simp).")
    out = df[["original", "simplified"]].copy()
    out["original"] = out["original"].astype(str).str.strip()
    out["simplified"] = out["simplified"].astype(str).str.strip()
    out = out[(out["original"] != "") & (out["simplified"] != "")]
    return out.drop_duplicates(subset=["original", "simplified"]).reset_index(drop=True)


def load_pairs_csv(path: Path) -> pd.DataFrame:
    return normalize_pairs(pd.read_csv(path))


def load_asset_aligned(orig_path: Path, simp_path: Path) -> pd.DataFrame:
    """Read ASSET validation split as line-aligned original/simplified files."""
    orig = [ln.rstrip("\n") for ln in orig_path.read_text(encoding="utf-8").splitlines()]
    simp = [ln.rstrip("\n") for ln in simp_path.read_text(encoding="utf-8").splitlines()]
    if len(orig) != len(simp):
        raise ValueError(f"ASSET files not aligned: {len(orig)} vs {len(simp)} lines")
    return normalize_pairs(pd.DataFrame({"original": orig, "simplified": simp}))
