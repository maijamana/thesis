"""Continuous CEFR expected scores for sentence pairs (thesis Eq. expected level)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

CEFR_ORDER = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}

if TYPE_CHECKING:
    from transformers import Pipeline


def load_cefr_labeler(device: int = 0):
    from transformers import pipeline

    from .config import CEFR_MODEL_ID

    return pipeline(
        task="text-classification",
        model=CEFR_MODEL_ID,
        top_k=None,
        device=device,
    )


def cefr_expected_levels(texts: list[str], labeler, batch_size: int = 32) -> list[float]:
    texts = [t if t and str(t).strip() else "simple" for t in texts]
    batch_preds = labeler(texts, truncation=True, batch_size=batch_size)
    return [
        float(sum(CEFR_ORDER[p["label"]] * float(p["score"]) for p in preds))
        for preds in batch_preds
    ]


def add_cefr_columns(df: pd.DataFrame, labeler, batch_size: int = 32) -> pd.DataFrame:
    """Add cefr_orig, cefr_simp, cefr_diff to a frame with original/simplified columns."""
    out = df.copy()
    out["cefr_orig"] = cefr_expected_levels(out["original"].tolist(), labeler, batch_size)
    out["cefr_simp"] = cefr_expected_levels(out["simplified"].tolist(), labeler, batch_size)
    out["cefr_diff"] = out["cefr_orig"] - out["cefr_simp"]
    return out


def filter_by_cefr_diff(df: pd.DataFrame, min_diff: float) -> pd.DataFrame:
    if "cefr_diff" not in df.columns:
        raise ValueError("DataFrame must have cefr_diff column; run add_cefr_columns first.")
    return df[df["cefr_diff"] >= min_diff].reset_index(drop=True)
