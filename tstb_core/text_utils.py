"""Helpers to assemble documents from sentence-level DataFrames."""

import re

import pandas as pd


def assemble_document_text(text_df: pd.DataFrame, text_id: int, use_simplified: bool = True) -> str:
    """Join sentences for a text_id in sentence_index order."""
    rows = text_df[text_df["text_id"] == text_id].sort_values("sentence_index")
    if use_simplified:
        col = rows["simplified_sentence"].fillna(rows["sentence"])
    else:
        col = rows["sentence"]
    return " ".join(col.astype(str).tolist())


def get_original_and_simplified_text(result_df: pd.DataFrame) -> tuple[str, str]:
    """Return (original_text, simplified_text) from a per-sentence result DataFrame."""
    ordered = result_df.sort_values("sentence_index", ascending=True)
    original_text = " ".join(ordered["sentence"].astype(str).tolist()).strip()

    def clean_quotes(s: str) -> str:
        return re.sub(r'^["\'\u201c\u201d\u2018\u2019]+|["\'\u201c\u201d\u2018\u2019]+$', "", s.strip())

    simplified_parts = []
    for orig, simp in zip(ordered["sentence"], ordered["simplified_sentence"]):
        if pd.isna(simp) or str(simp).strip() == "":
            simplified_parts.append(clean_quotes(str(orig)))
        else:
            simplified_parts.append(clean_quotes(str(simp)))

    return original_text, " ".join(simplified_parts).strip()
