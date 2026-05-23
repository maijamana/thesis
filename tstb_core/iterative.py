"""Iterative document simplification loop shared by LLM and T5 backends."""

from collections.abc import Callable
from typing import Any

import pandas as pd

from .constants import CEFR_ORDER
from .text_utils import assemble_document_text


def _level_distance(a: str, b: str) -> int:
    return abs(CEFR_ORDER.index(a) - CEFR_ORDER.index(b))


def _is_overshoot(new_cefr: str, target_level: str) -> bool:
    if new_cefr not in CEFR_ORDER or target_level not in CEFR_ORDER:
        return False
    return CEFR_ORDER.index(new_cefr) < CEFR_ORDER.index(target_level)


def _neighbor_sentence(text_df: pd.DataFrame, text_id: int, idx: int, offset: int) -> str:
    rows = text_df[text_df["text_id"] == text_id]
    result = rows.loc[rows["sentence_index"] == idx + offset, "simplified_sentence"]
    if result.empty:
        return ""
    value = result.squeeze()
    if pd.isna(value) or str(value).strip() == "":
        return ""
    return str(value)


def simplify_until_target_iterative(
    text_df: pd.DataFrame,
    text_id: int,
    target_level: str,
    scorer: Any,
    simplify_sentence: Callable[..., str],
    *,
    max_iterations: int = 8,
    llm_rollback: bool = False,
) -> pd.DataFrame:
    """
  Rank sentences by ExpectedScore, simplify iteratively, re-score the full document after each change.

  Args:
      text_df: Expanded sentence table (text_id, sentence_index, sentence, ExpectedScore).
      simplify_sentence: Backend callable. LLM backends accept prompt_style= when llm_rollback=True.
      llm_rollback: If True, retry with a mild prompt when CEFR drops below target_level.
    """
    subset = (
        text_df[text_df["text_id"] == text_id]
        .copy()
        .sort_values("ExpectedScore", ascending=False)
        .reset_index(drop=True)
    )
    text_df = text_df.copy()

    for col in ("simplified_sentence", "current_cefr_label"):
        if col not in text_df.columns:
            text_df[col] = None

    current_cefr = str(scorer.get_cefr_label(assemble_document_text(text_df, text_id, use_simplified=False))["label"])
    distance = _level_distance(current_cefr, target_level)
    coarse_mode = distance > 1

    print(
        f"Starting CEFR={current_cefr}, target={target_level}, "
        f"distance={distance}, coarse_mode={coarse_mode}"
    )

    iteration = 1
    while True:
        print(f"\nIteration {iteration}: starting simplification cycle...")

        for _, row in subset.iterrows():
            idx = int(row["sentence_index"])
            mask = (text_df["text_id"] == text_id) & (text_df["sentence_index"] == idx)

            base_sentence = text_df.loc[mask, "simplified_sentence"].squeeze()
            if pd.isna(base_sentence) or str(base_sentence).strip() == "":
                base_sentence = row["sentence"]

            prev_sent = _neighbor_sentence(text_df, text_id, idx, -1)
            next_sent = _neighbor_sentence(text_df, text_id, idx, 1)
            prev_value = text_df.loc[mask, "simplified_sentence"].squeeze()

            if llm_rollback:
                attempts = [("normal", "normal"), ("mild-1", "mild"), ("mild-2", "mild")]
            else:
                attempts = [("default", "normal")]

            committed = False
            for label, style in attempts:
                if llm_rollback:
                    simplified = simplify_sentence(
                        base_sentence,
                        prev_sent,
                        next_sent,
                        prompt_style=style,
                        target_level=target_level,
                    )
                else:
                    simplified = simplify_sentence(base_sentence, prev_sent, next_sent)

                preview = simplified[:120] + "..." if len(simplified) > 120 else simplified
                print(f"  Simplified sentence {idx} ({label}): {preview}")

                text_df.loc[mask, "simplified_sentence"] = simplified
                new_cefr = str(
                    scorer.get_cefr_label(assemble_document_text(text_df, text_id))["label"]
                )

                if llm_rollback and _is_overshoot(new_cefr, target_level):
                    print(
                        f"    Overshoot ({new_cefr} < target {target_level}) → rollback and retry."
                    )
                    text_df.loc[mask, "simplified_sentence"] = prev_value
                    continue

                committed = True
                current_cefr = new_cefr
                text_df.loc[text_df["text_id"] == text_id, "current_cefr_label"] = current_cefr
                print(f"    After simplifying {idx} → CEFR={current_cefr}")
                break

            if llm_rollback and not committed:
                print(f"    Keeping previous sentence {idx} (all attempts overshot).")

            if _level_distance(current_cefr, target_level) == 0:
                print(f"\nTarget {target_level} reached (iteration {iteration})")
                return (
                    text_df[text_df["text_id"] == text_id]
                    .sort_values("sentence_index")
                    .reset_index(drop=True)
                )

        current_cefr = str(
            scorer.get_cefr_label(assemble_document_text(text_df, text_id))["label"]
        )
        text_df.loc[text_df["text_id"] == text_id, "current_cefr_label"] = current_cefr
        print(f"End of iteration {iteration} → CEFR now {current_cefr}")

        if coarse_mode and _level_distance(current_cefr, target_level) <= 1:
            print("Switching to fine-grained mode (distance now ≤ 1).")
            coarse_mode = False

        if _level_distance(current_cefr, target_level) == 0:
            print(f"Reached target {target_level} after full iteration {iteration}")
            return (
                text_df[text_df["text_id"] == text_id]
                .sort_values("sentence_index")
                .reset_index(drop=True)
            )

        iteration += 1
        if iteration > max_iterations:
            print(f"Stopping after {max_iterations} iterations (max_iterations cap).")
            break

    return (
        text_df[text_df["text_id"] == text_id]
        .sort_values("sentence_index")
        .reset_index(drop=True)
    )
