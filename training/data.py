"""Load aligned simplification pairs and create train/validation/test splits."""

from __future__ import annotations

import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split


def load_pairs_csv(path: str) -> pd.DataFrame:
    """Load CSV with original/simplified columns (aliases orig/simp supported)."""
    df = pd.read_csv(path)
    if "original" in df.columns and "orig" not in df.columns:
        df = df.rename(columns={"original": "orig"})
    if "simplified" in df.columns and "simp" not in df.columns:
        df = df.rename(columns={"simplified": "simp"})
    if not {"orig", "simp"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: original, simplified (or orig, simp)")
    return df.dropna(subset=["orig", "simp"]).reset_index(drop=True)


def split_dataframe(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """80/10/10 split: train | valid | test."""
    train_df, temp_df = train_test_split(df, test_size=test_size, random_state=random_state)
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=random_state)
    return (
        train_df.reset_index(drop=True),
        valid_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def tokenize_datasets(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer,
    prefix: str,
    max_length: int,
):
    """Tokenize orig/simp pairs for Seq2SeqTrainer."""

    def preprocess_function(examples):
        inputs = [prefix + s for s in examples["orig"]]
        model_inputs = tokenizer(
            inputs,
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        labels = tokenizer(
            text_target=examples["simp"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def to_hf(frame: pd.DataFrame) -> Dataset:
        return Dataset.from_pandas(frame[["orig", "simp"]].reset_index(drop=True))

    tokenized_train = to_hf(train_df).map(
        preprocess_function, batched=True, remove_columns=["orig", "simp"]
    )
    tokenized_valid = to_hf(valid_df).map(
        preprocess_function, batched=True, remove_columns=["orig", "simp"]
    )
    tokenized_test = to_hf(test_df).map(
        preprocess_function, batched=True, remove_columns=["orig", "simp"]
    )
    return tokenized_train, tokenized_valid, tokenized_test
