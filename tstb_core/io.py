"""I/O helpers for TSAR JSONL and expanded sentence CSV files."""

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_tsar_jsonl(path: str) -> list[dict[str, Any]]:
    """Load a TSAR-style JSONL file (one JSON object per line)."""
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_tsar_jsonl(path: str, records: list[dict[str, Any]]) -> None:
    """Write records to JSONL."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_expanded_csv(path: str) -> pd.DataFrame:
    """Load sentence-level expanded CSV produced by tstb_impact_score."""
    df = pd.read_csv(path)
    required = {"text_id", "sentence_index", "sentence", "ExpectedScore"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"expanded CSV missing columns: {sorted(missing)}")
    return df
