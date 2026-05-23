#!/usr/bin/env python3
"""
Build expanded sentence-level CSV from JSON/JSONL inputs.

Segments each document with spaCy, scores sentences (ExpectedScore), ranks by complexity,
and records document-level CEFR from the three-model ensemble.
"""

import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tstb_core.cefr_scorer import CEFRScorer
from tstb_core.constants import LEVEL_MAP
from tstb_core.sentence_analyzer import SentenceInfluenceAnalyzerBySentence


def load_data(input_file: str) -> pd.DataFrame:
    """Load JSON or JSONL; require an 'original' or 'text' column."""
    print(f"Loading {input_file}...")
    path = Path(input_file)

    if path.suffix.lower() == ".jsonl":
        rows = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        df = pd.DataFrame(rows)
    else:
        with open(input_file, "r", encoding="utf-8") as f:
            df = pd.DataFrame(json.load(f))

    if "text" in df.columns and "original" not in df.columns:
        df["original"] = df["text"]
    if "original" not in df.columns:
        raise ValueError("Input must contain 'original' or 'text'.")

    print(f"Loaded {len(df)} records")
    return df


def process_data(
    data: pd.DataFrame,
    scorer: CEFRScorer,
    analyzer: SentenceInfluenceAnalyzerBySentence,
) -> pd.DataFrame:
    """Produce ranked sentence rows with ExpectedScore and impact_score."""
    print("Scoring document-level CEFR...")
    cefr_results = data["original"].apply(lambda x: pd.Series(scorer.get_cefr_label(x)))
    data[["cefr_level", "cefr_score"]] = cefr_results

    if "id" not in data.columns:
        data["id"] = data.index

    print("Analyzing sentences...")
    results = []
    for counter, (_, row) in enumerate(data.iterrows(), start=1):
        text_id = row["id"]
        original_cefr = row["cefr_level"]
        analysis = analyzer.analyze_sentence_impact(row["original"])
        orig_num = LEVEL_MAP.get(original_cefr)

        for sent_info in analysis["impact_analysis"]:
            exp_score = sent_info["ExpectedScore"]
            impact_score = exp_score - orig_num if orig_num is not None else None
            results.append(
                {
                    "text_id": text_id,
                    "original_cefr_level": original_cefr,
                    "original_model_level": sent_info["original_level"],
                    "original_cefr_num": orig_num,
                    "sentence_index": sent_info["sentence_index"],
                    "sentence": sent_info["sentence"],
                    "ExpectedScore": exp_score,
                    "impact_score": impact_score,
                }
            )

        if counter % 10 == 0:
            print(f"  Processed {counter} documents...")

    expanded = pd.DataFrame(results)
    return expanded.sort_values(
        by=["text_id", "ExpectedScore"], ascending=[True, False]
    ).reset_index(drop=True)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python main.py <input.json|jsonl> [output.csv]")
        sys.exit(1)

    input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        output_file = str(_ROOT / "data" / "data_tstb_expanded_target.csv")

    if not Path(input_file).exists():
        print(f"File not found: {input_file}")
        sys.exit(1)

    data = load_data(input_file)
    scorer = CEFRScorer()
    analyzer = SentenceInfluenceAnalyzerBySentence(scorer)
    result_df = process_data(data, scorer, analyzer)

    print(f"Saving {output_file}...")
    result_df.to_csv(output_file, index=False)
    print(f"Done. Sentences: {len(result_df)}, texts: {result_df['text_id'].nunique()}")


if __name__ == "__main__":
    main()
