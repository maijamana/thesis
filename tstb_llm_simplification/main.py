#!/usr/bin/env python3
"""
Batch iterative LLM simplification on expanded CSV with MeaningBERT and BERTScore.
"""

import os
import sys
import time
from pathlib import Path

import evaluate
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tstb_core.cefr_scorer import CEFRScorer
from tstb_core.text_utils import get_original_and_simplified_text

from evaluator import get_bertscore, get_meaningbert_score
from llm_simplifier import simplify_until_target_iterative_llm


def parse_text_ids(args: list[str]) -> list[int]:
    ids: list[int] = []
    for arg in args:
        if "-" in arg:
            start, end = map(int, arg.split("-"))
            ids.extend(range(start, end + 1))
        else:
            ids.append(int(arg))
    return sorted(set(ids))


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python main.py <expanded.csv> <output.csv> [text_ids...]")
        sys.exit(1)

    input_file, output_file = sys.argv[1], sys.argv[2]
    load_dotenv()
    data = pd.read_csv(input_file)
    text_ids = parse_text_ids(sys.argv[3:]) if len(sys.argv) > 3 else sorted(data["text_id"].unique())
    target_levels = ["A1", "A2", "B1"]

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    scorer = CEFRScorer()
    meaning_bert = evaluate.load("davebulaval/meaningbert")
    bertscore = evaluate.load("bertscore")

    results = []
    sentence_rows = []

    for idx, text_id in enumerate(text_ids):
        target = target_levels[idx % len(target_levels)]
        subset = data[data["text_id"] == text_id]
        if subset.empty:
            continue

        print(f"\n text_id={text_id} target={target}")
        original_text = " ".join(subset["sentence"].astype(str))
        cefr_before = scorer.get_cefr_label(original_text)

        simplified_df = simplify_until_target_iterative_llm(
            data, text_id, target, scorer, client, "groq", model
        )
        original_text, simplified_text = get_original_and_simplified_text(simplified_df)
        cefr_after = scorer.get_cefr_label(simplified_text)

        for _, row in simplified_df.iterrows():
            sentence_rows.append(
                {
                    "text_id": text_id,
                    "sentence_index": int(row["sentence_index"]),
                    "original_sentence": row["sentence"],
                    "simplified_sentence": row["simplified_sentence"]
                    if pd.notna(row["simplified_sentence"])
                    else row["sentence"],
                    "target_cefr_level": target,
                }
            )

        results.append(
            {
                "text_id": text_id,
                "target_cefr_level": target,
                "original_text": original_text,
                "simplified_text": simplified_text,
                "CEFR_before": cefr_before["label"],
                "CEFR_after": cefr_after["label"],
                "MeaningBERT": get_meaningbert_score([simplified_text], [original_text], meaning_bert),
                "BERTScore": get_bertscore([simplified_text], [original_text], bertscore),
            }
        )
        time.sleep(5)

    out = Path(output_file)
    pd.DataFrame(results).to_csv(out, index=False)
    pd.DataFrame(sentence_rows).to_csv(out.parent / f"{out.stem}_sentences.csv", index=False)
    print(f"Saved {output_file}")


if __name__ == "__main__":
    main()
