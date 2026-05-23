#!/usr/bin/env python3
"""
TSAR JSONL runner for iterative LLM simplification (Groq or OpenAI).

Requires expanded CSV from tstb_impact_score (spaCy segmentation + ExpectedScore ranking).
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tstb_core.cefr_scorer import CEFRScorer
from tstb_core.io import load_expanded_csv, load_tsar_jsonl, write_tsar_jsonl
from tstb_core.text_utils import get_original_and_simplified_text

from llm_simplifier import simplify_until_target_iterative_llm


def run_tsar_through_pipeline(
    input_jsonl: str,
    expanded_csv: str,
    output_jsonl: str,
    provider: str,
    model: str,
    limit: int | None = None,
    sleep_s: float = 2.0,
) -> None:
    load_dotenv()

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")
        client = Groq(api_key=api_key)
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        client = OpenAI(api_key=api_key)
    else:
        raise ValueError("provider must be 'groq' or 'openai'")

    records = load_tsar_jsonl(input_jsonl)
    if limit is not None:
        records = records[:limit]
    print(f"Loaded {len(records)} records from {input_jsonl}")

    data = load_expanded_csv(expanded_csv)
    if limit is not None:
        data = data[data["text_id"] < len(records)].copy()
    if data.empty:
        raise ValueError("expanded CSV has no rows for this run")

    print(f"Expanded: texts={data['text_id'].nunique()}, sentences={len(data)}")
    print(f"Provider={provider}, model={model}")

    scorer = CEFRScorer()
    output_records = []

    for i in range(len(records)):
        rec = dict(records[i])
        target_cefr = str(rec.get("target_cefr", "A2")).strip().upper()
        if target_cefr not in ("A1", "A2", "B1", "B2", "C1", "C2"):
            target_cefr = "A2"

        print(f"\n[{i + 1}/{len(records)}] text_id={rec.get('text_id', i)} target_cefr={target_cefr}")
        try:
            simplified_df = simplify_until_target_iterative_llm(
                data,
                text_id=i,
                target_level=target_cefr,
                scorer=scorer,
                client=client,
                provider=provider,
                model=model,
            )
            _, simplified_text = get_original_and_simplified_text(simplified_df)
            rec["simplified"] = simplified_text
        except Exception as e:
            print(f"  Error: {e}")
            rec["simplified"] = ""
        output_records.append(rec)
        if sleep_s:
            time.sleep(sleep_s)

    write_tsar_jsonl(output_jsonl, output_records)
    print(f"\nSaved {len(output_records)} records to {output_jsonl}")


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: python run_tsar_jsonl.py <input.jsonl> [output.jsonl] "
            "--provider groq|openai --model MODEL "
            "[--expanded_csv PATH] [--limit N] [--sleep SEC]"
        )
        sys.exit(1)

    input_jsonl = sys.argv[1]
    provider = None
    model = None
    expanded_csv = None
    limit = None
    sleep_s = 2.0
    rest = []

    argv = sys.argv[2:]
    i = 0
    while i < len(argv):
        if argv[i] == "--provider" and i + 1 < len(argv):
            provider = argv[i + 1]
            i += 2
        elif argv[i] == "--model" and i + 1 < len(argv):
            model = argv[i + 1]
            i += 2
        elif argv[i] == "--expanded_csv" and i + 1 < len(argv):
            expanded_csv = argv[i + 1]
            i += 2
        elif argv[i] == "--limit" and i + 1 < len(argv):
            limit = int(argv[i + 1])
            i += 2
        elif argv[i] == "--sleep" and i + 1 < len(argv):
            sleep_s = float(argv[i + 1])
            i += 2
        else:
            rest.append(argv[i])
            i += 1

    if not Path(input_jsonl).exists():
        print(f"File not found: {input_jsonl}")
        sys.exit(1)

    load_dotenv()
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "groq")
    if model is None:
        if provider == "groq":
            model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        else:
            model = os.getenv("OPENAI_MODEL", "gpt-4.1")

    default_out = _ROOT / "data" / "data_for_evaluation" / "tsar2025_test_llm_simplified.jsonl"
    output_jsonl = rest[0] if rest else str(default_out)
    if expanded_csv is None:
        expanded_csv = str(_ROOT / "data" / "data_for_evaluation" / "tsar2025_expanded.csv")

    run_tsar_through_pipeline(
        input_jsonl=input_jsonl,
        expanded_csv=expanded_csv,
        output_jsonl=output_jsonl,
        provider=provider,
        model=model,
        limit=limit,
        sleep_s=sleep_s,
    )


if __name__ == "__main__":
    main()
