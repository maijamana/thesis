#!/usr/bin/env python3
"""
TSAR JSONL runner for iterative T5 simplification.

Requires expanded CSV from tstb_impact_score (same path as the LLM pipeline).
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tstb_core.cefr_scorer import CEFRScorer
from tstb_core.io import load_expanded_csv, load_tsar_jsonl, write_tsar_jsonl
from tstb_core.text_utils import get_original_and_simplified_text

from t5_simplifier import load_t5_model, simplify_until_target_iterative_t5


def run_tsar_through_pipeline(
    input_jsonl: str,
    expanded_csv: str,
    output_jsonl: str,
    model_path: str,
    limit: int | None = None,
    max_iterations: int = 8,
) -> None:
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
    print(f"Loading T5 from {model_path}...")
    model, tokenizer, device = load_t5_model(model_path)
    scorer = CEFRScorer()

    output_records = []
    for i in range(len(records)):
        rec = dict(records[i])
        target_cefr = str(rec.get("target_cefr", "A2")).strip().upper()
        if target_cefr not in ("A1", "A2", "B1", "B2", "C1", "C2"):
            target_cefr = "A2"

        print(f"\n[{i + 1}/{len(records)}] text_id={rec.get('text_id', i)} target_cefr={target_cefr}")
        try:
            simplified_df = simplify_until_target_iterative_t5(
                data,
                text_id=i,
                target_level=target_cefr,
                scorer=scorer,
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_iterations=max_iterations,
            )
            _, simplified_text = get_original_and_simplified_text(simplified_df)
            rec["simplified"] = simplified_text
        except Exception as e:
            print(f"  Error: {e}")
            rec["simplified"] = ""
        output_records.append(rec)

    write_tsar_jsonl(output_jsonl, output_records)
    print(f"\nSaved {len(output_records)} records to {output_jsonl}")


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: python run_tsar_jsonl.py <input.jsonl> <output.jsonl> "
            "--model-dir PATH [--expanded_csv PATH] [--limit N] [--max-iterations N]"
        )
        sys.exit(1)

    input_jsonl = sys.argv[1]
    model_dir = None
    expanded_csv = None
    limit = None
    max_iterations = 8
    rest = []

    argv = sys.argv[2:]
    i = 0
    while i < len(argv):
        if argv[i] == "--model-dir" and i + 1 < len(argv):
            model_dir = argv[i + 1]
            i += 2
        elif argv[i] == "--expanded_csv" and i + 1 < len(argv):
            expanded_csv = argv[i + 1]
            i += 2
        elif argv[i] == "--limit" and i + 1 < len(argv):
            limit = int(argv[i + 1])
            i += 2
        elif argv[i] == "--max-iterations" and i + 1 < len(argv):
            max_iterations = int(argv[i + 1])
            i += 2
        else:
            rest.append(argv[i])
            i += 1

    if not Path(input_jsonl).exists():
        print(f"File not found: {input_jsonl}")
        sys.exit(1)
    if not model_dir:
        print("Error: --model-dir is required")
        sys.exit(1)

    model_path = str(Path(model_dir).resolve())
    if not Path(model_path).exists():
        print(f"Model directory not found: {model_path}")
        sys.exit(1)

    default_out = _ROOT / "data" / "data_for_evaluation" / "tsar2025_test_t5_simplified.jsonl"
    output_jsonl = rest[0] if rest else str(default_out)
    if expanded_csv is None:
        expanded_csv = str(_ROOT / "data" / "data_for_evaluation" / "tsar2025_expanded.csv")

    run_tsar_through_pipeline(
        input_jsonl,
        expanded_csv,
        output_jsonl,
        model_path,
        limit=limit,
        max_iterations=max_iterations,
    )


if __name__ == "__main__":
    main()
