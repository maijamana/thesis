#!/usr/bin/env python3
"""
Одноразовий скрипт: для кожного запису в tsar JSONL обчислює CEFR рівень оригінального тексту.
Якщо target_cefr вищий за рівень оригіналу (ціль складніша за текст) — замінює simplified на original.
Результат зберігає в новий JSONL.

Використання:
  python scripts/fix_simplified_when_target_above_original.py data/data_for_evaluation/tsar2025_test_simplified_rmse.jsonl [output.jsonl]
"""
import json
import sys
from pathlib import Path

# Імпорт CEFRScorer з проєкту
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tstb_t5_simplification.cefr_scorer import CEFRScorer

CEFR_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2"]
LABEL2IDX = {l: i for i, l in enumerate(CEFR_ORDER)}


def normalize_level(label: str) -> str:
    l = str(label).strip().upper()
    return l if l in LABEL2IDX else "B1"


def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_simplified_when_target_above_original.py <input.jsonl> [output.jsonl]")
        sys.exit(1)
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path.parent / (
        input_path.stem + "_fixed.jsonl"
    )

    if not input_path.exists():
        print(f"File not found: {input_path}")
        sys.exit(1)

    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    print(f"Loaded {len(records)} records. Computing CEFR for originals...")
    scorer = CEFRScorer()
    replaced = 0
    for i, rec in enumerate(records):
        original = rec.get("original", "").strip()
        target = normalize_level(rec.get("target_cefr", "A2"))
        pred = scorer.get_cefr_label(original)
        orig_level = normalize_level(pred["label"])
        target_idx = LABEL2IDX[target]
        orig_idx = LABEL2IDX[orig_level]
        if target_idx > orig_idx:
            rec["simplified"] = original
            replaced += 1
            if replaced <= 5:
                print(f"  [{rec.get('text_id')}] target={target} > original={orig_level} → simplified := original")
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(records)}...")

    print(f"Replaced simplified with original for {replaced} records (target level > original level).")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
