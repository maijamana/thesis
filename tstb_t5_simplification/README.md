# TSTB T5 Simplification

Iterative simplification with a fine-tuned T5 model. Uses the same expanded CSV and control loop as the LLM pipeline (`tstb_core`), without LLM rollback.

## Model

Place checkpoints under the project root (e.g. `t5-base-simplification-res4/`) with `model.safetensors`, config, and tokenizer files.

```bash
export T5_MODEL_PATH=/path/to/t5-base-simplification-res4
```

Inputs use the **`summarize: `** prefix (T5 training convention).

## TSAR run

```bash
# From repo root: build expanded CSV first (tstb_impact_score/main.py)

cd tstb_t5_simplification
python run_tsar_jsonl.py \
  ../data/data_for_evaluation/tsar2025_test.jsonl \
  ../data/data_for_evaluation/out.jsonl \
  --model-dir ../t5-base-simplification-res4 \
  --expanded_csv ../data/data_for_evaluation/tsar2025_expanded.csv \
  --limit 5 --max-iterations 8
```

## Batch CSV evaluation

```bash
python main.py ../data/data_tstb_expanded_target.csv ../data/results_t5.csv 0 1 2
```
