# Thesis — Text Simplification & Evaluation (TSAR 2025)

Pipelines for sentence-level impact scoring, iterative CEFR-controlled simplification (LLM or T5), and TSAR-style evaluation.

## Project layout

```
Thesis/
├── tstb_core/                 # Shared: CEFR, preprocessing, iterative loop, prompts
├── tstb_impact_score/         # JSON/JSONL → expanded CSV (spaCy + ExpectedScore)
├── tstb_llm_simplification/   # Iterative LLM (Groq / OpenAI) + one-shot baseline
├── tstb_t5_simplification/    # Iterative T5 (summarize: prefix)
├── tsar_evaluation/           # CEFR compliance + MeaningBERT
├── data_preparation/          # Training corpus construction (thesis data sections)
├── training/                  # T5 fine-tuning (thesis Section Training)
├── t5-base-simplification-*/  # Fine-tuned checkpoints
└── data/                      # Datasets (not modified by code refactors)
```

## Standard TSAR workflow

```bash
cd /Users/mac/Desktop/Thesis
source thesis_venv/bin/activate

# 1) Sentence segmentation + ranking (spaCy, ExpectedScore, document CEFR)
python tstb_impact_score/main.py \
  data/data_for_evaluation/tsar2025_test.jsonl \
  data/data_for_evaluation/tsar2025_expanded.csv

# 2a) LLM iterative simplification (Groq)
cd tstb_llm_simplification
python run_tsar_jsonl.py \
  ../data/data_for_evaluation/tsar2025_test.jsonl \
  ../data/data_for_evaluation/tsar2025_test_llm.jsonl \
  --provider groq --model llama-3.1-8b-instant

# 2b) LLM iterative simplification (OpenAI)
python run_tsar_jsonl.py \
  ../data/data_for_evaluation/tsar2025_test.jsonl \
  ../data/data_for_evaluation/tsar2025_test_gpt.jsonl \
  --provider openai --model gpt-4.1

# 2c) T5 iterative simplification
cd ../tstb_t5_simplification
python run_tsar_jsonl.py \
  ../data/data_for_evaluation/tsar2025_test.jsonl \
  ../data/data_for_evaluation/tsar2025_test_t5.jsonl \
  --model-dir ../t5-base-simplification-res4 \
  --expanded_csv ../data/data_for_evaluation/tsar2025_expanded.csv

# 3) Evaluate submission
cd ..
python -m tsar_evaluation.evaluate \
  data/data_for_evaluation/tsar2025_test_t5.jsonl
```

## One-shot LLM baseline

```bash
cd tstb_llm_simplification
python run_tsar_jsonl_oneshot.py \
  ../data/data_for_evaluation/tsar2025_test.jsonl \
  ../data/data_for_evaluation/tsar2025_test_oneshot.jsonl \
  --provider openai --model gpt-4.1
```

## Environment

- `GROQ_API_KEY`, `OPENAI_API_KEY` in `.env`
- `T5_MODEL_PATH` optional (default checkpoint folder name in `tstb_t5_simplification/main.py`)
- `python -m spacy download en_core_web_sm`

## Training data (thesis Sections Dataset / EDA / Training Data)

Three final CSVs for T5 runs: `data/data_for_training/data_wiki_asset_gpt.csv`, `data_asset_gpt.csv`, `data_simpwiki_asset_gpt.csv`.

Documentation: **`data_preparation/README.md`**

## T5 training (thesis Section Training)

Structured code (for committee / reproduction): **`training/README.md`**

```bash
python training/train.py --run run3 --data-csv YOUR_PAIRS.csv --output-dir ./t5-base-simplification-res4
```

Legacy Colab notebook: `Untitled30.ipynb` → see `training/TRAINING_LEGACY.md`.

## Pipeline notes (thesis alignment)

- **Preprocessing:** spaCy segmentation; sentence complexity = ExpectedScore from `ModernBERT-base-doc_sent_en-Cefr`; document CEFR = best confidence among three ModernBERT models.
- **Iterative control:** shared in `tstb_core/iterative.py`; sentences processed hardest-first.
- **Rollback + mild prompt:** LLM only (T5 has no mild prompt).
- **T5 input prefix:** `summarize: ` (training convention).
