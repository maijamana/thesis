# TSTB Impact Score

Inference-time preprocessing: spaCy sentence segmentation, continuous ExpectedScore per sentence, document-level CEFR (three-model ensemble), ranking for simplification.

## Usage

```bash
cd /Users/mac/Desktop/Thesis
source thesis_venv/bin/activate

python tstb_impact_score/main.py \
  data/data_for_evaluation/tsar2025_test.jsonl \
  data/data_for_evaluation/tsar2025_expanded.csv
```

Output CSV columns include `text_id`, `sentence_index`, `sentence`, `ExpectedScore`, `original_cefr_level`, `impact_score`.

This file is required by both `tstb_llm_simplification/run_tsar_jsonl.py` and `tstb_t5_simplification/run_tsar_jsonl.py`.
