# T5 Sentence Simplification — Training

Fine-tuning **T5-base** for readability-controlled sentence simplification, aligned with **Section Training** of the thesis.

Original experiments were run in Colab (`Untitled30.ipynb` at repo root). This folder is the **structured, review-ready** version of that workflow.

## What the model learns

| Item | Value |
|------|--------|
| Base model | `t5-base` |
| Task prefix | `summarize: ` (T5 text-to-text convention; same at train and inference) |
| Input | complex sentence |
| Target | simplified sentence |
| Max length | 128 tokens (input and output, padded per batch) |

## Training runs (thesis Table)

Three runs differ in **data mix** and **hyperparameters**. Model selection uses **lowest `eval_loss`** with **early stopping (patience 4)**.

| | Run 1 | Run 2 | Run 3 |
|---|-------|-------|-------|
| Training pairs | 17,430 | 3,414 | 7,291 |
| Data | WikiLarge + ASSET + GPT | ASSET + GPT | WikiLarge (CEFR Δ > 1.3) + ASSET + GPT |
| Learning rate | 5×10⁻⁵ | 3×10⁻⁵ | 3×10⁻⁵ |
| Warmup steps | 500 | 50 | 100 |
| Max epochs | 4 | 4 | 4 |
| Batch size (per device) | 16 | 16 | 16 |
| Weight decay | 0.01 | 0.01 | 0.01 |
| Mixed precision | fp16 | fp16 | fp16 |
| Best checkpoint | lowest `eval_loss` | lowest `eval_loss` | lowest `eval_loss` |

Checkpoints for thesis Runs 1–3:

| Run | Directory |
|-----|-----------|
| 1 | `t5-base-simplification-wikilarge_asset_gpt/` |
| 2 | `t5-base-simplification-asset_gpt/` |
| 3 | `t5-base-simplification-wikilarge_simple_asset_gpt/` |

## Custom evaluation during training

Standard BLEU/ROUGE are **not** used for checkpointing. Each validation epoch reports:

1. **`cefr_diff`** — mean reduction in continuous CEFR score (source − prediction), using `ModernBERT-base-doc_sent_en-Cefr` and the same expected-score formula as inference (Eq. in thesis).
2. **`cefr_close1`** — share of examples with \|diff − 1.0\| < 0.5 (≈ one CEFR level simpler).
3. **`sim_orig`** — mean character-level `SequenceMatcher` similarity (lightweight content-preservation signal).

Decoding for metrics uses **greedy argmax** on logits (`preprocess_logits_for_metrics`), not beam search, to keep evaluation fast.

## Project layout

```
training/
├── README.md                 # this file
├── requirements.txt
├── config.py                 # RunConfig presets (run1 / run2 / run3)
├── cefr_metrics.py           # CEFR expected level + compute_metrics
├── data.py                   # CSV load, train/valid/test split
├── train.py                  # CLI entry point
└── notebooks/
    └── T5_training_overview.ipynb   # short walkthrough for the committee
```

## Usage

```bash
cd /Users/mac/Desktop/Thesis
source thesis_venv/bin/activate
pip install -r training/requirements.txt

# Example: Run 3-style training (adjust --data-csv to your filtered CSV)
python training/train.py \
  --run run3 \
  --data-csv data/data_for_training/your_train_pairs.csv \
  --output-dir ./t5-base-simplification-res4
```

### CLI options

| Flag | Description |
|------|-------------|
| `--run` | Preset: `run1`, `run2`, `run3` (overrides lr/warmup unless `--learning-rate` set) |
| `--data-csv` | CSV with columns `original`, `simplified` |
| `--output-dir` | Where to save checkpoints |
| `--max-epochs` | Override epoch count |
| `--learning-rate` | Override learning rate |
| `--warmup-steps` | Override warmup |
| `--no-fp16` | Disable mixed precision |

## Data preparation (outside this folder)

Training CSVs are built in `data/data_for_training/` (filtering WikiLarge by CEFR difference, merging ASSET and GPT-augmented pairs). **This code does not modify data** — it only reads a prepared CSV.

## Relation to inference

Inference uses `tstb_t5_simplification/` with prefix `summarize: ` (`tstb_core.constants.T5_TASK_PREFIX`). Training and inference share the same prefix and CEFR scoring definition.

## Note on `Untitled30.ipynb`

The Colab notebook contains:

- **Primary pipeline** — `summarize: ` prefix, full training loop (matches thesis).
- **Discarded experiments** — e.g. second-stage fine-tune with `simplify: ` prefix on a small ASSET subset; not used in the final system.

For the defense, use **`training/`** and `notebooks/T5_training_overview.ipynb`, not the raw Colab export.
