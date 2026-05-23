# Training Data Construction

Pipeline for building T5 fine-tuning corpora, aligned with the thesis (Sections *Dataset Overview*, *EDA*, *Training Data Construction*).

**EDA figures** (CEFR diff / MeaningBERT histograms) are in the thesis — no plotting code here.

## Canonical training files (use these for `training/train.py`)

| File | Thesis run | Approx. size | Contents |
|------|------------|--------------|----------|
| `data/data_for_training/data_wiki_asset_gpt.csv` | **Run 1** | 17,430 | WikiLarge + ASSET + GPT |
| `data/data_for_training/data_asset_gpt.csv` | **Run 2** | 3,414 | ASSET + GPT only |
| `data/data_for_training/data_simpwiki_asset_gpt.csv` | **Run 3** | 7,291 | WikiLarge (stricter) + ASSET + GPT |

Columns: `original`, `simplified` (plus optional index column from pandas export).

```bash
python training/train.py --run run1 \
  --data-csv data/data_for_training/data_wiki_asset_gpt.csv \
  --output-dir ./t5-base-simplification-res4
```

## Data sources (thesis Table)

| Source | Role | Filtering in this work |
|--------|------|-------------------------|
| **WikiLarge** | Training (after filtering) | CEFR expected-level diff ≥ threshold; MeaningBERT > 0.65 |
| **ASSET** (validation split) | Training pairs | CEFR diff ≥ 0.8 only (crowdsourced → no MeaningBERT filter) |
| **GPT synthetic** | Training supplement | No extra filtering |
| **TSAR 2025** | **Evaluation only** | Not used for training |

### WikiLarge thresholds

| Run | CEFR diff (orig − simp) | MeaningBERT |
|-----|-------------------------|-------------|
| Run 1 | ≥ **0.8** | > **0.65** |
| Run 3 | ≥ **1.3** | > **0.65** |

### Preprocessing (WikiLarge only)

- Extract aligned complex/simple sentence pairs (Wikipedia ↔ Simple Wikipedia).
- Clean markup tokens (`-LRB-` → `(`, etc.) and encoding artifacts — `wiki_clean.py` / `scripts/clean_wiki_artifacts.py`.
- Score each sentence with `ModernBERT-base-doc_sent_en-Cefr` → continuous expected CEFR level.
- Compute MeaningBERT similarity (simplified vs original).
- Apply filters above.

### ASSET

- Line-aligned files: `asset.valid.orig`, `asset.valid.simp.0` (validation split, 2,000 sentences × references).
- Score CEFR → keep pairs with `cefr_diff ≥ 0.8`.

### GPT synthetic data

- Generated with GPT-5.1-class model using a fixed system prompt (diverse topics, preserve meaning, no target CEFR).
- Prompt text: `gpt_prompts.py` (from Colab log `Untitled38.ipynb`).
- Final table used in merges: `data/gpt_simplification_dataset_augmented_v2.csv` (~2,798 pairs).

## Pipeline diagram (thesis Figure: preprocessing)

```
WikiLarge → clean markup → CEFR score → filter (Δ, MeaningBERT) ──┐
ASSET val  → CEFR score → filter (Δ only) ─────────────────────────┼→ merge → dedupe → Run CSV
GPT pairs  (no filter) ───────────────────────────────────────────┘
```

## Intermediate files (large; optional)

| Path | Description |
|------|-------------|
| `aligned_pairs.csv` | Raw Wiki-aligned pairs (~1M rows) |
| `full_aligned_pairs_cefr_filtered_wiki_0_6.csv` | Wiki + CEFR scores (broad pre-filter) |
| `full_meaning_filtered_wiki.csv` | Wiki + CEFR + MeaningBERT (~70k) |
| `meaning_filtered_wiki.csv` | Smaller Wiki subset with scores (~7k) |
| `intermediate/full_merged_aligned_pairs_cefr_assetsimp3.csv` | ASSET + CEFR scores |
| `gpt_simplification_dataset_augmented_v2.csv` | Final GPT synthetic set |

You do **not** need these to train if you use the three canonical `data_*.csv` files.

## Scripts

| Script | Purpose |
|--------|---------|
| `build_corpus.py` | Merge wiki + asset + gpt into Run 1/2/3 |
| `score_pairs.py` | Add `cefr_orig`, `cefr_simp`, `cefr_diff` |
| `filter_wiki.py` | Apply wiki CEFR + MeaningBERT thresholds |
| `wiki_clean.py` | Wikipedia markup cleanup |
| `gpt_prompts.py` | Generation prompt constants |

Legacy one-off scripts remain under `scripts/` (same logic, older layout).

### Rebuild example (verify counts)

```bash
python data_preparation/build_corpus.py --run run1 \
  --wiki-csv data/data_for_training/full_meaning_filtered_wiki.csv \
  --asset-csv data/intermediate/full_merged_aligned_pairs_cefr_assetsimp3.csv \
  --gpt-csv data/gpt_simplification_dataset_augmented_v2.csv \
  --output _rebuilt_run1.csv --verify
```

Row counts may differ slightly from canonical files if intermediate snapshots differ; canonical CSVs are the reference for reported thesis numbers.

## Legacy Colab

`Untitled38.ipynb` — GPT generation experiments (see `DATA_LEGACY.md`).

## Related modules

- `training/` — T5 fine-tuning on these CSVs  
- `scripts/compute_meaningbert_aligned.py` — batch MeaningBERT for pair tables  
- `scripts/compute_cefr_diff_csv.py` — batch CEFR scoring  
- `tstb_impact_score/` — unrelated to training data; used at **inference** for TSAR
