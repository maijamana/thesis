"""
Training corpus presets (thesis Section Training Data Construction).

Canonical outputs already built for this project — see FINAL_CORPORA below.
"""

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_TRAINING = REPO_ROOT / "data" / "data_for_training"
DATA_ROOT = REPO_ROOT / "data"

CEFR_MODEL_ID = "AbdullahBarayan/ModernBERT-base-doc_sent_en-Cefr"
MEANINGBERT_MODEL_ID = "davebulaval/meaningbert"

# Thesis filtering thresholds
WIKI_CEFR_THRESHOLD_RUN1 = 0.8
WIKI_CEFR_THRESHOLD_RUN3 = 1.3
WIKI_MEANINGBERT_MIN = 0.65
ASSET_CEFR_THRESHOLD = 0.8


@dataclass(frozen=True)
class CorpusBuildConfig:
    """How to assemble one T5 training corpus (Run 1–3)."""

    name: str
    description: str
    include_wiki: bool
    include_asset: bool
    include_gpt: bool
    wiki_cefr_min: float
    wiki_meaning_min: float | None  # None = skip MeaningBERT filter (not used for ASSET/GPT)
    asset_cefr_min: float
    expected_pairs_note: str
    canonical_output: Path


CORPUS_CONFIGS: dict[str, CorpusBuildConfig] = {
    "run1": CorpusBuildConfig(
        name="run1",
        description="WikiLarge (CEFR Δ≥0.8, MeaningBERT>0.65) + ASSET (CEFR Δ≥0.8) + GPT",
        include_wiki=True,
        include_asset=True,
        include_gpt=True,
        wiki_cefr_min=WIKI_CEFR_THRESHOLD_RUN1,
        wiki_meaning_min=WIKI_MEANINGBERT_MIN,
        asset_cefr_min=ASSET_CEFR_THRESHOLD,
        expected_pairs_note="~17,430 pairs (thesis Table)",
        canonical_output=DATA_TRAINING / "data_wiki_asset_gpt.csv",
    ),
    "run2": CorpusBuildConfig(
        name="run2",
        description="ASSET (CEFR Δ≥0.8) + GPT only",
        include_wiki=False,
        include_asset=True,
        include_gpt=True,
        wiki_cefr_min=WIKI_CEFR_THRESHOLD_RUN1,
        wiki_meaning_min=None,
        asset_cefr_min=ASSET_CEFR_THRESHOLD,
        expected_pairs_note="~3,414 pairs (thesis Table)",
        canonical_output=DATA_TRAINING / "data_asset_gpt.csv",
    ),
    "run3": CorpusBuildConfig(
        name="run3",
        description="WikiLarge (CEFR Δ≥1.3, MeaningBERT>0.65) + ASSET (CEFR Δ≥0.8) + GPT",
        include_wiki=True,
        include_asset=True,
        include_gpt=True,
        wiki_cefr_min=WIKI_CEFR_THRESHOLD_RUN3,
        wiki_meaning_min=WIKI_MEANINGBERT_MIN,
        asset_cefr_min=ASSET_CEFR_THRESHOLD,
        expected_pairs_note="~7,291 pairs (thesis Table)",
        canonical_output=DATA_TRAINING / "data_simpwiki_asset_gpt.csv",
    ),
}

# Default intermediate paths (large files; optional for rebuild)
DEFAULT_PATHS = {
    "wiki_scored_with_meaning": DATA_TRAINING / "full_meaning_filtered_wiki.csv",
    "wiki_scored_cefr_only": DATA_TRAINING / "full_aligned_pairs_cefr_filtered_wiki_0_6.csv",
    "wiki_compact": DATA_TRAINING / "meaning_filtered_wiki.csv",
    "asset_scored": DATA_ROOT / "intermediate" / "full_merged_aligned_pairs_cefr_assetsimp3.csv",
    "asset_orig": DATA_TRAINING / "asset.valid.orig",
    "asset_simp": DATA_TRAINING / "asset.valid.simp.0",
    "gpt_pairs": DATA_ROOT / "gpt_simplification_dataset_augmented_v2.csv",
    "wiki_aligned_raw": DATA_TRAINING / "aligned_pairs.csv",
    "wiki_simplewiki_jsonl": DATA_TRAINING / "simplewiki-en_sentences.jsonl",
}
