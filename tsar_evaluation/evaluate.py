#!/usr/bin/env python3
"""
Evaluation for TSAR 2025 Shared Task on Readability-Controlled Text Simplification.

- CEFR compliance: 3 ModernBERT-based labelers, best-by-confidence; weighted_f1, adj_accuracy, rmse.
- Meaning preservation: MeaningBERT (simplified vs original, simplified vs reference).
- No BERTScore. Batch processing for speed.
"""
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import evaluate
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error as mse
from transformers import pipeline


CEFR_LABELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
LABEL2IDX = {label: idx for idx, label in enumerate(CEFR_LABELS)}

DEFAULT_CEFR_BATCH_SIZE = 32
DEFAULT_MEANINGBERT_BATCH_SIZE = 32


def read_jsonl(filepath: str) -> list[dict[str, Any]]:
    """Read JSONL file; each line is a JSON object."""
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_models(
    cefr_batch_size: int = DEFAULT_CEFR_BATCH_SIZE,
):
    """
    Load CEFR labelers and MeaningBERT once.
    Returns (cefr_models_list, meaning_bert_module).
    """
    print("Loading CEFR labelers...")
    cefr_labeler1 = pipeline(
        task="text-classification",
        model="AbdullahBarayan/ModernBERT-base-doc_en-Cefr",
    )
    cefr_labeler2 = pipeline(
        task="text-classification",
        model="AbdullahBarayan/ModernBERT-base-doc_sent_en-Cefr",
    )
    cefr_labeler3 = pipeline(
        task="text-classification",
        model="AbdullahBarayan/ModernBERT-base-reference_AllLang2-Cefr2",
    )
    cefr_models = [cefr_labeler1, cefr_labeler2, cefr_labeler3]
    print("Loading MeaningBERT...")
    meaning_bert = evaluate.load("davebulaval/meaningbert")
    return cefr_models, meaning_bert


def _get_cefr_labels_batched(
    simplifications: list[str],
    models: list,
    batch_size: int = DEFAULT_CEFR_BATCH_SIZE,
) -> list[str]:
    """
    For each text, run all 3 CEFR models and take the label with highest confidence.
    Process texts in batches per model to avoid OOM.
    """
    n = len(simplifications)
    # Collect per-model predictions: list of (label, score) per item
    all_preds = [[] for _ in range(n)]  # all_preds[i] = [(label, score), ...] from each model

    for model in models:
        for start in range(0, n, batch_size):
            batch = simplifications[start : start + batch_size]
            out = model(batch)
            if not isinstance(out, list):
                out = [out]
            for i, item in enumerate(out):
                idx = start + i
                if idx < n:
                    label = item["label"] if isinstance(item, dict) else item[0]["label"]
                    score = item["score"] if isinstance(item, dict) else item[0]["score"]
                    # Normalize label to CEFR (e.g. strip + uppercase)
                    label = str(label).strip().upper()
                    if label not in LABEL2IDX:
                        label = "B1"
                    all_preds[idx].append((label, float(score)))

    return [max(preds, key=lambda x: x[1])[0] for preds in all_preds]


def get_cefr_compliance_score(
    simplifications: list[str],
    reference_levels: list[str],
    models: list,
    batch_size: int = DEFAULT_CEFR_BATCH_SIZE,
) -> dict[str, float]:
    """
    CEFR compliance: weighted_f1, adj_accuracy (within 1 level), rmse.
    reference_levels = target_cefr for each item.
    """
    assert len(simplifications) == len(reference_levels)
    ref_normalized = [str(l).strip().upper() for l in reference_levels]
    for r in ref_normalized:
        if r not in LABEL2IDX:
            raise ValueError(f"Unknown CEFR level in reference_levels: {r}")

    predicted_labels = _get_cefr_labels_batched(simplifications, models, batch_size=batch_size)
    f1 = f1_score(ref_normalized, predicted_labels, average="weighted", zero_division=0)

    true_idx = np.array([LABEL2IDX[l] for l in ref_normalized])
    pred_idx = np.array([LABEL2IDX.get(l, 0) for l in predicted_labels])

    adj_acc = (np.abs(true_idx - pred_idx) <= 1).mean()
    rmse = float(np.sqrt(mse(true_idx, pred_idx)))

    return {
        "weighted_f1": round(f1, 4),
        "adj_accuracy": round(adj_acc, 4),
        "rmse": round(rmse, 4),
    }


def _get_meaningbert_score_batched(
    predictions: list[str],
    references: list[str],
    model,
    batch_size: int = DEFAULT_MEANINGBERT_BATCH_SIZE,
) -> float:
    """MeaningBERT similarity in batches; returns mean score (0–1 scale, as in TSAR script)."""
    assert len(predictions) == len(references)
    scores = []
    for start in range(0, len(predictions), batch_size):
        batch_pred = predictions[start : start + batch_size]
        batch_ref = references[start : start + batch_size]
        result = model.compute(predictions=batch_pred, references=batch_ref)
        batch_scores = result["scores"]
        scores.extend(batch_scores)
    return round(float(np.mean(scores)) / 100.0, 4)


def evaluate_submission(
    submission_data: list[dict],
    ref_data: list[dict] | None = None,
    cefr_models: list | None = None,
    meaning_bert=None,
    cefr_batch_size: int = DEFAULT_CEFR_BATCH_SIZE,
    meaningbert_batch_size: int = DEFAULT_MEANINGBERT_BATCH_SIZE,
) -> dict[str, Any]:
    """
    submission_data: list of dicts with at least 'text_id', 'simplified'.
    ref_data: list of dicts with 'text_id', 'original', 'reference', 'target_cefr'.
    If ref_data is None, submission_data must contain original, reference, target_cefr (e.g. single JSONL with everything).
    """
    if ref_data is None:
        ref_data = submission_data
    # Align by text_id
    ref_by_id = {r["text_id"]: r for r in ref_data}
    simplified_texts = []
    original_texts = []
    reference_texts = []
    target_cefr_levels = []
    for entry in submission_data:
        tid = entry["text_id"]
        if tid not in ref_by_id:
            continue
        ref = ref_by_id[tid]
        simplified_texts.append(entry["simplified"])
        original_texts.append(ref["original"])
        reference_texts.append(ref["reference"])
        target_cefr_levels.append(ref["target_cefr"].upper() if isinstance(ref["target_cefr"], str) else ref["target_cefr"])

    if not simplified_texts:
        raise ValueError("No matching text_id between submission and reference.")

    load_here = cefr_models is None or meaning_bert is None
    if load_here:
        cefr_models, meaning_bert = load_models()

    # CEFR compliance (batched)
    print("Computing CEFR compliance (batched)...")
    compliance = get_cefr_compliance_score(
        simplified_texts, target_cefr_levels, cefr_models, batch_size=cefr_batch_size
    )

    # Meaning preservation: simplified vs original (batched)
    print("Computing MeaningBERT (simplified vs original, batched)...")
    meaningbert_org = _get_meaningbert_score_batched(
        simplified_texts, original_texts, meaning_bert, batch_size=meaningbert_batch_size
    )
    # Similarity to reference (batched)
    print("Computing MeaningBERT (simplified vs reference, batched)...")
    meaningbert_ref = _get_meaningbert_score_batched(
        simplified_texts, reference_texts, meaning_bert, batch_size=meaningbert_batch_size
    )

    return {
        "CEFR_compliance": compliance,
        "Similarity_to_Original": {"MeaningBERT_Org": meaningbert_org},
        "Similarity_to_Reference": {"MeaningBERT_Ref": meaningbert_ref},
    }


def run_from_cli(
    submission_path: str,
    reference_path: str | None = None,
    cefr_batch_size: int = DEFAULT_CEFR_BATCH_SIZE,
    meaningbert_batch_size: int = DEFAULT_MEANINGBERT_BATCH_SIZE,
) -> dict[str, Any]:
    """
    Read submission and (optional) reference JSONL, run evaluation, print and return results.
    If reference_path is None, submission_path is used as both (file must have original, reference, target_cefr, simplified).
    """
    submission_data = read_jsonl(submission_path)
    ref_data = read_jsonl(reference_path) if reference_path else None
    if ref_data is None:
        ref_data = submission_data
    print(f"Submission: {len(submission_data)} rows. Reference: {len(ref_data)} rows.")
    results = evaluate_submission(
        submission_data,
        ref_data=ref_data,
        cefr_batch_size=cefr_batch_size,
        meaningbert_batch_size=meaningbert_batch_size,
    )
    print("\n--- TSAR 2025 Evaluation Results ---")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m tsar_evaluation.evaluate <submission.jsonl> [reference.jsonl]")
        print("  submission.jsonl: must have text_id, simplified.")
        print("  reference.jsonl: must have text_id, original, reference, target_cefr.")
        print("  If reference.jsonl is omitted, submission file is used as reference (single file with all fields).")
        print("  Optional env: CEFR_BATCH_SIZE, MEANINGBERT_BATCH_SIZE (default 32).")
        sys.exit(1)
    submission_path = sys.argv[1]
    reference_path = sys.argv[2] if len(sys.argv) > 2 else None
    if not Path(submission_path).exists():
        print(f"File not found: {submission_path}")
        sys.exit(1)
    if reference_path and not Path(reference_path).exists():
        print(f"File not found: {reference_path}")
        sys.exit(1)

    import os
    cefr_batch = int(os.environ.get("CEFR_BATCH_SIZE", DEFAULT_CEFR_BATCH_SIZE))
    mb_batch = int(os.environ.get("MEANINGBERT_BATCH_SIZE", DEFAULT_MEANINGBERT_BATCH_SIZE))
    run_from_cli(submission_path, reference_path, cefr_batch_size=cefr_batch, meaningbert_batch_size=mb_batch)


if __name__ == "__main__":
    main()
