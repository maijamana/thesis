"""
CEFR-based training metrics (thesis Section Training).

Uses ModernBERT sentence classifier and continuous expected CEFR level,
identical to inference-time ExpectedScore definition in tstb_core.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Callable

import numpy as np
import torch
from transformers import pipeline

CEFR_ORDER = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}
CEFR_MODEL_ID = "AbdullahBarayan/ModernBERT-base-doc_sent_en-Cefr"


def build_cefr_labeler(device: int | str = 0):
    """Load the sentence-level CEFR classifier used during training evaluation."""
    return pipeline(
        task="text-classification",
        model=CEFR_MODEL_ID,
        top_k=None,
        device=device,
    )


def cefr_expected_level(texts: list[str], labeler, batch_size: int = 32) -> list[float]:
    """
    Continuous expected CEFR score: sum over labels of ord(label) * P(label | text).
    """
    texts = [t if t and t.strip() else "simple" for t in texts]
    batch_preds = labeler(texts, truncation=True, batch_size=batch_size)
    return [
        sum(CEFR_ORDER[p["label"]] * float(p["score"]) for p in preds)
        for preds in batch_preds
    ]


def make_compute_metrics(
    tokenizer,
    orig_valid_texts: list[str],
    cefr_labeler,
) -> Callable:
    """
    Hugging Face Trainer compute_metrics with cefr_diff, cefr_close1, sim_orig.

    Uses greedy argmax decoding via preprocess_logits_for_metrics (fast eval).
    """

    def preprocess_logits_for_metrics(logits, labels):
        pred_ids = torch.argmax(logits[0], dim=-1)
        return pred_ids, labels

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred.predictions[0], eval_pred.label_ids

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_preds = [p.strip() if p.strip() else "simple" for p in decoded_preds]

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        tokenizer.batch_decode(labels, skip_special_tokens=True)

        sims = [
            SequenceMatcher(None, o.lower(), p.lower()).ratio()
            for o, p in zip(orig_valid_texts, decoded_preds)
        ]
        avg_sim = float(np.mean(sims))

        orig_levels = cefr_expected_level(orig_valid_texts, cefr_labeler)
        simp_levels = cefr_expected_level(decoded_preds, cefr_labeler)
        diffs = [o - s for o, s in zip(orig_levels, simp_levels)]
        avg_diff = float(np.mean(diffs))
        close_to_1 = float(np.mean([abs(d - 1.0) < 0.5 for d in diffs]))

        return {
            "cefr_diff": round(avg_diff, 4),
            "cefr_close1": round(close_to_1, 4),
            "sim_orig": round(avg_sim, 4),
        }

    return compute_metrics, preprocess_logits_for_metrics
