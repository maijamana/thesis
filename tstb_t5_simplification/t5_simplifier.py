"""
Sentence simplification with a fine-tuned T5 model (prefix: summarize:).

Uses the shared iterative loop from tstb_core without LLM-style rollback.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tstb_core.cefr_scorer import CEFRScorer
from tstb_core.constants import T5_TASK_PREFIX
from tstb_core.iterative import simplify_until_target_iterative
from tstb_core.text_utils import get_original_and_simplified_text


def load_t5_model(model_path: str):
    """Load T5 weights and tokenizer from a local checkpoint directory."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    model.eval()
    return model, tokenizer, device


def simplify_sentence_t5(
    sentence: str,
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    device: torch.device,
    max_length: int = 256,
    num_beams: int = 4,
) -> str:
    """Simplify one sentence; input is prefixed with summarize: per training setup."""
    if not sentence or not str(sentence).strip():
        return sentence
    input_text = T5_TASK_PREFIX + sentence.strip()
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
        )
    simplified = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return simplified.strip().strip(' "\'"')


def simplify_until_target_iterative_t5(
    text_df: pd.DataFrame,
    text_id: int,
    target_level: str,
    scorer: CEFRScorer,
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    device: torch.device,
    max_iterations: int = 8,
) -> pd.DataFrame:
    """Iterative simplification with T5 (no mild-prompt rollback)."""

    def _simplify(base: str, _prev: str, _next: str) -> str:
        return simplify_sentence_t5(base, model, tokenizer, device)

    return simplify_until_target_iterative(
        text_df,
        text_id,
        target_level,
        scorer,
        _simplify,
        max_iterations=max_iterations,
        llm_rollback=False,
    )


__all__ = [
    "load_t5_model",
    "simplify_sentence_t5",
    "simplify_until_target_iterative_t5",
    "get_original_and_simplified_text",
]
