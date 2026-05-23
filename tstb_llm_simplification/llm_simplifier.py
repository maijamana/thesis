"""
Sentence-level simplification via OpenAI or Groq APIs.

Uses shared iterative control (ranking, CEFR re-scoring, rollback + mild prompt) from tstb_core.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Literal

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tstb_core.cefr_scorer import CEFRScorer
from tstb_core.iterative import simplify_until_target_iterative
from tstb_core.prompts import ITERATIVE_SYSTEM_MESSAGE, build_iterative_user_prompt
from tstb_core.text_utils import get_original_and_simplified_text

Provider = Literal["groq", "openai"]


def _strip_response(text: str) -> str:
    return text.strip().strip(' "\'"“"''')


def simplify_sentence(
    provider: Provider,
    client,
    target_sentence: str,
    prev_sentence: str = "",
    next_sentence: str = "",
    target_level: str | None = None,
    prompt_style: str = "normal",
    model: str = "llama-3.1-8b-instant",
    max_attempts: int = 5,
) -> str:
    """Call the configured LLM provider to simplify one sentence."""
    prompt = build_iterative_user_prompt(
        target_sentence=target_sentence,
        prev_sentence=prev_sentence,
        next_sentence=next_sentence,
        prompt_style=prompt_style,
        target_level=target_level,
    )
    messages = [
        {"role": "system", "content": ITERATIVE_SYSTEM_MESSAGE},
        {"role": "user", "content": prompt},
    ]

    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            if provider == "groq":
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=200,
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.2,
                    max_completion_tokens=200,
                )
            return _strip_response(response.choices[0].message.content)
        except Exception as e:
            last_err = e
            if attempt == max_attempts:
                break
            sleep_s = 1.0 * (2 ** (attempt - 1))
            print(f"    LLM request failed ({attempt}/{max_attempts}): {e}. Retry in {sleep_s:.1f}s...")
            time.sleep(sleep_s)
    raise last_err  # type: ignore[misc]


def simplify_until_target_iterative_llm(
    text_df: pd.DataFrame,
    text_id: int,
    target_level: str,
    scorer: CEFRScorer,
    client,
    provider: Provider,
    model: str,
    max_iterations: int = 8,
) -> pd.DataFrame:
    """Iterative simplification with LLM rollback on CEFR overshoot."""

    def _simplify(base: str, prev: str, next_: str, prompt_style: str = "normal", target_level: str | None = None) -> str:
        return simplify_sentence(
            provider=provider,
            client=client,
            target_sentence=base,
            prev_sentence=prev,
            next_sentence=next_,
            target_level=target_level,
            prompt_style=prompt_style,
            model=model,
        )

    return simplify_until_target_iterative(
        text_df,
        text_id,
        target_level,
        scorer,
        _simplify,
        max_iterations=max_iterations,
        llm_rollback=True,
    )


__all__ = [
    "simplify_sentence",
    "simplify_until_target_iterative_llm",
    "get_original_and_simplified_text",
]
