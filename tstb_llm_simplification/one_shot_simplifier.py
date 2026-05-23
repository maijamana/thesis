"""
One-shot text simplification for TSAR-style JSONL (neighbor original→reference example).
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tstb_core.constants import CEFR_LABELS
from tstb_core.io import load_tsar_jsonl, write_tsar_jsonl
from tstb_core.prompts import ONE_SHOT_PROMPT_TEMPLATE, ONE_SHOT_SYSTEM_MESSAGE

Provider = Literal["openai", "groq"]


@dataclass
class OneShotConfig:
    provider: Provider
    model: str
    temperature: float = 0.2
    max_output_tokens: int = 700
    max_attempts: int = 5
    base_sleep_s: float = 1.0


def _pick_neighbor_example(records: list[dict[str, Any]], idx: int) -> tuple[str, str]:
    """Use previous record's original→reference pair, else next."""
    if idx > 0:
        ex = records[idx - 1]
    elif idx + 1 < len(records):
        ex = records[idx + 1]
    else:
        ex = records[idx]
    return str(ex.get("original", "")).strip(), str(ex.get("reference", "")).strip()


def build_prompt(
    target_level: str,
    example_original: str,
    example_simplified: str,
    input_text: str,
) -> str:
    return ONE_SHOT_PROMPT_TEMPLATE.format(
        TARGET_LEVEL=target_level,
        EXAMPLE_ORIGINAL=example_original,
        EXAMPLE_SIMPLIFIED=example_simplified,
        INPUT_TEXT=input_text,
    )


def _call_openai(prompt: str, cfg: OneShotConfig) -> str:
    from openai import OpenAI, APIError, APITimeoutError, RateLimitError

    load_dotenv()
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    last_err: Exception | None = None
    for attempt in range(1, cfg.max_attempts + 1):
        try:
            resp = client.chat.completions.create(
                model=cfg.model,
                messages=[
                    {"role": "system", "content": ONE_SHOT_SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ],
                temperature=cfg.temperature,
                max_completion_tokens=cfg.max_output_tokens,
            )
            return resp.choices[0].message.content.strip()
        except (APITimeoutError, RateLimitError, APIError, Exception) as e:
            last_err = e
        if attempt < cfg.max_attempts:
            time.sleep(cfg.base_sleep_s * (2 ** (attempt - 1)))
    raise last_err  # type: ignore[misc]


def _call_groq(prompt: str, cfg: OneShotConfig) -> str:
    from groq import Groq

    load_dotenv()
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    last_err: Exception | None = None
    for attempt in range(1, cfg.max_attempts + 1):
        try:
            resp = client.chat.completions.create(
                model=cfg.model,
                messages=[
                    {"role": "system", "content": ONE_SHOT_SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ],
                temperature=cfg.temperature,
                max_tokens=cfg.max_output_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
        if attempt < cfg.max_attempts:
            time.sleep(cfg.base_sleep_s * (2 ** (attempt - 1)))
    raise last_err  # type: ignore[misc]


def simplify_one_shot_text(
    input_text: str,
    target_level: str,
    example_original: str,
    example_simplified: str,
    cfg: OneShotConfig,
) -> str:
    prompt = build_prompt(target_level, example_original, example_simplified, input_text)
    if cfg.provider == "openai":
        return _call_openai(prompt, cfg)
    if cfg.provider == "groq":
        return _call_groq(prompt, cfg)
    raise ValueError(f"Unknown provider: {cfg.provider}")


def run_tsar_oneshot(
    input_jsonl: str,
    output_jsonl: str,
    cfg: OneShotConfig,
    limit: int | None = None,
    sleep_s: float = 0.0,
) -> None:
    records = load_tsar_jsonl(input_jsonl)
    if limit is not None:
        records = records[:limit]

    out: list[dict[str, Any]] = []
    for i, rec in enumerate(records):
        target = str(rec.get("target_cefr", "A2")).strip().upper()
        if target not in CEFR_LABELS:
            target = "A2"
        ex_orig, ex_simp = _pick_neighbor_example(records, i)
        print(f"[{i+1}/{len(records)}] text_id={rec.get('text_id')} target={target}")
        try:
            simplified = simplify_one_shot_text(
                str(rec.get("original", "")).strip(),
                target,
                ex_orig,
                ex_simp,
                cfg,
            )
        except Exception as e:
            print(f"  Error: {e}")
            simplified = ""
        new_rec = dict(rec)
        new_rec["simplified"] = simplified
        out.append(new_rec)
        if sleep_s:
            time.sleep(sleep_s)

    write_tsar_jsonl(output_jsonl, out)
    print(f"Saved {len(out)} records to {output_jsonl}")
