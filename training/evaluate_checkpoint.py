#!/usr/bin/env python3
"""
Post-training qualitative check: simplify sentences and report CEFR + similarity.

Matches the manual test cells in the original Colab notebook (without duplicating training).
"""

from __future__ import annotations

import argparse
import sys
from difflib import SequenceMatcher
from pathlib import Path

_TRAIN_DIR = Path(__file__).resolve().parent
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from config import TASK_PREFIX
from cefr_metrics import build_cefr_labeler, cefr_expected_level


SAMPLE_SENTENCES = [
    "The proliferation of misinformation on social media platforms poses significant challenges to democratic discourse.",
    "The government announced new measures to reduce carbon emissions over the next decade.",
    "The local community organized a fundraising event to support families affected by the flood.",
]


def simplify(model, tokenizer, text: str, device: str, max_length: int = 128) -> str:
    inputs = tokenizer(
        TASK_PREFIX + text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = T5Tokenizer.from_pretrained(args.model_dir)
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir).to(device)
    model.eval()
    labeler = build_cefr_labeler(device=0 if device == "cuda" else -1)

    for sent in SAMPLE_SENTENCES:
        pred = simplify(model, tokenizer, sent, device, args.max_length)
        o = cefr_expected_level([sent], labeler)[0]
        s = cefr_expected_level([pred], labeler)[0]
        sim = SequenceMatcher(None, sent.lower(), pred.lower()).ratio()
        print(f"\nOriginal   ({o:.2f}): {sent}")
        print(f"Simplified ({s:.2f}): {pred}")
        print(f"Diff {o - s:+.2f} | Sim {sim:.3f}")


if __name__ == "__main__":
    main()
