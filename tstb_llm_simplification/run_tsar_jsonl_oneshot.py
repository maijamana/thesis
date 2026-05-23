#!/usr/bin/env python3
"""
One-shot TSAR runner (Groq llama-* or OpenAI gpt-*).

Uses a NEIGHBOR example (original->reference) as a one-shot demonstration.
Never uses the current record as its own example.

Usage:
  python run_tsar_jsonl_oneshot.py <input.jsonl> <output.jsonl> --provider openai --model gpt-4.1
  python run_tsar_jsonl_oneshot.py <input.jsonl> <output.jsonl> --provider groq   --model llama-3.3-70b-versatile
Optional:
  --limit N
"""

import sys
from pathlib import Path

from one_shot_simplifier import OneShotConfig, run_tsar_oneshot


def main():
    if len(sys.argv) < 5:
        print("Usage: python run_tsar_jsonl_oneshot.py <input.jsonl> <output.jsonl> --provider (openai|groq) --model MODEL [--limit N]")
        sys.exit(1)

    input_jsonl = sys.argv[1]
    output_jsonl = sys.argv[2]

    provider = None
    model = None
    limit = None

    argv = sys.argv[3:]
    i = 0
    while i < len(argv):
        if argv[i] == "--provider" and i + 1 < len(argv):
            provider = argv[i + 1]
            i += 2
            continue
        if argv[i] == "--model" and i + 1 < len(argv):
            model = argv[i + 1]
            i += 2
            continue
        if argv[i] == "--limit" and i + 1 < len(argv):
            limit = int(argv[i + 1])
            i += 2
            continue
        i += 1

    if provider not in ("openai", "groq"):
        raise ValueError("--provider must be 'openai' or 'groq'")
    if not model:
        raise ValueError("--model is required")

    if not Path(input_jsonl).exists():
        raise FileNotFoundError(input_jsonl)

    cfg = OneShotConfig(provider=provider, model=model)
    run_tsar_oneshot(input_jsonl=input_jsonl, output_jsonl=output_jsonl, cfg=cfg, limit=limit)


if __name__ == "__main__":
    main()

