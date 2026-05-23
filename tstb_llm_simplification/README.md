# TSTB LLM Simplification

Iterative sentence simplification via Groq or OpenAI, with rollback and a mild prompt on CEFR overshoot. One-shot baseline in `one_shot_simplifier.py`.

## TSAR run (iterative)

Requires `tsar2025_expanded.csv` from `tstb_impact_score`.

```bash
cd tstb_llm_simplification

# Groq
python run_tsar_jsonl.py IN.jsonl OUT.jsonl --provider groq --model llama-3.1-8b-instant

# OpenAI
python run_tsar_jsonl.py IN.jsonl OUT.jsonl --provider openai --model gpt-4.1
```

Optional: `--expanded_csv PATH`, `--limit N`, `--sleep SEC`.

## One-shot baseline

```bash
python run_tsar_jsonl_oneshot.py IN.jsonl OUT.jsonl --provider openai --model gpt-4.1
```
