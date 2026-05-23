"""
GPT synthetic pair generation prompts (thesis Section Training Data — Synthetic data).

Used to supplement WikiLarge/ASSET where CEFR reduction is weak (EDA Section).
"""

# Primary generation prompt (Untitled38.ipynb — matches thesis description)
SYSTEM_PROMPT = """
You create high-quality English sentence simplification training data.

Generate pairs of sentences:
1. original: a grammatically complex but natural English sentence.
2. simplified: a simpler version that preserves the original meaning.

Requirements:
- Do not target a specific CEFR level.
- The simplified version must be easier to read.
- Simplification may include syntactic simplification, lexical simplification, and splitting into shorter sentences.
- Do not add new facts.
- Do not remove important meaning.
- Avoid trivial pairs where the simplified sentence is almost identical.
- Avoid very domain-specific facts, names, dates, or obscure entities.
- Use diverse everyday, academic, social, scientific, cultural, and work-related topics.
- Each original sentence should be one sentence.
- The simplified version may be one or two sentences.
- Return only valid JSON.
""".strip()

USER_PROMPT_TEMPLATE = """
Generate {n} sentence simplification pairs.

Return JSON in this exact format:
{{
  "pairs": [
    {{
      "original": "...",
      "simplified": "...",
      "simplification_type": "syntactic|lexical|mixed",
      "quality_note": "short explanation"
    }}
  ]
}}
""".strip()
