"""LLM prompt templates for iterative and one-shot simplification."""


def build_iterative_user_prompt(
    target_sentence: str,
    prev_sentence: str = "",
    next_sentence: str = "",
    prompt_style: str = "normal",
    target_level: str | None = None,
) -> str:
    """Build the user prompt for sentence-level iterative simplification."""
    context_parts = []
    if prev_sentence:
        context_parts.append(f"Previous sentence: {prev_sentence.strip()}")
    if next_sentence:
        context_parts.append(f"Next sentence: {next_sentence.strip()}")
    context = "\n".join(context_parts)

    extra = ""
    if prompt_style == "mild":
        extra = (
            "\nAdditional constraints:\n"
            "- Make ONLY minimal changes. Prefer small lexical swaps and light clause simplification.\n"
            "- Do NOT make the sentence extremely simple.\n"
        )
        if target_level:
            extra += f"- Aim for CEFR {target_level}. Avoid simplifying below {target_level}.\n"
        extra += "- Keep key content words if possible.\n"

    return f"""
Instruction:
You are a professional English text simplification expert.
Your task is to simplify the **Target Sentence (T)**, making it easier to read
while preserving its original meaning and ensuring it fits smoothly within the provided **Context (C)**.

Simplification Guidelines:
1. Break it Down — If the sentence is long or has multiple ideas, split it into shorter ones (2–3 max).
2. Lexical Swap — Replace complex, rare, or academic words with simple, common synonyms.
3. Syntactic Refactoring — Prefer active voice, simplify clauses, reduce long relative constructions.
4. Maintain Cohesion — The result must logically connect with surrounding sentences.
{extra}
Context (C):
{context if context else "No context provided."}

Target Sentence (T):
{target_sentence.strip()}

Return ONLY the simplified version of the Target Sentence. Do NOT include explanations, comments, or notes.
Simplified:
""".strip()


ITERATIVE_SYSTEM_MESSAGE = "You simplify English sentences to easier CEFR levels."

ONE_SHOT_PROMPT_TEMPLATE = """You are an expert text simplification specialist for English language learners.
Your task is to simplify the following text to CEFR level {TARGET_LEVEL},
making it accessible for learners at that proficiency level.

## Simplification rules

- Preserve ALL factual information, numbers, dates, and proper names exactly.
- Do NOT add information that is not in the original text.
- Do NOT remove key ideas — only make them easier to express.
- Keep the original order of events and logical flow.
- Replace difficult vocabulary with simpler, high-frequency alternatives.
- Break long or complex sentences into shorter, clearer ones.
- Use active voice wherever possible.
- Maintain grammatical correctness and natural fluency.
- Do NOT include any meta-commentary, explanations, or task descriptions in your output.

## Target level guidance

For A1: Use only the most basic, everyday words. Very short sentences (Subject–Verb–Object structure). No subordinate clauses. No abstract concepts.

For A2: Use simple, direct SVO sentence structure. Common everyday vocabulary only. Avoid relative clauses and passive constructions. Maximum ~20 words per sentence.

For B1: Use everyday vocabulary; may include some abstract concepts if briefly explained. Simple connectors (because, but, so, although). Present perfect and simple passive are acceptable. Maximum ~29 words per sentence.

## Example

Original:
{EXAMPLE_ORIGINAL}

Simplified ({TARGET_LEVEL}):
{EXAMPLE_SIMPLIFIED}

## Your task

Now simplify the following text to CEFR level {TARGET_LEVEL}.
Return only the simplified text. Do not include any other comments, labels, or explanations.

Original:
{INPUT_TEXT}

Simplified ({TARGET_LEVEL}):
""".strip()

ONE_SHOT_SYSTEM_MESSAGE = "You simplify English texts to a target CEFR level."
