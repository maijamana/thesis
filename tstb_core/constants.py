"""Shared constants for CEFR levels and model prefixes."""

CEFR_LABELS = ("A1", "A2", "B1", "B2", "C1", "C2")
CEFR_ORDER = list(CEFR_LABELS)
LEVEL_MAP = {label: idx + 1 for idx, label in enumerate(CEFR_LABELS)}

# T5 was trained with the standard T5 task prefix (thesis: summarize:).
T5_TASK_PREFIX = "summarize: "
