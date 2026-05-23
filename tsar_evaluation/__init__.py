"""
TSAR 2025 Shared Task — оцінка readability-controlled text simplification.
Метрики: CEFR compliance (weighted F1, adj. accuracy, RMSE), MeaningBERT (original, reference).
"""
from .evaluate import (
    read_jsonl,
    load_models,
    evaluate_submission,
    run_from_cli,
)

__all__ = [
    "read_jsonl",
    "load_models",
    "evaluate_submission",
    "run_from_cli",
]
