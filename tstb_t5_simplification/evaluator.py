"""
Оцінка якості спрощення (MeaningBERT, BERTScore).
"""
import numpy as np
import evaluate
from typing import List


def get_meaningbert_score(
    predictions: List[str],
    references: List[str],
    model,
) -> float:
    assert len(predictions) == len(references)
    result = []
    for pred, ref in zip(predictions, references):
        score = model.compute(predictions=[pred], references=[ref])
        result.append(score["scores"][0])
    return round(np.mean(result) / 100, 4)


def get_bertscore(
    predictions: List[str],
    references: List[str],
    scorer,
    scoretype: str = "f1",
) -> float:
    result = scorer.compute(references=references, predictions=predictions, lang="en")
    return round(np.mean(result[scoretype]), 4)
