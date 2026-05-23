"""
Модуль для оцінки якості спрощення тексту.
"""
import numpy as np
import evaluate
from typing import List


def get_meaningbert_score(
    predictions: List[str],
    references: List[str],
    model
) -> float:
    """
    Обчислює MeaningBERT score для оцінки збереження смислу.
    
    Args:
        predictions: Список спрощених текстів
        references: Список оригінальних текстів
        model: Завантажена модель MeaningBERT
    
    Returns:
        Середній MeaningBERT score
    """
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
    scoretype: str = "f1"
) -> float:
    """
    Обчислює BERTScore для оцінки збереження смислу.
    
    Args:
        predictions: Список спрощених текстів
        references: Список оригінальних текстів
        scorer: Завантажений BERTScore scorer
        scoretype: Тип скору ('f1', 'precision', 'recall')
    
    Returns:
        Середній BERTScore
    """
    result = scorer.compute(references=references, predictions=predictions, lang="en")
    return round(np.mean(result[scoretype]), 4)
