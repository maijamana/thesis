#!/usr/bin/env python3
"""
Filter aligned_pairs.csv by CEFR expected-level difference.

Input: data/aligned_pairs.csv (with columns: topic, index_normal, index_simple, original, simplified)
Output: CSV with columns: topic, index_normal, index_simple, original, simplified, cefr_orig, cefr_simp, cefr_diff
  filtered by cefr_diff >= threshold (default 0.8)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from transformers import pipeline


CEFR_ORDER = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}


def cefr_expected_level(
    labeler,
    texts: List[str],
    batch_size: int = 32,
) -> List[float]:
    """Обчислює очікуваний CEFR рівень для списку текстів."""
    results: List[List[Dict]] = labeler(texts, truncation=True, batch_size=batch_size)
    return [
        float(sum(CEFR_ORDER[p["label"]] * p["score"] for p in preds))
        for preds in results
    ]


def filter_cefr_diff(
    df: pd.DataFrame,
    threshold: float = 0.8,
    sample_size: int | None = None,
    first_n: int | None = None,
    skip: int = 0,
    seed: int = 42,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Фільтрує пари речень за CEFR різницею.
    
    Args:
        df: DataFrame з колонками 'original' та 'simplified' (може містити додаткові колонки)
        threshold: Мінімальна різниця cefr_orig - cefr_simp
        sample_size: Обробити випадковий семпл N пар
        first_n: Обробити перші N пар після skip (детерміновано)
        skip: Пропустити перші skip рядків (для обробки з рядка 15001 по 30000: skip=15000, first_n=15000)
        seed: Seed для випадкового семплу
        batch_size: Розмір батчу для CEFR класифікатора
    
    Returns:
        DataFrame з додатковими колонками cefr_orig, cefr_simp, cefr_diff та всіма оригінальними колонками
    """
    # Перевіряємо наявність необхідних колонок
    required_cols = ['original', 'simplified']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Відсутні колонки: {missing_cols}")
    
    # Зберігаємо всі колонки (включаючи додаткові як topic, index_normal, тощо)
    data = df.copy()
    data = data.replace("", np.nan)
    data = data.dropna(subset=required_cols).reset_index(drop=True)
    
    # Фільтруємо за довжиною (мінімум 5 символів)
    data = data[
        (data['original'].str.len() >= 5) & 
        (data['simplified'].str.len() >= 5)
    ].reset_index(drop=True)
    
    # Видаляємо дублікати (якщо original == simplified)
    data = data[data['original'] != data['simplified']].reset_index(drop=True)
    
    # Пропускаємо перші skip рядків
    if skip > 0:
        if skip >= len(data):
            print("⚠️  skip >= кількості рядків, немає даних для обробки!")
            return pd.DataFrame()
        data = data.iloc[skip:].reset_index(drop=True)
        print(f"Пропущено перші {skip} рядків, залишилось {len(data)} пар.")
    
    # Обмежуємо кількість пар для обробки
    if first_n is not None and first_n > 0:
        data = data.head(first_n).reset_index(drop=True)
        print(f"Обробляємо цей діапазон: {len(data)} пар...")
    elif sample_size is not None and sample_size > 0 and sample_size < len(data):
        data = data.sample(n=sample_size, random_state=seed).reset_index(drop=True)
        print(f"Обробляємо випадковий семпл {len(data)} пар...")
    
    print(f"Аналізуємо {len(data)} пар...\n")
    
    if len(data) == 0:
        print("⚠️  Немає пар для обробки!")
        return pd.DataFrame()
    
    # Завантажуємо CEFR класифікатор
    print("Завантажуємо CEFR класифікатор...")
    cefr_labeler = pipeline(
        task="text-classification",
        model="AbdullahBarayan/ModernBERT-base-doc_sent_en-Cefr",
        top_k=None,
    )
    
    # Обчислюємо CEFR рівні
    print("    -> Рахуємо CEFR рівні оригіналів...")
    orig_levels = cefr_expected_level(cefr_labeler, data["original"].tolist(), batch_size=batch_size)
    
    print("    -> Рахуємо CEFR рівні спрощень...")
    simp_levels = cefr_expected_level(cefr_labeler, data["simplified"].tolist(), batch_size=batch_size)
    
    # Додаємо результати до даних
    data = data.copy()
    data["cefr_orig"] = orig_levels
    data["cefr_simp"] = simp_levels
    data["cefr_diff"] = data["cefr_orig"] - data["cefr_simp"]
    
    # Фільтруємо за порогом
    filtered = data[data["cefr_diff"] >= threshold].reset_index(drop=True)
    kept = len(filtered)
    total = len(data)
    print(f"\nПісля фільтрації (cefr_diff >= {threshold}) залишилось: {kept} пар ({kept/total:.1%})")
    
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Фільтрує aligned_pairs.csv за CEFR різницею"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/aligned_pairs.csv"),
        help="Шлях до aligned_pairs.csv (default: data/aligned_pairs.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Шлях для збереження CSV (default: data/aligned_pairs_filtered.csv)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Мінімальна CEFR різниця (cefr_orig - cefr_simp >= threshold). Default: 0.8",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Обробити випадковий семпл N пар для швидкого тестування",
    )
    parser.add_argument(
        "--first-n",
        type=int,
        default=None,
        help="Обробити перші N пар після --skip (детерміновано). Не можна разом з --sample-size.",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Пропустити перші N рядків (наприклад 15000), потім обробити --first-n рядків. Для рядків 15001–30000: --skip 15000 --first-n 15000",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed для випадкового семплу (default: 42)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Розмір батчу для CEFR класифікатора (default: 32)",
    )
    
    args = parser.parse_args()
    
    if args.first_n is not None and args.sample_size is not None:
        raise SystemExit("Помилка: використовуйте або --first-n, або --sample-size (не разом).")
    
    if not args.input.exists():
        raise SystemExit(f"❌ Помилка: файл {args.input} не знайдено")
    
    # Визначаємо шлях для збереження
    if args.output is None:
        args.output = args.input.parent / f"{args.input.stem}_filtered.csv"
    
    print(f"Читаємо дані з: {args.input}")
    df = pd.read_csv(args.input, encoding='utf-8')
    print(f"Завантажено {len(df)} пар\n")
    
    # Фільтруємо (функція збереже всі колонки з df)
    filtered = filter_cefr_diff(
        df,
        threshold=args.threshold,
        sample_size=args.sample_size,
        first_n=args.first_n,
        skip=args.skip,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    
    if filtered.empty:
        print("⚠️  Після фільтрації не залишилось пар!")
        return
    
    # Зберігаємо результат
    args.output.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(args.output, index=False, encoding='utf-8')
    print(f"\n✅ Результати збережено у: {args.output}")
    print(f"   Всього пар: {len(filtered)}")
    print(f"   Середня CEFR різниця: {filtered['cefr_diff'].mean():.2f}")


if __name__ == "__main__":
    main()
