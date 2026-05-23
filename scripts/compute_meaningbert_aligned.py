#!/usr/bin/env python3
"""
Пайплайн: обчислення MeaningBERT для всіх пар з aligned_pairs_filtered.csv.

Використовує transformers (davebulaval/meaningbert).
predictions = simplified, references = original — оцінка збереження смислу.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def meaningbert_batch(
    preds: list[str],
    refs: list[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int = 32,
    verbose: bool = True,
):
    """
    Обчислює MeaningBERT scores для пар (prediction, reference).

    preds: спрощені речення (predictions)
    refs: оригінальні речення (references)
    Повертає список scores (масштаб як у автора моделі, наприклад 0–100).
    """
    model.eval()
    scores = []

    for i in range(0, len(preds), batch_size):
        if verbose:
            print(f"  batch {i} – {min(i + batch_size, len(preds))}/{len(preds)}")
        batch_preds = preds[i : i + batch_size]
        batch_refs = refs[i : i + batch_size]

        inputs = tokenizer(
            batch_preds,
            batch_refs,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # logits [batch_size, 1]
        batch_scores = outputs.logits.squeeze(-1).cpu().numpy()
        scores.extend(batch_scores.tolist())

    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Обчислення MeaningBERT для aligned_pairs_filtered.csv"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Вхідний CSV з колонками: index_normal,index_simple,original,simplified",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Вихідний CSV файл",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Залишити лише пари з meaning_score >= MIN_SCORE (наприклад 60)",
    )
    parser.add_argument(
        "--max-score",
        type=float,
        default=None,
        help="Залишити лише пари з meaning_score <= MAX_SCORE (наприклад 95)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Розмір батчу (default: 32)",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Не застосовувати фільтри; зберегти всі пари з колонкою meaning_score",
    )
    parser.add_argument(
        "--first-n",
        type=int,
        default=None,
        help="Обробити перші N пар для тестування",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Файл не знайдено: {args.input}")

    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input, encoding="utf-8")

    if "original" not in df.columns or "simplified" not in df.columns:
        raise SystemExit("У CSV мають бути колонки 'original' та 'simplified'.")

    # Remove empty rows
    df = df.replace("", pd.NA).dropna(subset=['original', 'simplified']).reset_index(drop=True)
    
    # Apply first-n limit if specified
    if args.first_n is not None and args.first_n > 0:
        df = df.head(args.first_n).reset_index(drop=True)
        print(f"Using first {len(df)} pairs")

    print(f"Processing {len(df)} sentence pairs...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Loading MeaningBERT...")
    tokenizer = AutoTokenizer.from_pretrained("davebulaval/meaningbert")
    model = AutoModelForSequenceClassification.from_pretrained(
        "davebulaval/meaningbert"
    ).to(device)
    model.eval()

    preds = df["simplified"].astype(str).tolist()
    refs = df["original"].astype(str).tolist()
    print(f"Computing MeaningBERT for {len(preds)} pairs...")
    df["meaning_score"] = meaningbert_batch(
        preds, refs, tokenizer, model, device, batch_size=args.batch_size
    )

    # Apply filters
    before_filter = len(df)
    
    if args.min_score is not None and not args.no_filter:
        df = df[df["meaning_score"] >= args.min_score].reset_index(drop=True)
        print(f"After min_score >= {args.min_score}: {len(df)} pairs ({len(df)/before_filter:.1%})")
    
    if args.max_score is not None and not args.no_filter:
        before_max_filter = len(df)
        df = df[df["meaning_score"] <= args.max_score].reset_index(drop=True)
        print(f"After max_score <= {args.max_score}: {len(df)} pairs ({len(df)/before_max_filter:.1%})")
    
    if args.min_score is not None and args.no_filter:
        print(
            f"Warning: --min-score {args.min_score} specified but --no-filter used; filter not applied."
        )
    
    if args.max_score is not None and args.no_filter:
        print(
            f"Warning: --max-score {args.max_score} specified but --no-filter used; filter not applied."
        )

    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"\nSaved: {out}")
    print(f"Pairs: {len(df)}")
    if len(df) > 0:
        print(
            f"meaning_score: min={df['meaning_score'].min():.2f}, max={df['meaning_score'].max():.2f}, mean={df['meaning_score'].mean():.2f}"
        )
    
    # Also save full results if filtering was applied
    if (args.min_score is not None or args.max_score is not None) and not args.no_filter:
        full_out = out.parent / f"full_{out.name}"
        # Reload original data and compute meaning scores for full dataset
        print(f"\nComputing meaning scores for full dataset...")
        full_df = pd.read_csv(args.input, encoding="utf-8")
        full_df = full_df.replace("", pd.NA).dropna(subset=['original', 'simplified']).reset_index(drop=True)
        
        if args.first_n is not None and args.first_n > 0:
            full_df = full_df.head(args.first_n).reset_index(drop=True)
        
        full_preds = full_df["simplified"].astype(str).tolist()
        full_refs = full_df["original"].astype(str).tolist()
        full_df["meaning_score"] = meaningbert_batch(
            full_preds, full_refs, tokenizer, model, device, batch_size=args.batch_size
        )
        
        full_df.to_csv(full_out, index=False, encoding="utf-8")
        print(f"Full results saved: {full_out}")


if __name__ == "__main__":
    main()
