#!/usr/bin/env python3
"""
Compute CEFR difficulty difference for sentence pairs from CSV file.

Input: CSV with columns: index_normal,index_simple,original,simplified
Output: CSV with additional columns: cefr_orig, cefr_simp, cefr_diff
"""

from __future__ import annotations

import argparse
import sys
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
    """Compute expected CEFR level for a list of texts."""
    results: List[List[Dict]] = labeler(texts, truncation=True, batch_size=batch_size)
    return [
        float(sum(CEFR_ORDER[p["label"]] * p["score"] for p in preds))
        for preds in results
    ]


def process_batch_with_progress(
    labeler,
    original_texts: List[str],
    simplified_texts: List[str],
    batch_size: int = 32,
    progress_interval: int = 1000,
) -> tuple[List[float], List[float]]:
    """Process texts in batches with progress reporting."""
    orig_levels = []
    simp_levels = []
    total_batches = len(original_texts) // batch_size + (1 if len(original_texts) % batch_size else 0)
    
    print(f"Processing {len(original_texts)} sentence pairs in {total_batches} batches...")
    
    for i in range(0, len(original_texts), batch_size):
        batch_end = min(i + batch_size, len(original_texts))
        orig_batch = original_texts[i:batch_end]
        simp_batch = simplified_texts[i:batch_end]
        
        # Process original sentences
        orig_batch_levels = cefr_expected_level(labeler, orig_batch, batch_size)
        orig_levels.extend(orig_batch_levels)
        
        # Process simplified sentences
        simp_batch_levels = cefr_expected_level(labeler, simp_batch, batch_size)
        simp_levels.extend(simp_batch_levels)
        
        # Progress reporting
        processed = batch_end
        if processed % progress_interval == 0 or processed == len(original_texts):
            progress_percent = (processed / len(original_texts)) * 100
            print(f"Processed: {processed}/{len(original_texts)} ({progress_percent:.1f}%)")
            sys.stdout.flush()
    
    return orig_levels, simp_levels


def compute_cefr_diff(
    input_path: Path,
    output_path: Path,
    threshold: float = 0.5,
    sample_size: int | None = None,
    first_n: int | None = None,
    seed: int = 42,
    batch_size: int = 32,
    progress_interval: int = 1000,
) -> None:
    """Compute CEFR difference for sentence pairs and save results."""
    
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Remove empty rows
    df = df.replace("", np.nan).dropna(subset=['original', 'simplified']).reset_index(drop=True)
    
    # Sampling/limiting
    if first_n is not None and first_n > 0:
        df = df.head(first_n).reset_index(drop=True)
        print(f"Using first {len(df)} pairs")
    
    if sample_size is not None and sample_size > 0 and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
        print(f"Using random sample of {len(df)} pairs")
    
    print(f"Processing {len(df)} sentence pairs...\n")
    
    # Initialize CEFR labeler
    print("Loading CEFR model...")
    cefr_labeler = pipeline(
        task="text-classification",
        model="AbdullahBarayan/ModernBERT-base-doc_sent_en-Cefr",
        top_k=None,
    )
    
    # Extract sentences
    original_texts = df["original"].tolist()
    simplified_texts = df["simplified"].tolist()
    
    # Compute CEFR levels with progress
    print("Computing CEFR levels...")
    orig_levels, simp_levels = process_batch_with_progress(
        cefr_labeler,
        original_texts,
        simplified_texts,
        batch_size=batch_size,
        progress_interval=progress_interval,
    )
    
    # Add CEFR columns
    df["cefr_orig"] = orig_levels
    df["cefr_simp"] = simp_levels
    df["cefr_diff"] = df["cefr_orig"] - df["cefr_simp"]
    
    # Filter by threshold
    filtered = df[df["cefr_diff"] >= threshold].reset_index(drop=True)
    kept = len(filtered)
    total = len(df)
    
    print(f"\nResults:")
    print(f"Total pairs processed: {total}")
    print(f"Pairs with cefr_diff >= {threshold}: {kept} ({kept/total:.1%})")
    print(f"Average CEFR diff: {df['cefr_diff'].mean():.2f}")
    print(f"CEFR diff range: {df['cefr_diff'].min():.2f} to {df['cefr_diff'].max():.2f}")
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_path, index=False)
    print(f"\nFiltered results saved to: {output_path}")
    
    # Also save full results
    full_output_path = output_path.parent / f"full_{output_path.name}"
    df.to_csv(full_output_path, index=False)
    print(f"Full results saved to: {full_output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute CEFR difficulty difference for sentence pairs")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input CSV file with columns: index_normal,index_simple,original,simplified",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV file path",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Minimum CEFR difference threshold (default: 0.5)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Random sample size for testing",
    )
    parser.add_argument(
        "--first-n",
        type=int,
        default=None,
        help="Use first N pairs (deterministic). Cannot be used together with --sample-size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing (default: 32)",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=1000,
        help="Progress reporting interval (default: 1000)",
    )
    
    args = parser.parse_args()
    
    if args.first_n is not None and args.sample_size is not None:
        raise SystemExit("Error: use either --first-n or --sample-size, not both.")
    
    compute_cefr_diff(
        input_path=args.input,
        output_path=args.output,
        threshold=args.threshold,
        sample_size=args.sample_size,
        first_n=args.first_n,
        seed=args.seed,
        batch_size=args.batch_size,
        progress_interval=args.progress_interval,
    )


if __name__ == "__main__":
    main()
