#!/usr/bin/env python3
"""
Fine-tune T5-base for sentence simplification (thesis training procedure).

Example:
  python training/train.py --run run3 \\
    --data-csv data/data_for_training/full_aligned_pairs_cefr_filtered_wiki_0_6.csv \\
    --output-dir ./t5-base-simplification-res4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_TRAIN_DIR = Path(__file__).resolve().parent
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))

import torch
from transformers import (
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

from config import BASE_MODEL, RUNS, TASK_PREFIX
from cefr_metrics import build_cefr_labeler, make_compute_metrics
from data import load_pairs_csv, split_dataframe, tokenize_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune T5 for CEFR-oriented simplification")
    parser.add_argument("--run", choices=list(RUNS.keys()), default="run3", help="Thesis preset")
    parser.add_argument("--data-csv", required=True, help="Prepared CSV (original, simplified)")
    parser.add_argument("--output-dir", required=True, help="Checkpoint output directory")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RUNS[args.run]

    lr = args.learning_rate if args.learning_rate is not None else cfg.learning_rate
    warmup = args.warmup_steps if args.warmup_steps is not None else cfg.warmup_steps
    epochs = args.max_epochs if args.max_epochs is not None else cfg.num_train_epochs
    batch_size = args.batch_size if args.batch_size is not None else cfg.per_device_train_batch_size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Run: {cfg.name} — {cfg.description}")
    print(f"Expected scale: {cfg.train_pairs_note}")

    df = load_pairs_csv(args.data_csv)
    print(f"Loaded {len(df)} pairs from {args.data_csv}")

    train_df, valid_df, test_df = split_dataframe(df, random_state=args.seed)
    print(f"Split — train: {len(train_df)}, valid: {len(valid_df)}, test: {len(test_df)}")

    tokenizer = T5Tokenizer.from_pretrained(BASE_MODEL)
    model = T5ForConditionalGeneration.from_pretrained(BASE_MODEL)

    tokenized_train, tokenized_valid, tokenized_test = tokenize_datasets(
        train_df,
        valid_df,
        test_df,
        tokenizer,
        prefix=TASK_PREFIX,
        max_length=cfg.max_length,
    )

    cefr_device = 0 if device == "cuda" else -1
    cefr_labeler = build_cefr_labeler(device=cefr_device)
    orig_valid_texts = valid_df["orig"].tolist()
    compute_metrics, preprocess_logits = make_compute_metrics(
        tokenizer, orig_valid_texts, cefr_labeler
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_steps=warmup,
        weight_decay=cfg.weight_decay,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        fp16=device == "cuda" and not args.no_fp16,
        predict_with_generate=False,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True),
        preprocess_logits_for_metrics=preprocess_logits,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
    )

    print(f"\nTraining — prefix={TASK_PREFIX!r}, max_len={cfg.max_length}, lr={lr}, warmup={warmup}")
    trainer.train()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved model and tokenizer to {output_dir}")

    # Optional quick test-set CEFR summary (beam search, inference-style)
    model = model.to(device)
    model.eval()
    sample = test_df["orig"].head(20).tolist()
    preds = []
    for sent in sample:
        inputs = tokenizer(
            TASK_PREFIX + sent,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.max_length,
        ).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_length=cfg.max_length, num_beams=4, early_stopping=True)
        preds.append(tokenizer.decode(out[0], skip_special_tokens=True).strip())

    from cefr_metrics import cefr_expected_level

    diffs = [
        o - s
        for o, s in zip(
            cefr_expected_level(sample, cefr_labeler),
            cefr_expected_level(preds, cefr_labeler),
        )
    ]
    print(f"Test sample (n=20) mean CEFR diff: {sum(diffs)/len(diffs):.3f}")


if __name__ == "__main__":
    main()
