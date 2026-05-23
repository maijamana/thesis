# TSAR 2025 Evaluation

Модуль для оцінки submission у форматі TSAR 2025 Shared Task (readability-controlled text simplification).

## Метрики

- **CEFR compliance**: 3 моделі ModernBERT (doc_en, doc_sent_en, reference_AllLang2); для кожного тексту береться передбачення з найвищою впевненістю. Далі:
  - `weighted_f1`
  - `adj_accuracy` (відповідність у межах ±1 рівень)
  - `rmse`
- **Meaning preservation**: MeaningBERT
  - `MeaningBERT_Org` — спрощений vs оригінал
  - `MeaningBERT_Ref` — спрощений vs референс

BERTScore не використовується.

## Батчі

- CEFR: тексти передаються в pipeline батчами (за замовчуванням 32). Розмір: `CEFR_BATCH_SIZE`.
- MeaningBERT: обчислення батчами (за замовчуванням 32). Розмір: `MEANINGBERT_BATCH_SIZE`.

## Залежності

- `transformers`, `evaluate`, `scikit-learn`, `numpy`
- Для CEFR потрібен доступ до Hugging Face (моделі AbdullahBarayan/ModernBERT-*).

## Використання

### CLI

Один файл (submission з усіма полями: text_id, original, reference, target_cefr, simplified):

```bash
python -m tsar_evaluation.evaluate data/data_for_evaluation/tsar2025_test_t5_simplified.jsonl
```

Два файли (submission + reference):

```bash
python -m tsar_evaluation.evaluate submission.jsonl reference.jsonl
```

Опційно через змінні середовища:

- `CEFR_BATCH_SIZE=16` — розмір батча для CEFR
- `MEANINGBERT_BATCH_SIZE=16` — розмір батча для MeaningBERT

### Імпорт

```python
from tsar_evaluation import read_jsonl, load_models, evaluate_submission, run_from_cli

# Один виклик з файлів
results = run_from_cli("tsar2025_test_t5_simplified.jsonl")

# Або з даних у пам’яті
submission_data = read_jsonl("submission.jsonl")
ref_data = read_jsonl("reference.jsonl")
results = evaluate_submission(submission_data, ref_data=ref_data)
# results["CEFR_compliance"], results["Similarity_to_Original"], results["Similarity_to_Reference"]
```

## Формат JSONL

- **Submission**: кожен рядок — JSON з полями `text_id`, `simplified` (обов’язково).
- **Reference** (окремий файл або ті самі записи): `text_id`, `original`, `reference`, `target_cefr`.

Якщо викликати з одним файлом, він використовується і як submission, і як reference (мають бути поля original, reference, target_cefr, simplified).
