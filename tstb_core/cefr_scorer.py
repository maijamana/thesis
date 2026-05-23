"""Document- and sentence-level CEFR scoring via ModernBERT classifiers."""

from transformers import pipeline

from .constants import CEFR_LABELS


class CEFRScorer:
    """Three-model document CEFR (best confidence) and sentence model for expected scores."""

    CEFR_LABELS = list(CEFR_LABELS)
    LABEL2IDX = {label: idx for idx, label in enumerate(CEFR_LABELS)}

    def __init__(self) -> None:
        self.cefr_models = [
            pipeline(
                task="text-classification",
                model="AbdullahBarayan/ModernBERT-base-doc_en-Cefr",
            ),
            pipeline(
                task="text-classification",
                model="AbdullahBarayan/ModernBERT-base-doc_sent_en-Cefr",
            ),
            pipeline(
                task="text-classification",
                model="AbdullahBarayan/ModernBERT-base-reference_AllLang2-Cefr2",
            ),
        ]
        self.sentence_model = pipeline(
            task="text-classification",
            model="AbdullahBarayan/ModernBERT-base-doc_sent_en-Cefr",
            top_k=None,
        )

    def get_cefr_label(self, text: str) -> dict:
        """Return the highest-confidence label among the three document classifiers."""
        top_preds = (model(text)[0] for model in self.cefr_models)
        best = max(top_preds, key=lambda d: d["score"])
        return {"label": best["label"], "score": best["score"]}
