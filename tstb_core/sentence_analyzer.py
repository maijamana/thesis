"""Sentence segmentation (spaCy) and continuous CEFR expected scores per sentence."""

from typing import Dict, List

import spacy

from .constants import LEVEL_MAP
from .cefr_scorer import CEFRScorer


def analyze_sentence_probs(model, sentence: str) -> dict:
    """Compute per-label probabilities and ExpectedScore for one sentence."""
    preds = model(sentence, top_k=None)
    probs = {d["label"]: round(float(d["score"]), 1) for d in preds}
    expected = sum(LEVEL_MAP[d["label"]] * float(d["score"]) for d in preds)
    return {**probs, "ExpectedScore": round(expected, 3)}


class SentenceInfluenceAnalyzerBySentence:
    """Segment documents and rank sentences by continuous CEFR complexity."""

    def __init__(self, scorer: CEFRScorer) -> None:
        self.scorer = scorer
        self.nlp = spacy.load("en_core_web_sm")

    def split_sentences(self, text: str) -> List[str]:
        """Split text with spaCy; further split on embedded newlines."""
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        expanded: List[str] = []
        for sent in sentences:
            if "\n" in sent:
                parts = [part.strip() for part in sent.split("\n") if part.strip()]
                expanded.extend(parts)
            else:
                expanded.append(sent)
        return expanded

    def analyze_sentence_impact(self, text: str) -> Dict:
        """Return ordered sentences with ExpectedScore and document-level label."""
        sentences = self.split_sentences(text)
        document_level = self.scorer.get_cefr_label(text)
        top_label = document_level["label"]

        impact_scores = []
        for i, sentence in enumerate(sentences):
            preds_info = analyze_sentence_probs(self.scorer.sentence_model, sentence)
            impact_scores.append(
                {
                    "sentence": sentence,
                    "sentence_index": i,
                    "original_level": top_label,
                    **preds_info,
                }
            )

        return {"sentences": sentences, "impact_analysis": impact_scores}
