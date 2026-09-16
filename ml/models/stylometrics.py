"""
Review Sense - Stylometric Feature & Explainable Risk Signal Extractor
Computes explainable heuristics and signals for deceptive, spam, or fake reviews.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

# Common promotional / hyperbolic keywords frequently found in spam & incentivized fake reviews
PROMOTIONAL_SUPERLATIVES = {
    "amazing", "incredible", "unbelievable", "perfection", "flawless",
    "best ever", "must buy", "game changer", "revolutionary", "miracle",
    "10/10", "100%", "buy now", "don't hesitate", "worth every penny",
    "changed my life", "blown away", "holy grail", "exceeded all expectations",
    "five stars", "5 stars", "highly recommend", "super cute", "super great",
}

# Generic / vague phrases commonly found in computer-generated bot reviews
GENERIC_BOT_PHRASES = {
    "works as intended", "good for the price", "very pretty", "nice product",
    "love it", "great product", "good quality", "as described",
    "item arrived", "does the job", "well made", "really like it",
}


def extract_stylometric_features(text: str) -> Dict[str, float]:
    """
    Extract numerical stylometric features from review text.
    """
    if not text or not text.strip():
        return {
            "char_length": 0.0,
            "word_count": 0.0,
            "avg_word_length": 0.0,
            "caps_ratio": 0.0,
            "exclamation_ratio": 0.0,
            "question_ratio": 0.0,
            "punctuation_density": 0.0,
            "lexical_diversity": 0.0,
            "superlative_density": 0.0,
            "generic_phrase_count": 0.0,
        }

    cleaned_text = text.strip()
    char_length = len(cleaned_text)
    words = re.findall(r"\b\w+\b", cleaned_text)
    word_count = max(len(words), 1)

    # Word lengths
    avg_word_length = sum(len(w) for w in words) / word_count

    # Capitalization & Punctuation
    alpha_chars = [c for c in cleaned_text if c.isalpha()]
    caps_count = sum(1 for c in alpha_chars if c.isupper())
    caps_ratio = (caps_count / len(alpha_chars)) if alpha_chars else 0.0

    exclamation_count = cleaned_text.count("!")
    question_count = cleaned_text.count("?")
    exclamation_ratio = exclamation_count / word_count
    question_ratio = question_count / word_count

    punct_chars = re.findall(r"[!?,.:;\"'()\-]", cleaned_text)
    punctuation_density = len(punct_chars) / max(char_length, 1)

    # Lexical Diversity (Type-Token Ratio)
    unique_words = set(w.lower() for w in words)
    lexical_diversity = len(unique_words) / word_count

    # Superlatives & Generic Phrases
    text_lower = cleaned_text.lower()
    superlative_count = sum(1 for s in PROMOTIONAL_SUPERLATIVES if s in text_lower)
    superlative_density = superlative_count / (word_count / 10.0)

    generic_phrase_count = sum(1 for g in GENERIC_BOT_PHRASES if g in text_lower)

    return {
        "char_length": float(char_length),
        "word_count": float(word_count),
        "avg_word_length": round(avg_word_length, 3),
        "caps_ratio": round(caps_ratio, 3),
        "exclamation_ratio": round(exclamation_ratio, 3),
        "question_ratio": round(question_ratio, 3),
        "punctuation_density": round(punctuation_density, 3),
        "lexical_diversity": round(lexical_diversity, 3),
        "superlative_density": round(superlative_density, 3),
        "generic_phrase_count": float(generic_phrase_count),
    }


def detect_explainable_signals(text: str, fake_prob: float) -> List[str]:
    """
    Generate human-interpretable risk signal tags explaining why a review is flagged as suspicious.
    """
    signals: List[str] = []
    if not text or not text.strip():
        return signals

    features = extract_stylometric_features(text)
    text_lower = text.lower()

    # 1. Punctuation and capitalization intensity
    if features["caps_ratio"] > 0.35 and features["word_count"] > 3:
        signals.append("Excessive capitalization (SHOUTING)")
    if features["exclamation_ratio"] > 0.20 or "!!" in text or "!?" in text:
        signals.append("High punctuation intensity (repeated exclamation marks)")

    # 2. Promotional superlatives
    found_superlatives = [s for s in PROMOTIONAL_SUPERLATIVES if s in text_lower]
    if len(found_superlatives) >= 2 or features["superlative_density"] > 0.8:
        signals.append(f"Heavy promotional superlatives ({', '.join(found_superlatives[:3])})")

    # 3. Very low lexical diversity
    if features["lexical_diversity"] < 0.60 and features["word_count"] > 10:
        signals.append("Repetitive phrasing / low lexical diversity")

    # 4. Generic boilerplate wording
    found_generic = [g for g in GENERIC_BOT_PHRASES if g in text_lower]
    if len(found_generic) >= 2:
        signals.append(f"Vague template language ({', '.join(found_generic[:2])})")

    # 5. Short extreme review pattern
    if features["word_count"] < 8 and (features["superlative_density"] > 0 or features["caps_ratio"] > 0.25):
        signals.append("Extremely brief polarized review")

    # If model probability is high but stylometrics didn't trigger obvious flags
    if fake_prob >= 0.70 and not signals:
        signals.append("Unnatural syntactic phrasing detected by neural language model")

    return signals
