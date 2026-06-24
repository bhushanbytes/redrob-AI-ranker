"""Conservative pre-ranking filters for obviously unrelated profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

try:
    from .feature_extraction import CandidateFeatures, extract_features
    from .preprocessing import Candidate, candidate_text, clean_candidate
except ImportError:  # pragma: no cover - supports notebook imports from src path.
    from feature_extraction import CandidateFeatures, extract_features
    from preprocessing import Candidate, candidate_text, clean_candidate


MIN_AI_RELEVANCE_TO_KEEP = 0.18
MIN_ADJACENT_RELEVANCE_TO_KEEP = 0.25

ADJACENT_TECH_TERMS = {
    "backend engineer",
    "software engineer",
    "data engineer",
    "search engineer",
    "recommendation engineer",
    "analytics engineer",
    "platform engineer",
    "ml engineer",
    "machine learning engineer",
    "data scientist",
}

CLEAR_NON_FIT_TITLES = {
    "accountant",
    "civil engineer",
    "hr manager",
    "human resources",
    "sales executive",
    "sales manager",
    "marketing manager",
    "operations manager",
    "mechanical engineer",
    "graphic designer",
}


@dataclass(frozen=True)
class FilterDecision:
    """A filter decision with enough context to defend in an interview."""

    keep: bool
    reason: str
    features: CandidateFeatures


def should_keep_candidate(candidate: Candidate) -> FilterDecision:
    """Return whether a candidate should proceed to ranking."""
    cleaned = clean_candidate(candidate)
    features = extract_features(cleaned)
    text = candidate_text(cleaned)
    title = features.current_title
    adjacent_technical = any(term in title for term in ADJACENT_TECH_TERMS)
    domain_relevance = max(
        features.ai_ml_relevance_score,
        features.retrieval_search_score,
        features.ranking_score,
        features.recommendation_score,
    )
    has_relevant_skill = any(value > 0 for value in features.skill_group_scores.values())
    clear_non_fit = any(term in title for term in CLEAR_NON_FIT_TITLES)

    strong_system_signal = max(
        features.retrieval_search_score,
        features.ranking_score,
        features.recommendation_score,
    ) >= 0.4
    if clear_non_fit and not strong_system_signal:
        return FilterDecision(False, "clear non-technical/non-AI current title without strong system relevance", features)

    if adjacent_technical and (domain_relevance >= MIN_AI_RELEVANCE_TO_KEEP or has_relevant_skill):
        return FilterDecision(True, "adjacent technical profile with AI/search/ML evidence", features)

    if domain_relevance >= MIN_ADJACENT_RELEVANCE_TO_KEEP and has_relevant_skill:
        return FilterDecision(True, "sufficient AI/search/ML relevance", features)

    return FilterDecision(False, "insufficient AI/search/ML relevance for Senior AI Engineer ranking", features)


def filter_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    """Return candidates that pass the conservative relevance filter."""
    kept: list[Candidate] = []
    for candidate in candidates:
        if should_keep_candidate(candidate).keep:
            kept.append(candidate)
    return kept


def filter_candidates_with_decisions(candidates: Iterable[Candidate]) -> list[tuple[Candidate, FilterDecision]]:
    """Return candidates paired with filter decisions for audit and EDA."""
    return [(candidate, should_keep_candidate(candidate)) for candidate in candidates]
