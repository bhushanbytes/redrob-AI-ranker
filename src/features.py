"""Backward-compatible feature API.

Prefer importing from feature_extraction.py in new code.
"""

from __future__ import annotations

try:
    from .feature_extraction import CandidateFeatures, extract_features
    from .ranker import score_candidate_features
except ImportError:  # pragma: no cover - supports notebook imports from src path.
    from feature_extraction import CandidateFeatures, extract_features
    from ranker import score_candidate_features


def career_score(candidate: dict) -> float:
    """Return the candidate career component on a 0-1 scale."""
    features = extract_features(candidate)
    experience_score = 0.25 if features.years_of_experience < 3 else min(1.0, 0.75 + features.years_of_experience * 0.03)
    return max(
        0.0,
        min(
            1.0,
            0.25 * experience_score
            + 0.35 * features.ai_ml_relevance_score
            + 0.20 * features.retrieval_search_score
            + 0.10 * features.ranking_score
            + 0.10 * features.recommendation_score,
        ),
    )


def skill_score(candidate: dict) -> float:
    """Return a simple skill coverage score on a 0-1 scale."""
    features = extract_features(candidate)
    groups = features.skill_group_scores
    return sum(groups.values()) / len(groups) if groups else 0.0


def behaviour_score(candidate: dict) -> float:
    """Return a behavioral signal score on a 0-1 scale."""
    features = extract_features(candidate)
    return max(
        0.0,
        min(
            1.0,
            0.20 * features.open_to_work_flag
            + 0.25 * features.recruiter_response_rate
            + 0.20 * features.interview_completion_rate
            + 0.20 * features.profile_completeness_score
            + 0.15 * features.recency_score,
        ),
    )
