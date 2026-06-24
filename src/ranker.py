"""Weighted ranking engine for the Redrob candidate challenge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

try:
    from .candidate_filter import should_keep_candidate
    from .feature_extraction import CandidateFeatures, extract_features
    from .preprocessing import Candidate, clean_candidate, extract_skill_names
except ImportError:  # pragma: no cover - supports notebook imports from src path.
    from candidate_filter import should_keep_candidate
    from feature_extraction import CandidateFeatures, extract_features
    from preprocessing import Candidate, clean_candidate, extract_skill_names


@dataclass(frozen=True)
class RankWeights:
    """Configurable weights for final ranking components."""

    career: float = 42.0
    skills: float = 28.0
    behavior: float = 14.0
    logistics: float = 6.0
    authenticity: float = 10.0


@dataclass(frozen=True)
class RankedCandidate:
    """Ranked candidate output plus optional internals for audit."""

    candidate_id: str
    rank: int
    score: float
    features: CandidateFeatures


def rank_candidates(
    candidates: Iterable[Candidate],
    top_n: int = 100,
    weights: RankWeights | None = None,
    apply_filter: bool = True,
) -> list[RankedCandidate]:
    """Rank candidates and return the top N candidate ids with scores."""
    active_weights = weights or RankWeights()
    scored: list[tuple[Candidate, CandidateFeatures, float]] = []

    for candidate in candidates:
        cleaned = clean_candidate(candidate)
        if apply_filter:
            decision = should_keep_candidate(cleaned)
            if not decision.keep:
                continue
            features = decision.features
        else:
            features = extract_features(cleaned)
        score = score_candidate_features(features, active_weights)
        scored.append((cleaned, features, score))

    scored.sort(key=lambda item: (-item[2], item[1].candidate_id))
    return [
        RankedCandidate(
            candidate_id=features.candidate_id,
            rank=index,
            score=round(score, 4),
            features=features,
        )
        for index, (_, features, score) in enumerate(scored[:top_n], start=1)
    ]


def top_100_rows(
    candidates: Iterable[Candidate],
    weights: RankWeights | None = None,
    apply_filter: bool = True,
) -> list[dict[str, float | int | str]]:
    """Return challenge-ready rows containing candidate_id, rank, and score."""
    return [
        {"candidate_id": item.candidate_id, "rank": item.rank, "score": item.score}
        for item in rank_candidates(candidates, top_n=100, weights=weights, apply_filter=apply_filter)
    ]


def score_candidate(candidate: Candidate, weights: RankWeights | None = None) -> float:
    """Score one candidate after cleaning and feature extraction."""
    return score_candidate_features(extract_features(clean_candidate(candidate)), weights or RankWeights())


def score_candidate_features(features: CandidateFeatures, weights: RankWeights) -> float:
    """Combine feature groups into a deterministic weighted score."""
    career = _career_score(features)
    skills = _skill_score(features)
    behavior = _behavior_score(features)
    logistics = _logistics_score(features)
    authenticity = _authenticity_score(features)
    return (
        weights.career * career
        + weights.skills * skills
        + weights.behavior * behavior
        + weights.logistics * logistics
        + weights.authenticity * authenticity
    )


def generate_reason(candidate: Candidate, ranked: RankedCandidate | None = None) -> str:
    """Generate a concise factual explanation from candidate data only."""
    cleaned = clean_candidate(candidate)
    features = ranked.features if ranked else extract_features(cleaned)
    profile = cleaned.get("profile") or {}
    signals = cleaned.get("redrob_signals") or {}
    strengths: list[str] = _career_evidence(cleaned, features)
    concerns: list[str] = []

    skill_details = _relevant_skill_details(cleaned)
    if skill_details:
        strengths.append("skills include " + ", ".join(skill_details[:5]))

    if features.consulting_only_penalty > 0:
        concerns.append("services-heavy career history")
    if features.honeypot_indicator_score > 0:
        concerns.append("possible keyword-stuffing or honeypot signal")
    if features.unrealistic_proficiency_penalty > 0:
        concerns.append("unusually broad high-proficiency skill pattern")
    if features.disfavored_domain_penalty > 0:
        concerns.append("domain is less preferred unless IR/NLP depth is strong")
    if features.jd_experience_fit_score < 0.7:
        concerns.append(f"experience {features.years_of_experience:g} yrs is outside JD sweet spot")
    if features.recruiter_response_rate < 0.35:
        concerns.append(f"recruiter response rate {_format_number(features.recruiter_response_rate)}")
    if not signals.get("open_to_work_flag"):
        concerns.append("not marked open to work")
    if features.notice_period_score <= 0.2:
        concerns.append(f"notice period {signals.get('notice_period_days', 'unknown')} days")

    title = profile.get("current_title") or "unknown title"
    years = profile.get("years_of_experience", "unknown")
    company = profile.get("current_company") or "unknown company"
    behavior = (
        f"behavior: response {_format_number(features.recruiter_response_rate)}, "
        f"interview {_format_number(features.interview_completion_rate)}, "
        f"profile {_format_number(features.profile_completeness_score * 100)}%, "
        f"saved {int(signals.get('saved_by_recruiters_30d') or 0)} times in 30d"
    )
    if ranked and ranked.rank >= 90:
        base = f"{features.candidate_id}: lower Top-100 fit, {title} at {company}, {years} years."
    else:
        base = f"{features.candidate_id}: {title} at {company}, {years} years."
    strength_text = " Evidence: " + "; ".join(strengths[:4]) + f"; {behavior}." if strengths else f" Evidence: {behavior}."
    concern_text = " Concerns: " + "; ".join(concerns[:3]) + "." if concerns else ""
    rank_text = f" Rank {ranked.rank}, score {ranked.score}." if ranked else ""
    return base + rank_text + strength_text + concern_text


def _career_evidence(candidate: Candidate, features: CandidateFeatures) -> list[str]:
    """Extract candidate-specific career evidence for explanations."""
    evidence: list[str] = []
    text_by_area = {
        "retrieval/search": features.retrieval_search_score,
        "ranking": features.ranking_score,
        "recommendation": features.recommendation_score,
        "AI/ML": features.ai_ml_relevance_score,
        "product/platform": features.product_company_score,
    }
    strong_areas = [name for name, score in text_by_area.items() if score >= 0.5]
    if strong_areas:
        evidence.append("career evidence in " + ", ".join(strong_areas[:3]))

    current_job = next(
        (
            job
            for job in candidate.get("career_history") or []
            if isinstance(job, dict) and job.get("is_current")
        ),
        None,
    )
    if current_job:
        details = _job_signal_summary(current_job)
        if details:
            evidence.append(f"current role mentions {details}")
    return evidence


def _job_signal_summary(job: dict) -> str:
    description = str(job.get("description") or "")
    terms = [
        "retrieval",
        "search",
        "ranking",
        "recommendation",
        "personalization",
        "embeddings",
        "vector",
        "rag",
        "ml",
        "model",
        "spark",
        "kafka",
        "production",
    ]
    found = [term for term in terms if term in description]
    return ", ".join(found[:5])


def _relevant_skill_details(candidate: Candidate) -> list[str]:
    wanted = {
        "python",
        "nlp",
        "machine learning",
        "scikit-learn",
        "pytorch",
        "tensorflow",
        "llm",
        "fine-tuning llms",
        "embeddings",
        "milvus",
        "faiss",
        "pinecone",
        "weaviate",
        "qdrant",
        "elasticsearch",
        "open search",
        "ranking",
        "spark",
    }
    details: list[str] = []
    for skill in candidate.get("skills") or []:
        if not isinstance(skill, dict):
            continue
        name = str(skill.get("name") or "")
        if name not in wanted:
            continue
        proficiency = str(skill.get("proficiency") or "").strip()
        months = skill.get("duration_months")
        if months:
            details.append(f"{name} ({proficiency}, {months} mo)")
        elif proficiency:
            details.append(f"{name} ({proficiency})")
        else:
            details.append(name)
    return details


def _format_number(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _career_score(features: CandidateFeatures) -> float:
    experience_score = _bounded_experience_score(features.years_of_experience)
    seniority_score = 1.0 if "senior" in features.current_title or "lead" in features.current_title else 0.65
    domain_score = (
        0.32 * features.ai_ml_relevance_score
        + 0.24 * features.retrieval_search_score
        + 0.18 * features.ranking_score
        + 0.14 * features.recommendation_score
        + 0.12 * features.product_company_score
    )
    penalty = 0.18 * features.consulting_only_penalty + 0.12 * features.disfavored_domain_penalty
    return _clamp(
        0.18 * experience_score
        + 0.17 * features.jd_experience_fit_score
        + 0.12 * seniority_score
        + 0.53 * domain_score
        - penalty
    )


def _skill_score(features: CandidateFeatures) -> float:
    groups = features.skill_group_scores
    core = (
        0.18 * groups.get("python", 0.0)
        + 0.14 * groups.get("ml", 0.0)
        + 0.12 * groups.get("nlp", 0.0)
        + 0.12 * groups.get("llm", 0.0)
        + 0.14 * groups.get("embeddings", 0.0)
        + 0.12 * groups.get("vector_db", 0.0)
        + 0.10 * groups.get("search", 0.0)
        + 0.08 * groups.get("ranking", 0.0)
    )
    duration_score = _clamp(features.avg_relevant_skill_duration_months / 48.0)
    return _clamp(0.72 * core + 0.18 * features.avg_skill_proficiency + 0.10 * duration_score)


def _behavior_score(features: CandidateFeatures) -> float:
    return _clamp(
        0.12 * features.open_to_work_flag
        + 0.22 * features.recruiter_response_rate
        + 0.18 * features.interview_completion_rate
        + 0.14 * features.profile_completeness_score
        + 0.14 * features.recency_score
        + 0.10 * features.github_activity_score
        + 0.10 * features.saved_by_recruiters_30d
    )


def _logistics_score(features: CandidateFeatures) -> float:
    return _clamp(
        0.50 * features.notice_period_score
        + 0.25 * features.willing_to_relocate
        + 0.25 * features.location_preference_score
    )


def _authenticity_score(features: CandidateFeatures) -> float:
    penalty = (
        0.30 * features.suspicious_skill_count_score
        + 0.32 * features.unrealistic_proficiency_penalty
        + 0.20 * features.timeline_consistency_penalty
        + 0.18 * features.honeypot_indicator_score
    )
    return _clamp(1.0 - penalty)


def _bounded_experience_score(years: float) -> float:
    if years < 3:
        return 0.25
    if years <= 8:
        return 0.75 + (years - 3) * 0.05
    if years <= 15:
        return 0.78
    return 0.45


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))
