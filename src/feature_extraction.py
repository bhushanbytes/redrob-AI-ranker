"""Deterministic engineered features for Senior AI Engineer ranking."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

try:
    from .preprocessing import Candidate, candidate_text, clean_candidate, extract_skill_names
except ImportError:  # pragma: no cover - supports notebook imports from src path.
    from preprocessing import Candidate, candidate_text, clean_candidate, extract_skill_names


REFERENCE_DATE = date(2026, 6, 15)

AI_ML_TERMS = {
    "ai",
    "artificial intelligence",
    "machine learning",
    "ml",
    "deep learning",
    "nlp",
    "llm",
    "transformer",
    "model",
    "feature engineering",
    "production ml",
    "mlops",
}
RETRIEVAL_TERMS = {
    "retrieval",
    "information retrieval",
    "semantic search",
    "vector search",
    "embedding search",
    "rag",
    "search relevance",
}
RANKING_TERMS = {"ranking", "ranker", "learning to rank", "ltr", "relevance ranking"}
RECOMMENDER_TERMS = {"recommendation", "recommender", "personalization", "collaborative filtering"}
PRODUCT_TERMS = {"saas", "product", "platform", "marketplace", "consumer", "user-facing"}
CONSULTING_TERMS = {"it services", "consulting", "client project", "systems integrator"}
UNRELATED_TITLE_TERMS = {
    "accountant",
    "civil engineer",
    "hr manager",
    "human resources",
    "sales executive",
    "marketing manager",
    "operations manager",
    "mechanical engineer",
    "designer",
}

SKILL_GROUPS = {
    "python": {"python"},
    "nlp": {"nlp", "natural language processing", "spacy", "nltk"},
    "ml": {"machine learning", "scikit-learn", "tensorflow", "pytorch", "xgboost", "feature engineering"},
    "llm": {"llm", "fine-tuning llms", "transformers", "langchain", "lora", "rag"},
    "embeddings": {"embeddings", "sentence transformers", "vector search", "semantic search"},
    "vector_db": {"vector db", "faiss", "milvus", "pinecone", "weaviate", "qdrant", "chromadb"},
    "search": {"elasticsearch", "open search", "solr", "lucene", "search relevance"},
    "ranking": {"ranking", "learning to rank", "ltr", "recommender systems"},
}

PROFICIENCY_WEIGHT = {
    "beginner": 0.25,
    "intermediate": 0.55,
    "advanced": 0.8,
    "expert": 1.0,
}


@dataclass(frozen=True)
class CandidateFeatures:
    """Feature bundle used by filtering, ranking, and explanation generation."""

    candidate_id: str
    years_of_experience: float
    current_title: str
    ai_ml_relevance_score: float
    retrieval_search_score: float
    ranking_score: float
    recommendation_score: float
    product_company_score: float
    consulting_only_penalty: float
    skill_group_scores: dict[str, float] = field(default_factory=dict)
    avg_skill_proficiency: float = 0.0
    avg_relevant_skill_duration_months: float = 0.0
    open_to_work_flag: float = 0.0
    recruiter_response_rate: float = 0.0
    interview_completion_rate: float = 0.0
    profile_completeness_score: float = 0.0
    recency_score: float = 0.0
    github_activity_score: float = 0.0
    saved_by_recruiters_30d: float = 0.0
    notice_period_score: float = 0.0
    willing_to_relocate: float = 0.0
    location_preference_score: float = 0.0
    suspicious_skill_count_score: float = 0.0
    unrealistic_proficiency_penalty: float = 0.0
    timeline_consistency_penalty: float = 0.0
    honeypot_indicator_score: float = 0.0
    jd_experience_fit_score: float = 0.0
    disfavored_domain_penalty: float = 0.0


def extract_features(candidate: Candidate, reference_date: date = REFERENCE_DATE) -> CandidateFeatures:
    """Extract all deterministic features for one candidate."""
    cleaned = clean_candidate(candidate)
    profile = cleaned.get("profile") or {}
    signals = cleaned.get("redrob_signals") or {}
    text = candidate_text(cleaned)
    skills = cleaned.get("skills") or []
    skill_names = extract_skill_names(cleaned)
    career_history = cleaned.get("career_history") or []

    skill_group_scores = {
        group: _skill_group_score(skill_names, aliases) for group, aliases in SKILL_GROUPS.items()
    }
    relevant_skill_rows = [
        skill for skill in skills if isinstance(skill, dict) and _is_relevant_skill(skill.get("name", ""))
    ]
    current_title = str(profile.get("current_title") or "")

    return CandidateFeatures(
        candidate_id=str(cleaned.get("candidate_id", "")),
        years_of_experience=_to_float(profile.get("years_of_experience")),
        current_title=current_title,
        ai_ml_relevance_score=_term_score(text, AI_ML_TERMS, cap=8),
        retrieval_search_score=max(
            _term_score(text, RETRIEVAL_TERMS, cap=5),
            skill_group_scores["embeddings"],
            skill_group_scores["vector_db"],
            skill_group_scores["search"],
        ),
        ranking_score=max(_term_score(text, RANKING_TERMS, cap=3), skill_group_scores["ranking"]),
        recommendation_score=_term_score(text, RECOMMENDER_TERMS, cap=3),
        product_company_score=_product_company_score(cleaned, text),
        consulting_only_penalty=_consulting_only_penalty(cleaned, text),
        skill_group_scores=skill_group_scores,
        avg_skill_proficiency=_avg_skill_proficiency(skills),
        avg_relevant_skill_duration_months=_avg_duration(relevant_skill_rows),
        open_to_work_flag=1.0 if signals.get("open_to_work_flag") else 0.0,
        recruiter_response_rate=_clamp(_to_float(signals.get("recruiter_response_rate")), 0.0, 1.0),
        interview_completion_rate=_clamp(_to_float(signals.get("interview_completion_rate")), 0.0, 1.0),
        profile_completeness_score=_clamp(_to_float(signals.get("profile_completeness_score")) / 100.0, 0.0, 1.0),
        recency_score=_last_active_score(signals.get("last_active_date"), reference_date),
        github_activity_score=_normalize_github(signals.get("github_activity_score")),
        saved_by_recruiters_30d=_clamp(_to_float(signals.get("saved_by_recruiters_30d")) / 20.0, 0.0, 1.0),
        notice_period_score=_notice_period_score(signals.get("notice_period_days")),
        willing_to_relocate=1.0 if signals.get("willing_to_relocate") else 0.0,
        location_preference_score=_location_preference_score(profile, signals),
        suspicious_skill_count_score=_suspicious_skill_count_score(skills),
        unrealistic_proficiency_penalty=_unrealistic_proficiency_penalty(skills),
        timeline_consistency_penalty=_timeline_consistency_penalty(cleaned),
        honeypot_indicator_score=_honeypot_indicator_score(cleaned, text, current_title),
        jd_experience_fit_score=_jd_experience_fit_score(_to_float(profile.get("years_of_experience"))),
        disfavored_domain_penalty=_disfavored_domain_penalty(text, current_title),
    )


def _term_score(text: str, terms: set[str], cap: int) -> float:
    hits = sum(1 for term in terms if term in text)
    return _clamp(hits / cap, 0.0, 1.0)


def _skill_group_score(skill_names: set[str], aliases: set[str]) -> float:
    return 1.0 if skill_names.intersection(aliases) else 0.0


def _is_relevant_skill(name: str) -> bool:
    return any(name in aliases for aliases in SKILL_GROUPS.values())


def _avg_skill_proficiency(skills: list[Any]) -> float:
    values = [
        PROFICIENCY_WEIGHT.get(str(skill.get("proficiency", "")).lower(), 0.0)
        for skill in skills
        if isinstance(skill, dict)
    ]
    return sum(values) / len(values) if values else 0.0


def _avg_duration(skills: list[Any]) -> float:
    values = [
        _to_float(skill.get("duration_months"))
        for skill in skills
        if isinstance(skill, dict) and _to_float(skill.get("duration_months")) > 0
    ]
    return sum(values) / len(values) if values else 0.0


def _product_company_score(candidate: Candidate, text: str) -> float:
    profile = candidate.get("profile") or {}
    current_industry = str(profile.get("current_industry") or "")
    history = candidate.get("career_history") or []
    term_score = _term_score(text, PRODUCT_TERMS, cap=3)
    non_services_roles = sum(
        1
        for job in history
        if isinstance(job, dict) and "it services" not in str(job.get("industry", ""))
    )
    industry_score = 0.5 if "it services" not in current_industry and current_industry else 0.0
    return _clamp(max(term_score, industry_score, non_services_roles / 3.0), 0.0, 1.0)


def _consulting_only_penalty(candidate: Candidate, text: str) -> float:
    history = [job for job in candidate.get("career_history") or [] if isinstance(job, dict)]
    if not history:
        return 0.0
    consulting_roles = sum(
        1
        for job in history
        if any(term in str(job.get("industry", "")) or term in str(job.get("description", "")) for term in CONSULTING_TERMS)
    )
    if consulting_roles == len(history) and not _term_score(text, PRODUCT_TERMS, cap=1):
        return 1.0
    return 0.4 if consulting_roles / len(history) > 0.6 else 0.0


def _last_active_score(value: Any, reference_date: date) -> float:
    parsed = _parse_date(value)
    if parsed is None:
        return 0.0
    days = max((reference_date - parsed).days, 0)
    if days <= 30:
        return 1.0
    if days <= 90:
        return 0.75
    if days <= 180:
        return 0.45
    return 0.15


def _notice_period_score(value: Any) -> float:
    days = _to_float(value)
    if days <= 15:
        return 1.0
    if days <= 30:
        return 0.8
    if days <= 60:
        return 0.45
    if days <= 90:
        return 0.2
    return 0.0


def _location_preference_score(profile: dict[str, Any], signals: dict[str, Any]) -> float:
    country = str(profile.get("country") or "").lower()
    work_mode = str(signals.get("preferred_work_mode") or "").lower()
    if country == "india":
        return 1.0
    if work_mode in {"remote", "flexible"}:
        return 0.7
    return 0.4 if signals.get("willing_to_relocate") else 0.1


def _normalize_github(value: Any) -> float:
    score = _to_float(value)
    if score < 0:
        return 0.0
    return _clamp(score / 100.0, 0.0, 1.0)


def _suspicious_skill_count_score(skills: list[Any]) -> float:
    count = len(skills)
    if count <= 25:
        return 0.0
    if count <= 40:
        return 0.4
    return 1.0


def _unrealistic_proficiency_penalty(skills: list[Any]) -> float:
    if len(skills) < 8:
        return 0.0
    expert_like = sum(
        1
        for skill in skills
        if isinstance(skill, dict) and skill.get("proficiency") in {"advanced", "expert"}
    )
    ratio = expert_like / len(skills)
    return _clamp((ratio - 0.75) / 0.25, 0.0, 1.0)


def _timeline_consistency_penalty(candidate: Candidate) -> float:
    profile_years = _to_float((candidate.get("profile") or {}).get("years_of_experience"))
    total_months = sum(
        _to_float(job.get("duration_months"))
        for job in candidate.get("career_history") or []
        if isinstance(job, dict)
    )
    if not profile_years or not total_months:
        return 0.0
    derived_years = total_months / 12.0
    return 1.0 if abs(profile_years - derived_years) > 5.0 else 0.0


def _honeypot_indicator_score(candidate: Candidate, text: str, current_title: str) -> float:
    unrelated_title = any(term in current_title for term in UNRELATED_TITLE_TERMS)
    ai_mentions = _term_score(text, AI_ML_TERMS | RETRIEVAL_TERMS | RANKING_TERMS, cap=4)
    skills = candidate.get("skills") or []
    many_ai_skills = sum(
        1
        for skill in skills
        if isinstance(skill, dict) and _is_relevant_skill(str(skill.get("name", "")))
    ) >= 6
    if unrelated_title and (ai_mentions > 0.5 or many_ai_skills):
        return 1.0
    if "chatgpt" in text and ai_mentions < 0.35:
        return 0.4
    return 0.0


def _jd_experience_fit_score(years: float) -> float:
    if 5.0 <= years <= 9.0:
        return 1.0
    if 4.0 <= years < 5.0 or 9.0 < years <= 11.0:
        return 0.7
    if 3.0 <= years < 4.0 or 11.0 < years <= 14.0:
        return 0.35
    return 0.05


def _disfavored_domain_penalty(text: str, current_title: str) -> float:
    disfavored = {"computer vision", "speech", "robotics", "research scientist", "applied scientist"}
    has_ir = any(term in text for term in RETRIEVAL_TERMS | RANKING_TERMS | RECOMMENDER_TERMS)
    if any(term in current_title or term in text for term in disfavored) and not has_ir:
        return 1.0
    if any(term in current_title for term in disfavored):
        return 0.45
    return 0.0


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except ValueError:
        return None


def _to_float(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return 0.0


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
