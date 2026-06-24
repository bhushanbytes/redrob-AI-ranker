"""Candidate cleaning and text extraction helpers."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any


Candidate = dict[str, Any]

_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_EDGE_RE = re.compile(r"^[^\w+#.]+|[^\w+#.]+$")

TITLE_ALIASES = {
    "ml engineer": "machine learning engineer",
    "ai engineer": "artificial intelligence engineer",
    "sde": "software development engineer",
    "swe": "software engineer",
    "data scientist ii": "data scientist",
    "sr software engineer": "senior software engineer",
    "sr. software engineer": "senior software engineer",
}

SKILL_ALIASES = {
    "py": "python",
    "pyspark": "spark",
    "scikit learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "llms": "llm",
    "large language models": "llm",
    "fine tuning llms": "fine-tuning llms",
    "fine-tuning": "fine-tuning llms",
    "vector databases": "vector db",
    "vector database": "vector db",
    "elastic search": "elasticsearch",
    "opensearch": "open search",
}


def normalize_text(value: Any) -> str:
    """Lowercase text and collapse whitespace while safely handling nulls."""
    if value is None:
        return ""
    text = str(value).replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u00a0", " ")
    return _WHITESPACE_RE.sub(" ", text).strip().lower()


def normalize_title(title: Any) -> str:
    """Normalize a job title into a comparable lowercase string."""
    normalized = normalize_text(title)
    normalized = _NON_ALNUM_EDGE_RE.sub("", normalized)
    return TITLE_ALIASES.get(normalized, normalized)


def normalize_skill_name(skill_name: Any) -> str:
    """Normalize a skill name while preserving meaningful tokens like c++."""
    normalized = normalize_text(skill_name).replace("_", " ")
    normalized = _NON_ALNUM_EDGE_RE.sub("", normalized)
    return SKILL_ALIASES.get(normalized, normalized)


def clean_candidate(candidate: Candidate) -> Candidate:
    """Return a cleaned candidate copy without dropping any records."""
    cleaned = deepcopy(candidate)
    profile = cleaned.setdefault("profile", {})
    profile["headline"] = normalize_text(profile.get("headline"))
    profile["summary"] = normalize_text(profile.get("summary"))
    profile["location"] = normalize_text(profile.get("location"))
    profile["country"] = normalize_text(profile.get("country"))
    profile["current_title"] = normalize_title(profile.get("current_title"))
    profile["current_industry"] = normalize_text(profile.get("current_industry"))

    for job in cleaned.setdefault("career_history", []) or []:
        if not isinstance(job, dict):
            continue
        job["title"] = normalize_title(job.get("title"))
        job["description"] = normalize_text(job.get("description"))
        job["industry"] = normalize_text(job.get("industry"))
        job["company"] = normalize_text(job.get("company"))

    normalized_skills = []
    for skill in cleaned.get("skills") or []:
        if isinstance(skill, dict):
            item = dict(skill)
            item["name"] = normalize_skill_name(item.get("name"))
            item["proficiency"] = normalize_text(item.get("proficiency"))
            normalized_skills.append(item)
        elif skill:
            normalized_skills.append({"name": normalize_skill_name(skill)})
    cleaned["skills"] = normalized_skills
    cleaned.setdefault("redrob_signals", {})
    cleaned.setdefault("education", [])
    return cleaned


def candidate_text(candidate: Candidate) -> str:
    """Build a compact text blob from fields relevant to matching JD intent."""
    profile = candidate.get("profile") or {}
    career_history = candidate.get("career_history") or []
    skills = candidate.get("skills") or []
    parts: list[str] = [
        profile.get("current_title", ""),
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_industry", ""),
    ]
    for job in career_history:
        if isinstance(job, dict):
            parts.extend([job.get("title", ""), job.get("description", ""), job.get("industry", "")])
    for skill in skills:
        if isinstance(skill, dict):
            parts.append(skill.get("name", ""))
        elif skill:
            parts.append(str(skill))
    return normalize_text(" ".join(str(part) for part in parts if part))


def extract_skill_names(candidate: Candidate) -> set[str]:
    """Return normalized skill names for a candidate."""
    names: set[str] = set()
    for skill in candidate.get("skills") or []:
        if isinstance(skill, dict):
            name = normalize_skill_name(skill.get("name"))
        else:
            name = normalize_skill_name(skill)
        if name:
            names.add(name)
    return names
