"""Streaming utilities for candidate JSONL files."""

from __future__ import annotations

import json
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Iterator


Candidate = dict[str, Any]


def load_candidates(path: str | Path, limit: int | None = None) -> Iterator[Candidate]:
    """Yield candidate records from a JSONL file without loading it all at once.

    Args:
        path: Path to the candidate JSONL file.
        limit: Optional maximum number of records to yield, useful for EDA.

    Raises:
        ValueError: If a line cannot be decoded as JSON.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        lines: Iterable[str] = handle if limit is None else islice(handle, limit)
        for line_number, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc


def load_candidates_list(path: str | Path, limit: int | None = None) -> list[Candidate]:
    """Materialize candidate records when a notebook or test needs a list."""
    return list(load_candidates(path, limit=limit))


def _safe_get(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key, default)
    return value


def summarize_candidates(
    candidates: Iterable[Candidate],
    top_n: int = 20,
) -> dict[str, Any]:
    """Return notebook-friendly EDA summaries over an iterable of candidates."""
    title_counts: Counter[str] = Counter()
    skill_counts: Counter[str] = Counter()
    location_counts: Counter[str] = Counter()
    experience_values: list[float] = []
    missing_counts: Counter[str] = Counter()
    behavior_values: dict[str, list[float]] = {
        "profile_completeness_score": [],
        "recruiter_response_rate": [],
        "interview_completion_rate": [],
        "github_activity_score": [],
        "saved_by_recruiters_30d": [],
    }

    total = 0
    for candidate in candidates:
        total += 1
        profile = candidate.get("profile") or {}
        signals = candidate.get("redrob_signals") or {}
        skills = candidate.get("skills") or []

        title = profile.get("current_title")
        if title:
            title_counts[str(title).strip().lower()] += 1
        else:
            missing_counts["profile.current_title"] += 1

        location = profile.get("location") or profile.get("country")
        if location:
            location_counts[str(location).strip().lower()] += 1
        else:
            missing_counts["profile.location"] += 1

        experience = profile.get("years_of_experience")
        if isinstance(experience, (int, float)):
            experience_values.append(float(experience))
        else:
            missing_counts["profile.years_of_experience"] += 1

        if not profile.get("summary"):
            missing_counts["profile.summary"] += 1
        if not skills:
            missing_counts["skills"] += 1

        for skill in skills:
            name = skill.get("name") if isinstance(skill, dict) else skill
            if name:
                skill_counts[str(name).strip().lower()] += 1

        for key in behavior_values:
            value = signals.get(key)
            if isinstance(value, (int, float)):
                behavior_values[key].append(float(value))
            else:
                missing_counts[f"redrob_signals.{key}"] += 1

    return {
        "record_count": total,
        "top_titles": title_counts.most_common(top_n),
        "top_skills": skill_counts.most_common(top_n),
        "top_locations": location_counts.most_common(top_n),
        "experience_distribution": _numeric_summary(experience_values),
        "behavioral_signal_distribution": {
            key: _numeric_summary(values) for key, values in behavior_values.items()
        },
        "missing_values": missing_counts.most_common(),
    }


def _numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    """Summarize numeric values without requiring pandas."""
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
        "p50": ordered[len(ordered) // 2],
    }
