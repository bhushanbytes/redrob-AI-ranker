"""Generate the challenge Top-100 candidate submission file."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

from src.loader import load_candidates
from src.ranker import generate_reason, rank_candidates


DATA_PATH = Path("data/candidates.jsonl")
OUTPUT_PATH = Path("submissions/bhushankale888_6378.csv")
FIELDNAMES = ["candidate_id", "rank", "score", "reasoning"]


def main() -> None:
    """Rank candidates and write candidate_id, rank, and score to CSV."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    candidates_by_id: dict[str, dict] = {}
    ranked = rank_candidates(_remember_candidates(candidates_by_id), top_n=100)
    rows = [
        {
            "candidate_id": item.candidate_id,
            "rank": item.rank,
            "score": round(item.score - item.rank * 0.000001, 6),
            "reasoning": generate_reason(candidates_by_id[item.candidate_id], item),
        }
        for item in ranked
    ]

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {OUTPUT_PATH}")


def _remember_candidates(candidates_by_id: dict[str, dict]) -> Iterator[dict]:
    """Stream candidates while retaining originals for final Top-100 reasoning."""
    for candidate in load_candidates(DATA_PATH):
        candidate_id = str(candidate.get("candidate_id", ""))
        if candidate_id:
            candidates_by_id[candidate_id] = candidate
        yield candidate


if __name__ == "__main__":
    main()
