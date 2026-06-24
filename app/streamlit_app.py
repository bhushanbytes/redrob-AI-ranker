"""Small-sample Streamlit sandbox for the Redrob ranking pipeline."""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ranker import generate_reason, rank_candidates


MAX_CANDIDATES = 100


def parse_jsonl(contents: bytes) -> list[dict]:
    """Parse and validate a bounded candidate JSONL upload."""
    candidates: list[dict] = []
    for line_number, raw_line in enumerate(contents.decode("utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
        if not isinstance(candidate, dict) or not candidate.get("candidate_id"):
            raise ValueError(f"Line {line_number} is missing candidate_id.")
        candidates.append(candidate)
        if len(candidates) > MAX_CANDIDATES:
            raise ValueError(f"Upload at most {MAX_CANDIDATES} candidates.")
    return candidates


def build_csv(candidates: list[dict]) -> tuple[list[dict], str]:
    """Rank uploaded candidates and return display rows plus downloadable CSV."""
    by_id = {str(candidate["candidate_id"]): candidate for candidate in candidates}
    ranked = rank_candidates(candidates, top_n=min(100, len(candidates)))
    rows = [
        {
            "candidate_id": item.candidate_id,
            "rank": item.rank,
            "score": round(item.score - item.rank * 0.000001, 6),
            "reasoning": generate_reason(by_id[item.candidate_id], item),
        }
        for item in ranked
    ]
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["candidate_id", "rank", "score", "reasoning"],
    )
    writer.writeheader()
    writer.writerows(rows)
    return rows, output.getvalue()


st.set_page_config(page_title="Redrob AI Ranker", layout="wide")
st.title("Redrob AI Candidate Ranker")
st.caption("CPU-only deterministic ranking for a JSONL sample of up to 100 candidates.")

upload = st.file_uploader("Candidate JSONL", type=["jsonl", "json"])

if upload is not None:
    try:
        uploaded_candidates = parse_jsonl(upload.getvalue())
        if not uploaded_candidates:
            st.warning("The uploaded file contains no candidate records.")
        elif st.button("Rank candidates", type="primary"):
            with st.spinner("Ranking candidates..."):
                result_rows, result_csv = build_csv(uploaded_candidates)
            st.success(f"Ranked {len(result_rows)} relevant candidates.")
            st.dataframe(result_rows, use_container_width=True, hide_index=True)
            st.download_button(
                "Download ranked CSV",
                result_csv,
                file_name="ranked_candidates.csv",
                mime="text/csv",
            )
    except (UnicodeDecodeError, ValueError) as exc:
        st.error(str(exc))
