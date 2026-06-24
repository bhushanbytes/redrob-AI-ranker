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
SAMPLE_PATH = ROOT / "samples" / "sample_candidates.json"


def parse_candidates(contents: bytes) -> list[dict]:
    """Parse a JSON array or JSONL upload and validate its candidate records."""
    text = contents.decode("utf-8").strip()
    if not text:
        return []

    if text.startswith("["):
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError("JSON input must contain an array of candidates.")
        candidates = parsed
    else:
        candidates = _parse_jsonl(text)

    for index, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict) or not candidate.get("candidate_id"):
            raise ValueError(f"Candidate {index} is missing candidate_id.")
    if len(candidates) > MAX_CANDIDATES:
        raise ValueError(f"Upload at most {MAX_CANDIDATES} candidates.")
    return candidates


def _parse_jsonl(text: str) -> list[dict]:
    """Parse newline-delimited candidate objects."""
    candidates: list[dict] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
        candidates.append(candidate)
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

st.markdown(
    """
    <style>
    .main .block-container { padding-top: 2rem; max-width: 1180px; }
    .hero {
        border: 1px solid #e7e9ef;
        border-radius: 8px;
        padding: 1.25rem 1.4rem;
        background: linear-gradient(135deg, #ffffff 0%, #f7f8fb 58%, #fff4ed 100%);
    }
    .hero h1 { margin: 0 0 .35rem 0; font-size: 2.1rem; line-height: 1.15; }
    .hero p { margin: 0; color: #4b5563; font-size: 1rem; }
    .signal {
        border: 1px solid #eceef3;
        border-radius: 8px;
        padding: .85rem .95rem;
        background: #ffffff;
        min-height: 92px;
    }
    .signal small { color: #6b7280; text-transform: uppercase; letter-spacing: .04em; }
    .signal strong { display: block; margin-top: .35rem; font-size: 1.15rem; color: #111827; }
    div[data-testid="stMetricValue"] { font-size: 1.45rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def reset_results() -> None:
    """Clear cached ranking output when the upload changes."""
    st.session_state.pop("result_rows", None)
    st.session_state.pop("result_csv", None)


st.markdown(
    """
    <div class="hero">
      <h1>Redrob AI Candidate Ranker</h1>
      <p>Deterministic CPU ranking sandbox for Senior AI Engineer discovery. Upload JSON/JSONL, review ranked candidates, and export a submission-style CSV.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
signal_cols = st.columns(4)
signals = [
    ("Primary intent", "Search, retrieval, ranking"),
    ("Not embedding-only", "Engineered scoring"),
    ("Risk handling", "Honeypot penalties"),
    ("Reviewer mode", "100-record sandbox"),
]
for column, (label, value) in zip(signal_cols, signals, strict=True):
    column.markdown(
        f'<div class="signal"><small>{label}</small><strong>{value}</strong></div>',
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.header("Sandbox Run")
    st.caption("Use the sample file or upload up to 100 candidates.")
    if SAMPLE_PATH.exists():
        st.download_button(
            "Download sample JSON",
            SAMPLE_PATH.read_bytes(),
            file_name="sample_candidates.json",
            mime="application/json",
            use_container_width=True,
        )
    upload = st.file_uploader(
        "Candidate file",
        type=["jsonl", "json"],
        on_change=reset_results,
    )
    st.divider()
    st.caption("Expected output columns: candidate_id, rank, score, reasoning.")

uploaded_candidates: list[dict] = []
if upload is not None:
    try:
        uploaded_candidates = parse_candidates(upload.getvalue())
    except (UnicodeDecodeError, ValueError) as exc:
        st.error(str(exc))

left, right = st.columns([0.62, 0.38])

with left:
    st.subheader("Rank Candidates")
    if upload is None:
        st.info("Upload a JSON or JSONL candidate sample from the sidebar to start.")
    elif not uploaded_candidates:
        st.warning("The uploaded file contains no candidate records.")
    else:
        profile_titles = [
            str(candidate.get("profile", {}).get("current_title", "Unknown"))
            for candidate in uploaded_candidates
        ]
        st.caption(f"Loaded {len(uploaded_candidates)} candidates. Most common visible title: {max(set(profile_titles), key=profile_titles.count)}.")
        if st.button("Run deterministic ranker", type="primary", use_container_width=True):
            with st.spinner("Scoring career, skills, behavior, logistics, and authenticity signals..."):
                result_rows, result_csv = build_csv(uploaded_candidates)
            st.session_state["result_rows"] = result_rows
            st.session_state["result_csv"] = result_csv

with right:
    st.subheader("Run Status")
    ranked_count = len(st.session_state.get("result_rows", []))
    filtered_count = max(0, len(uploaded_candidates) - ranked_count) if uploaded_candidates else 0
    top_score = (
        st.session_state["result_rows"][0]["score"]
        if st.session_state.get("result_rows")
        else "Pending"
    )
    metric_cols = st.columns(3)
    metric_cols[0].metric("Loaded", len(uploaded_candidates))
    metric_cols[1].metric("Ranked", ranked_count)
    metric_cols[2].metric("Top score", top_score)
    if ranked_count:
        st.caption(f"Conservative filter excluded {filtered_count} low-relevance profiles from this sample.")

rows = st.session_state.get("result_rows")
if rows:
    st.success("Ranking complete. Download the CSV or inspect candidate-specific reasoning below.")
    results_tab, reasoning_tab, method_tab = st.tabs(["Ranked CSV", "Reasoning", "Method"])
    with results_tab:
        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
            column_config={
                "score": st.column_config.NumberColumn("score", format="%.6f"),
                "reasoning": st.column_config.TextColumn("reasoning", width="large"),
            },
        )
        st.download_button(
            "Download ranked CSV",
            st.session_state["result_csv"],
            file_name="ranked_candidates.csv",
            mime="text/csv",
            type="primary",
        )
    with reasoning_tab:
        for row in rows[:10]:
            st.markdown(f"**#{row['rank']} | {row['candidate_id']} | {row['score']:.6f}**")
            st.write(row["reasoning"])
    with method_tab:
        st.write(
            "The ranker combines career relevance, normalized skills, behavioral engagement, logistics, and authenticity checks. "
            "It intentionally rewards search, retrieval, ranking, recommendation, production ML, and product-company experience while reducing scores for suspicious or unrelated profiles."
        )
