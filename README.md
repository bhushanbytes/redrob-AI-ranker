# Redrob AI Ranker

Deterministic CPU-only ranker for the Redrob Intelligent Candidate Discovery & Ranking Challenge.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\pip install -r requirements.txt
```

## Reproduce Submission

Place `candidates.jsonl` at `data/candidates.jsonl`, then run:

```powershell
python run_ranker.py
```

Output:

```text
submissions/bhushankale888_6378.csv
```

Validate locally:

```powershell
python validate_submission.py submissions\bhushankale888_6378.csv
```

## Small-Sample Sandbox

Run the Streamlit demo locally:

```powershell
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

The demo accepts a JSONL upload containing up to 100 candidates and produces
a ranked CSV using the same filtering, feature extraction, scoring, and
reasoning pipeline as the full submission.

## Method

The ranker is a modular, feature-engineered pipeline rather than an embedding-only system. It streams JSONL candidates, normalizes profile text, extracts career/skill/behavior/logistics/authenticity features, conservatively filters unrelated profiles, and combines weighted component scores.

The scoring is tuned to the JD intent: production retrieval, search, ranking, recommendation systems, strong Python/ML background, product-company experience, 5-9 year senior engineer fit, India/Pune/Noida-friendly logistics, and active/reachable Redrob behavior. Authenticity features down-rank keyword stuffing, unrealistic skill patterns, timeline inconsistencies, consulting-only history, and honeypot-like profiles.

No network calls, hosted LLM APIs, or GPU inference are used during ranking.
