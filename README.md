# Bias-Aware Scoring Engine 
[![CI](https://github.com/25-26J-433-RP/bias-aware-scoring-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/25-26J-433-RP/bias-aware-scoring-engine/actions/workflows/ci.yml)

This is the backend service for the Bias-Aware Sinhala Essay Grading Research Project, built using FastAPI + Python 3.14.

It provides endpoints for:

Sinhala essay scoring (Phase 4 baseline)

English TF-IDF scoring

Fairness metrics (SPD, DIR, EOD)

Bias-adjusted scoring

# 📦 Get started
1️. Create virtual environment
python -m venv venv


Activate it:

Windows

venv\Scripts\activate


Mac/Linux

source venv/bin/activate

2️. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# ▶️ Run the backend
uvicorn app.main:app --reload


Server will start at:

http://127.0.0.1:8000


Open API docs:

http://127.0.0.1:8000/docs

# 📝 API Endpoints
Sinhala essay scoring
POST /score-sinhala


Example body:

{
  "text": "මගේ අම්මා ගැන මම ලියමි...",
  "grade": 7,
  "topic": "මාගේ අම්මා"
}

# 🧪 Run tests
pytest


Verbose:

pytest -vv

# 📁 Project structure
app/
 ├── main.py              # API entry
 ├── scorer.py            # English scoring
 ├── sinhala_baseline.py  # Sinhala rule-based scorer
 ├── fairness.py          # Fairness metrics
 └── schemas.py           # Request/response models

tests/
 ├── test_api.py
 └── test_fairness.py

# 🐳 Run with Docker

Build:

docker build -t scoring-engine .


Run:

docker run -p 8000:8000 scoring-engine

# 🔮 Future development

Transformer-based Sinhala content model (Phase 5)

Full fairness-aware scoring module

Dashboard & reporting

Dataset normalization + training pipeline

# 👥 Contributors

Nuwani Fonseka

# Research Group 433 — SLIIT 2025