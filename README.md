# Bias-Aware Scoring Engine 
[![CI](https://github.com/25-26J-433-RP/bias-aware-scoring-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/25-26J-433-RP/bias-aware-scoring-engine/actions/workflows/ci.yml)

Built using FastAPI + Python, this service provides semantic essay scoring and fairness-aware evaluation for Sinhala student essays, with special consideration for dyslexic learners.

# ✨ Key Features

✅ Transformer-based Sinhala essay scoring (Phase 5)

Trained XLM-RoBERTa Large (Sinhala, multi-head) model

Content-focused scoring (richness, organization, technical quality)

✅ Rule-based Sinhala baseline scorer (Phase 4)

✅ English TF-IDF scoring

✅ Fairness metrics (system-level)

Statistical Parity Difference (SPD)

Disparate Impact Ratio (DIR)

Equal Opportunity Difference (EOD)

✅ CI-safe inference

Heavy ML model disabled automatically during tests

✅ Dockerized & CI-tested

# 📦 Getting Started
# 1️⃣ Create virtual environment
python -m venv venv


Activate:

Windows

venv\Scripts\activate


Mac / Linux

source venv/bin/activate

# 2️⃣ Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# ▶️ Run the Backend (Real ML Inference)
uvicorn app.main:app --reload


Server:

http://127.0.0.1:8000


API Docs:

http://127.0.0.1:8000/docs

# 🧠 ML Model Setup (Important)

The trained Sinhala essay scoring model is hosted on Hugging Face Hub:

👉 https://huggingface.co/akura-official/xlm-roberta-large-sinhala-multihead

On first run:

The model is downloaded automatically

Cached locally

No training code required

# 🧪 CI / Development Mode (No ML Loading)

For testing or frontend development:

set DISABLE_ML=1   # Windows
pytest


This disables the transformer model and uses a CI-safe dummy scorer.

# 📝 API Endpoints
🔹 Sinhala Essay Scoring (ML)

POST /score-sinhala-ml

Example request:

{
  "text": "මගේ පරිසරය පිළිබඳ මගේ අදහස් මෙසේ වේ...",
  "grade": 7,
  "topic": "මගේ පරිසරය",
  "dyslexic_flag": false,
  "error_tags": []
}


Example response:

{
  "score": 82.36,
  "rubric": {
    "richness_5": 4.14,
    "organization_6": 4.47,
    "technical_3": 2.79,
    "total_14": 11.53
  },
  "details": {
    "grade": 7,
    "topic": "මගේ පරිසරය",
    "model": "xlm-roberta-large-sinhala-multihead"
  },
  "fairness_report": null
}


ℹ️ Fairness reports are system-level and generated after evaluating multiple essays.

# 🧪 Run Tests
set DISABLE_ML=1   # Windows
pytest


Verbose:

pytest -vv

# 📁 Project Structure
app/
 ├── main.py                 # API entry
 ├── scorer.py               # English scoring
 ├── sinhala_baseline.py     # Sinhala rule-based scorer
 ├── sinhala_ml_v2.py        # Transformer-based Sinhala scorer
 ├── fairness.py             # Fairness metrics
 └── schemas.py              # Request/response models

tests/
 ├── test_api.py
 └── test_fairness.py

.github/workflows/
 └── ci.yml

# 🐳 Run with Docker

Build:

docker build -t bias-aware-scoring-engine .


Run:

docker run -p 8000:8000 bias-aware-scoring-engine

# 🔮 Future Work

🔄 Fairness mitigation (Reweighing, Adversarial Debiasing)

📊 Educator-facing fairness dashboards & reports

🧩 Dyslexia-aware semantic normalization

📚 Expanded Sinhala essay dataset

📈 Longitudinal bias monitoring

# 👥 Contributors

Nuwani Fonseka
Sadeesha Perera