# Bias-Aware Scoring Engine 
[![CI](https://github.com/25-26J-433-RP/bias-aware-scoring-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/25-26J-433-RP/bias-aware-scoring-engine/actions/workflows/ci.yml)

A semantic essay scoring microservice with **built-in fairness mechanisms** for Sinhala student essays, providing equitable evaluation for dyslexic learners.

Built with FastAPI + PyTorch, deployed on Google Cloud Run.

---

## ✨ Key Features

### 🧠 ML-Powered Scoring
- **XLM-RoBERTa Large** fine-tuned for Sinhala essays (multi-head regression)
- **Grade-aware scoring** (Grades 3-8) with difficulty calibration
- **Rubric-based output**: Richness (5), Organization (6), Technical Skills (3), Total (14)
- Model hosted on [HuggingFace Hub](https://huggingface.co/akura-official/xlm-roberta-large-sinhala-multihead)

### ⚖️ Fairness & Bias Mitigation
- **Statistical Parity Difference (SPD)** detection
- **Disparate Impact Ratio (DIR)** monitoring (80% rule compliance)
- **Conditional post-processing mitigation** aligned with [IBM AIF360](https://github.com/Trusted-AI/AIF360)
- **Grade-specific calibration** based on historical bias analysis
- **Full transparency logging** with `MitigationRecord` audit trail

### 🏗️ Architecture
- **Lazy model loading** for fast cold starts on Cloud Run
- **CI-safe testing** with `DISABLE_ML=1` flag
- **RESTful API** with OpenAPI documentation
- **Firebase integration** for fairness metrics storage

---

## 📦 Quick Start

### Prerequisites
- Python 3.11+
- pip

### 1️⃣ Setup Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 2️⃣ Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3️⃣ Run the Server
```bash
uvicorn app.main:app --reload --port 8001
```

- **Server:** http://127.0.0.1:8001
- **API Docs:** http://127.0.0.1:8001/docs

> 💡 On first run, the model (~1.4GB) downloads automatically from HuggingFace Hub.

---

## 📝 API Endpoints

### 🔹 Health Check
```http
GET /health
```

### 🔹 Sinhala Essay Scoring (ML)
```http
POST /score-sinhala-ml
```

**Request:**
```json
{
  "text": "මගේ පරිසරය පිළිබඳ මගේ අදහස් මෙසේ වේ...",
  "grade": 7,
  "topic": "මගේ පරිසරය",
  "dyslexic_flag": true,
  "error_tags": []
}
```

**Response:**
```json
{
  "score": 72.14,
  "rubric": {
    "richness_5": 3.45,
    "organization_6": 4.02,
    "technical_3": 2.63,
    "total_14": 10.10,
    "fairness_report": {
      "mitigation_applied": true,
      "original_score_100": 68.57,
      "adjusted_score_100": 72.14,
      "protected_attribute": "dyslexic_flag",
      "method": "Conditional Post-Processing (AIF360-aligned)"
    }
  },
  "details": {
    "dyslexic_flag": true,
    "detected_grade": 7,
    "model": "✅ RETRAINED MODEL (Cloud)"
  }
}
```

### 🔹 Batch Fairness Evaluation
```http
POST /fairness-eval
```

---

## 🧪 Testing

### Run Tests (CI Mode - No ML Model)
```bash
# Windows
set DISABLE_ML=1
pytest -vv

# Mac/Linux
DISABLE_ML=1 pytest -vv
```

### Test Coverage
```bash
pytest --cov=app --cov-report=html
```

---

## 🐳 Docker

### Build
```bash
docker build -t bias-aware-scoring-engine .
```

### Run
```bash
docker run -p 8000:8000 -e PORT=8000 bias-aware-scoring-engine
```

---

## ☁️ Cloud Deployment

Deployed on **Google Cloud Run** (europe-west1):
- **URL:** https://bias-aware-scoring-engine-651457725719.europe-west1.run.app
- **Memory:** 8 GiB
- **CPU:** 2
- **Concurrency:** 80

### Deployment via GitHub Actions
Push to `main` branch triggers automatic deployment.

---

## 📁 Project Structure

```
bias-aware-scoring-engine/
├── app/
│   ├── main.py                    # FastAPI application & routes
│   ├── sinhala_ml_v2.py           # ML scoring with lazy loading
│   ├── model_multitask_xlmr.py    # Custom HuggingFace model class
│   ├── mitigation.py              # Fairness mitigation engine
│   ├── fairness.py                # SPD, DIR, EOD calculations
│   ├── grade_detector.py          # Auto grade detection
│   ├── schemas.py                 # Pydantic request/response models
│   ├── scorer.py                  # English scoring (legacy)
│   └── sinhala_baseline.py        # Rule-based scorer (fallback)
├── analysis/
│   └── firestore_fairness_eval.py # Batch fairness evaluation script
├── tests/
│   ├── test_api.py
│   └── test_fairness.py
├── docs/
│   └── THESIS_TODO.md             # Thesis completion checklist
├── Dockerfile
├── pyproject.toml
└── README.md
```

---

## ⚖️ Fairness Metrics

### Measured Metrics (Per Grade)
| Grade | SPD | DIR | Mitigation Active |
|-------|-----|-----|-------------------|
| 3 | -0.05 | 0.92 | ⚪ No |
| 4 | -0.08 | 0.88 | ⚪ No |
| 5 | -0.12 | 0.85 | 🟢 Yes |
| 6 | -0.09 | 0.87 | ⚪ No |
| 7 | -0.11 | 0.83 | 🟢 Yes |
| 8 | -0.15 | 0.78 | 🟢 Yes |

### Thresholds
- **SPD Threshold:** |SPD| > 0.1 triggers mitigation
- **DIR Threshold:** DIR < 0.8 triggers mitigation (EEOC 80% rule)

---

## 🔮 Research Objectives (In Progress)

| Objective | Status |
|-----------|--------|
| Semantic-aware scoring pipeline | ✅ Complete |
| Bias detection (SPD, DIR) | ✅ Complete |
| Post-processing bias mitigation | ✅ Complete |
| Grade-aware calibration | ✅ Complete |
| Educator fairness dashboard | ✅ Complete |
| Accuracy validation (Pearson r ≥ 0.90) | ⏳ In Progress |
| Performance testing (P95 < 2s) | ⏳ In Progress |
| Load testing (10k essays/day) | ⏳ Pending |

See [`docs/THESIS_TODO.md`](docs/THESIS_TODO.md) for detailed completion checklist.

---

## 📚 References

- [IBM AI Fairness 360](https://github.com/Trusted-AI/AIF360)
- [XLM-RoBERTa](https://huggingface.co/docs/transformers/model_doc/xlm-roberta)
- [Hardt et al., 2016 - Equality of Opportunity in Supervised Learning](https://arxiv.org/abs/1610.02413)
- [EEOC 80% Rule](https://www.eeoc.gov/laws/guidance/uniform-guidelines-employee-selection-procedures)

---

## 👥 Contributors

- **Nuwani Fonseka** - Bias-Aware Scoring Engine Lead
- **Sadeesha Perera** - Research Collaboration

---

## 📄 License

This project is part of academic research at [University Name]. All rights reserved.