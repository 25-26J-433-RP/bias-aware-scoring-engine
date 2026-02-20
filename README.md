# Bias-Aware Sinhala Essay Scoring Engine
[![CI](https://github.com/25-26J-433-RP/bias-aware-scoring-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/25-26J-433-RP/bias-aware-scoring-engine/actions/workflows/ci.yml)

A state-of-the-art semantic essay scoring microservice designed for the **Sinhala language**, featuring built-in **bias detection and fairness mitigation** for dyslexic learners.

Built with FastAPI + PyTorch, deployed on Google Cloud Run.

---

## ✨ Key Features

### 🧠 ML-Powered Scoring
- **XLM-RoBERTa Large** fine-tuned for Sinhala essays (multi-head regression).
- **Grade-aware scoring** (Grades 3-8) with difficulty calibration.
- **Rubric-based output**: Richness (5), Organization (6), Technical Skills (3), Total (14).
- Model hosted on [HuggingFace Hub](https://huggingface.co/akura-official/xlm-roberta-large-sinhala-multihead).

### ⚖️ Fairness & Bias Mitigation
- **Statistical Parity Difference (SPD)** detection.
- **Disparate Impact Ratio (DIR)** monitoring (80% rule compliance).
- **Conditional post-processing mitigation** aligned with [IBM AIF360](https://github.com/Trusted-AI/AIF360).
- **Grade-specific calibration** based on historical bias analysis of 446 empirical samples.
- **Full transparency logging** with `MitigationRecord` audit trail.

### 🏗️ Architecture
- **Strict Offline Loading**: Models pre-downloaded during build to avoid Hugging Face 429 errors.
- **Memory Optimized**: Single-pass encoder execution to prevent Out-of-Memory (OOM) on Cloud Run.
- **Lazy model loading** for fast cold starts (if running outside Docker).
- **CI-safe testing** with `DISABLE_ML=1` flag.
- **RESTful API** with OpenAPI documentation.
- **Firebase integration** for fairness metrics storage.

---

## 🔬 Technical Deep Dive

### Model Architecture: `SinhalaMultiHeadRegressor`
The core of the system is a custom **Multi-Task Learning (MTL)** architecture:
1. **Transformer Encoder**: `xlm-roberta-large` (24-layers, 1024-hidden).
2. **Grade-Aware Layer**: A dedicated `nn.Embedding(10, hidden)` layer that injects the student's grade level directly into the CLS token representation.
3. **Multi-Head Regression**: Four distinct regression heads predicting scores simultaneously.

### Hybrid Scoring Pipeline
1. **Phase 1 (ML)**: Semantic analysis via the XLM-R encoder for base scores.
2. **Phase 2 (Rule-Based)**: `RubricEvaluator` applies 6 specific Sinhala marking scheme rules (Punctuation, Verb Markers, Pacing, etc.).
3. **Phase 3 (Theme)**: Dual-signal relevance check (60% Keyword frequency + 40% Semantic Cosine Similarity).

### Fairness Mitigation Logic
Mitigation triggers only if $|SPD| > 0.1$ OR $DIR < 0.8$. We apply **Component-Specific Calibration** using exact multipliers calculated from Firestore data.

**Formula:**
$$Score_{adjusted} = Score_{original} \times \frac{MeanScore_{non\_dyslexic}}{MeanScore_{dyslexic}}$$

---

## 🧪 Testing & Validation Results

### 1. Research Validation (Same-Essay-Different-Flag Tests)
We use a "counterfactual" test suite where the same essay text is sent twice—once with `dyslexic_flag: false` and once with `true`.

| Test Case | Scenario | Result | Status |
| :--- | :--- | :--- | :--- |
| **T-INT-01 (Grade 7)** | SPD 0.242 (Positive) | Non-Dyslexic: **61** / Dyslexic: **61** | **PASS** (Correctly ignored) |
| **T-INT-02 (Grade 4)** | SPD -0.8 (Extreme) | Non-Dyslexic: **10.8** / Dyslexic: **12.6** | **PASS** (~18% boost applied) |
| **T-UNIT-01** | Score Capping | Score 98 * 1.15 = **100.0** | **PASS** (No overflow) |

### 2. Performance & Latency
- **Cold Start:** ~15-20s (Pre-downloaded model loading).
- **P95 Latency:** **1.8s** per essay scoring request (on 4-core Cloud Run).
- **Throughput:** Supports up to **80 concurrent requests** per instance.

### 3. Rate Limiting Limits
- **Scoring (ML):** 20 requests/minute.
- **Scoring (Baseline):** 30 requests/minute.
- **Research Analysis:** 5 requests/minute.

---

## 🛠️ Commands & Scripts

### Run Automated Test Suite
```bash
# Standard Unit/Integration Tests
set DISABLE_ML=1
pytest -vv tests/

# Verification of Rate Limiting
python test_rate_limit_verification.py
```

### Trigger Fairness Dashboard Analysis
To refresh the fairness metrics in Firestore based on the latest student data:
```bash
python -m analysis.firestore_fairness_eval
```

### Recalculate Calibration Multipliers
If new training data is added to the research dataset:
```bash
python -m analysis.calculate_exact_multipliers
```

---

## 📦 Quick Start (Local Development)

### 1️⃣ Setup Environment
```bash
python -m venv venv
# Windows: venv\Scripts\activate | Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

### 2️⃣ Run the Server
```bash
uvicorn app.main:app --reload --port 8001
```
- **API Docs:** http://127.0.0.1:8001/docs

---

## 🐳 Docker & Cloud Specs

### Build Image
```bash
docker build -t bias-aware-scoring-engine .
```

### Google Cloud Run Configuration
- **Memory:** **12 GiB** (Required for stable XLMR-Large runtime).
- **CPU:** **4 vCPUs** (Required for < 2s P95 latency).
- **Auth:** Headers must include `X-API-KEY`.

---

## ⚖️ Empirical Fairness Metrics (Current Dataset)
| Grade | SPD | DIR | Mitigation Active |
|---|---|---|---|
| 3 | -0.05 | 0.92 | ⚪ No |
| 4 | -0.18 | 0.72 | 🟢 **ACTIVE** (Most Biased) |
| 8 | -0.15 | 0.78 | 🟢 **ACTIVE** |

---

## 📚 Global Research References
- **IBM AI Fairness 360:** Industry standard for bias metrics.
- **XLM-RoBERTa (Conneau et al., 2019):** Advanced cross-lingual representation.
- **EEOC 80% Rule:** Legal standard for disparate impact.

---

## 👥 Research Team
- **Nuwani Fonseka** 

*This engine is a key component of active academic research into equitable AI for education.*