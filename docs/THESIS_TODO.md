# 📋 Thesis Completion Checklist

## Project: Bias-Aware Sinhala Essay Grading System
**Last Updated:** 2026-02-08

---

## ✅ COMPLETED COMPONENTS

### Core Research Implementation

| Objective | Status | Implementation File | Notes |
|-----------|--------|---------------------|-------|
| 1. Semantic-aware scoring pipeline | ✅ Done | `app/sinhala_ml_v2.py` | XLM-RoBERTa Large with multi-head regression |
| 2. Bias detection (SPD, DIR) | ✅ Done | `app/fairness.py` | Statistical Parity Difference & Disparate Impact Ratio |
| 3. Bias mitigation (Post-processing) | ✅ Done | `app/mitigation.py` | Proportional Calibration aligned with AIF360 |
| 4. Grade-aware calibration (3-8) | ✅ Done | `app/mitigation.py` | Grade-specific metrics stored in Firebase |
| 5. Educator fairness dashboard | ✅ Done | Frontend `/internal/fairness` | React-based visualization |
| 6. Transparency logging | ✅ Done | `MitigationRecord` dataclass | Full audit trail |
| 7. RESTful API | ✅ Done | `app/main.py` | FastAPI microservice |
| 8. Cloud deployment | ✅ Done | Google Cloud Run | 8GB memory, europe-west1 |

### Technical Infrastructure

| Component | Status | Details |
|-----------|--------|---------|
| Docker containerization | ✅ Done | `Dockerfile` with lazy model loading |
| CI/CD pipeline | ✅ Done | GitHub Actions → Cloud Run |
| API Gateway integration | ✅ Done | Unified frontend-backend routing |
| Model hosting | ✅ Done | HuggingFace Hub: `akura-official/xlm-roberta-large-sinhala-multihead` |

---

## ⏳ REMAINING TASKS

### 🔴 CRITICAL PRIORITY (Must Complete for Thesis)

#### 1. Accuracy Validation (Pearson Correlation)
**Requirement:** "Maintain ≥90% Pearson correlation with expert-assigned scores"

| Task | Effort | File to Create |
|------|--------|----------------|
| Collect 50-100 expert-graded essays | 2-3 hours | `data/expert_graded_essays.csv` |
| Create validation script | 1 hour | `analysis/accuracy_validation.py` |
| Calculate Pearson correlation | 30 min | - |
| Document results in thesis | 1 hour | - |

**Script Template:**
```python
# analysis/accuracy_validation.py
from scipy.stats import pearsonr
import pandas as pd

# Load expert grades and model predictions
df = pd.read_csv("data/expert_graded_essays.csv")
# Columns: essay_text, expert_score, model_score

r, p_value = pearsonr(df['expert_score'], df['model_score'])
print(f"Pearson r = {r:.3f} (p = {p_value:.4f})")
# Target: r ≥ 0.90
```

#### 2. Latency/Performance Testing
**Requirement:** "Deliver scoring results within 2 seconds for essays ≤1,000 words"

| Task | Effort | File to Create |
|------|--------|----------------|
| Create latency test script | 1 hour | `analysis/latency_test.py` |
| Test with varying essay lengths | 30 min | 100, 500, 1000 words |
| Document P95 latency | 30 min | Should be <2 seconds |

**Script Template:**
```python
# analysis/latency_test.py
import requests
import time
import statistics

API_URL = "https://bias-aware-scoring-engine-651457725719.europe-west1.run.app"

def measure_latency(essay_text, runs=10):
    times = []
    for _ in range(runs):
        start = time.time()
        response = requests.post(f"{API_URL}/score-sinhala-ml", json={
            "text": essay_text,
            "grade": 7,
            "dyslexic_flag": False
        })
        elapsed = time.time() - start
        times.append(elapsed)
    return {
        "mean": statistics.mean(times),
        "p95": sorted(times)[int(len(times) * 0.95)],
        "max": max(times)
    }
```

#### 3. Thesis Documentation
| Document | Purpose | Status |
|----------|---------|--------|
| System Architecture Diagram | Figure for methodology section | ⏳ Pending |
| API Documentation (OpenAPI) | Appendix | ✅ Available at `/docs` |
| Fairness Metrics Report | Results section (SPD, DIR per grade) | ⏳ Pending |
| Accuracy Validation Results | Results section | ⏳ Pending |
| Performance Benchmarks | Results section | ⏳ Pending |

---

### 🟡 IMPORTANT (Should Complete)

#### 4. Load/Scalability Testing
**Requirement:** "Process up to 10,000 essays/day with linear performance scaling"

| Task | Effort | Tool |
|------|--------|------|
| Install locust | 10 min | `pip install locust` |
| Create load test script | 1 hour | `analysis/load_test.py` |
| Run 50-100 concurrent users | 30 min | - |
| Document throughput | 30 min | Essays/second, error rate |

#### 5. Error Handling Tests
**Test Cases:**
- [ ] Malformed JSON input → Should return 422
- [ ] Empty essay text → Should return validation error
- [ ] Invalid grade (e.g., grade=99) → Should handle gracefully
- [ ] Special characters / injection attempts → Should be sanitized

#### 6. Formal Test Case Documentation

| Test ID | Description | Status |
|---------|-------------|--------|
| T-BA-01 | Statistical parity measurement | ✅ Done |
| T-BA-02 | Score adjustment validation | ✅ Done |
| T-BA-03 | Semantic Scoring Accuracy | ⏳ Need validation |
| T-BA-04 | Latency & Performance | ⏳ Need script |
| T-BA-05 | Malformed Input Handling | ⏳ Need tests |
| T-BA-06 | Fairness Report Generation | ✅ Done |
| T-BA-07 | Security & Access Control | ⏳ Optional |

---

### 🟢 OPTIONAL (If Time Permits)

#### 7. EOD (Equal Opportunity Difference)
**Requirement:** "EOD requires teacher-annotated ground truth labels"

| Task | Notes |
|------|-------|
| Collect ground truth pass/fail labels | Requires teacher participation |
| Calculate EOD | Compare true positive rates |
| Document as "future work" if no data | Acceptable for thesis |

#### 8. Security Testing
| Task | Status |
|------|--------|
| JWT Authentication | ⏳ Not implemented (using API Gateway) |
| TLS 1.3 (HTTPS) | ✅ Cloud Run provides this |
| Input validation | ✅ Pydantic schemas |

#### 9. Advanced Mitigation Comparison
**Potential comparison study:**
- Current: Proportional Calibration
- Alternative: Adversarial Debiasing
- Alternative: Reweighing

---

## 📊 QUICK REFERENCE: Key Metrics to Report

### Fairness Metrics (From Firebase)
```
Grade 3: SPD = -0.05, DIR = 0.92
Grade 4: SPD = -0.08, DIR = 0.88
Grade 5: SPD = -0.12, DIR = 0.85
Grade 6: SPD = -0.09, DIR = 0.87
Grade 7: SPD = -0.11, DIR = 0.83
Grade 8: SPD = -0.15, DIR = 0.78
```

### Performance Targets
| Metric | Target | Actual |
|--------|--------|--------|
| Pearson correlation | ≥ 0.90 | ⏳ TBD |
| P95 Latency (cold) | < 60s | ~45s |
| P95 Latency (warm) | < 2s | ~1.2s |
| Daily throughput | 10,000 essays | ⏳ TBD |

---

## 📅 SUGGESTED TIMELINE

| Week | Focus Area | Deliverables |
|------|------------|--------------|
| Week 1 | Accuracy Validation | Script + results table |
| Week 1 | Latency Testing | Script + benchmark chart |
| Week 2 | Load Testing | Locust results |
| Week 2 | Error Handling | Test cases documented |
| Week 3 | Thesis Writing | Methodology + Results sections |
| Week 4 | Final Review | Complete thesis draft |

---

## 🔗 USEFUL COMMANDS

```bash
# Run local server
uvicorn app.main:app --reload --port 8001

# Run tests (CI-safe mode)
set DISABLE_ML=1 && pytest -vv

# View Cloud Run logs
gcloud logging read "resource.type=cloud_run_revision" --project=project-dyslexia-v1 --limit=20

# Check deployed service
curl https://bias-aware-scoring-engine-651457725719.europe-west1.run.app/health
```

---

**Document Created:** 2026-02-08  
**Author:** Research Team
