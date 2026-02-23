# 🛡️ Thesis Fairness & Mitigation Proof Report

**Date:** 2026-02-22 22:28:32
**Dataset Size:** 53 scored essays with mitigation logs

## 1. Fairness Metrics: Baseline vs. Mitigated
| Grade | Metric | Baseline (Biased) | Mitigated (Fair) | Improvement |
|-------|--------|-------------------|------------------|-------------|
| 3 | SPD | 0.000 | 0.000 | 0.0% |
| 3 | DIR | 0.000 | 0.000 | 0.0% |
| 4 | SPD | 0.000 | 0.000 | 0.0% |
| 4 | DIR | 0.000 | 0.000 | 0.0% |
| 5 | SPD | -0.500 | -0.667 | 0.0% |
| 5 | DIR | 0.500 | 0.667 | 33.3% |
| 6 | SPD | -0.789 | -0.789 | 0.0% |
| 6 | DIR | 0.789 | 0.789 | 0.0% |
| 7 | SPD | -0.400 | -0.400 | 0.0% |
| 7 | DIR | 0.400 | 0.400 | 0.0% |
| 8 | SPD | -0.083 | -0.083 | 0.0% |
| 8 | DIR | 0.083 | 0.083 | 0.0% |

## 2. Mitigation Impact Statement (Dyslexic Students Only)
| Grade | Avg Baseline Score | Avg Mitigated Score | Mean Absolute Boost |
|-------|--------------------|---------------------|---------------------|
| 3 | 5.18 | 5.52 | +0.34 marks |
| 4 | 4.81 | 5.50 | +0.69 marks |
| 5 | 6.49 | 6.76 | +0.27 marks |
| 6 | 9.49 | 9.68 | +0.19 marks |
| 7 | 5.35 | 5.61 | +0.26 marks |
| 8 | 4.11 | 4.40 | +0.29 marks |

## 3. Statistical Proof of Fairness
Using independent samples t-test to compare Dyslexic vs. Non-Dyslexic distributions.

| Grade | Baseline P-Value | Mitigated P-Value | Conclusion |
|-------|------------------|-------------------|------------|
