# Bias Mitigation & Integration Report

## 1. Findings: "Positive Bias" Anomaly
During the initial audit of the scoring model, we observed a counter-intuitive phenomenon where essays flagged as "dyslexic" (or containing features common in dyslexic writing) were occasionally receiving **higher** scores than control essays of similar quality.

### Hypothesis
This "Positive Bias" suggests the model might be interpreting unique sentence structures or creative phonetic spellings as "complexity" or "vocabulary richness" rather than errors. Alternatively, the model may differ in how it attends to "noisy" text, bypassing standard grammar penalties that apply to well-structured but simple essays.

## 2. Fairness Metrics (Baseline)
Before mitigation, the following fairness metrics were recorded on the verification set:

- **Statistical Parity Difference (SPD):** [INSERT SCORE, e.g., -0.15]
  - *Interpretation:* A negative value indicates the unprivileged group (dyslexic) is receiving positive outcomes at a lower rate. A positive value indicates the opposite.
- **Disparate Impact Ratio (DIR):** [INSERT SCORE]
- **Equalized Odds Difference (EOD):** [INSERT SCORE]

*(Note: These values will be updated as the full dataset is processed.)*

## 3. Methodology: Mixed Data Retraining
To address potential biases, we adopted a "Data Augmentation" strategy rather than purely post-processing.

### Steps:
1. **Synthetic Data Generation:**
   - Used `nlpaug` and phonetically grounded substitution rules (Sinhala-specific) to simulate dyslexic errors (mirror writing, phonetic variation) in a subset of the original training data.
2. **Mixed Training Set:**
   - Combined Original Essays + Synthetic Dyslexic Essays.
   - Ratio: roughly 1:1 or as per dataset balance.
3. **Retraining:**
   - Fine-tuned the `xlm-roberta-large-sinhala-multihead` model on this combined dataset.
   - Objective: Teach the model to be robust to surface-level noise and validation of semantic content over orthographic perfection.

## 4. Mitigation Logic (Implemented Strategy)
The system currently implements a **Hybrid Pre-Processing + Conditional Post-Processing** strategy, specifically tailored for the high-stakes educational domain.

- **Pre-Processing (Linguistic):** Instead of mathematical transformations, the system uses a **Sentence Reconstruction** layer (Member 2's component) to fix orthographic and grammatical dyslexic errors before scoring.
- **Post-Processing (Calibrated):** The `ConditionalFairnessMitigator` in `app/mitigation.py` applies a grade-aware adjustment ONLY when unfavorable bias is statistically detected (|SPD| > 0.1 or DIR < 0.8).
- **Log:** All adjustments are logged via `MitigationRecord` objects to ensure full auditability for researchers and educators.

## 5. Research Rationale: Comparative Analysis of AIF360 Strategies
In the development of the Bias-Aware Scoring Engine, several common AIF360 mitigation strategies were evaluated. Below is the justification for why certain methods were rejected in favor of our hybrid approach.

| Method | Status | Research Justification for Rejection |
| :--- | :--- | :--- |
| **Statistical Pre-processing (DIR)** | Rejected | Black-box mathematical transformations destroy the linguistic signal. In an educational context, preserving the original errors is necessary for pedagogical feedback. |
| **In-Processing (Reweighing)** | Future Work | Requires instance weights during retraining. Rejected for the initial phase due to high computational overhead and the risk of overfitting to synthetic data patterns. |
| **Adversarial Debiasing** | Concept Only | Requires architectural changes to the Transformer-based model (GAN approach). Rejected due to complexity for the Sinhala language and limited real-world dyslexic data. |
| **Conditional Calibration (Implementation)** | **Chosen** | **Merit-Preserving:** Allows the model to grade naturally. Only intervenes when systemic disparity is measured. Minimizes "False Positive" fairness boosts. |

### Why This Hybrid Strategy?
1. **Explainability:** We can show the "Before/After" of both the sentence reconstruction and the score calibration.
2. **Pedagogical Validity:** Unlike adversarial debiasing, our method does not "blind" the model to errors; it simply corrects the *penalty* associated with those errors.
3. **Modularity:** The scoring engine remains stable (post-processing) while linguistic improvements can be made separately in the reconstruction module.

## 6. Next Steps
- Clear the legacy "contaminated" data in Firebase (affected by old naive boosting).
- Re-run the batch fairness evaluation (`firestore_fairness_eval.py`) to establish a clean baseline.
- Monitor the Disparate Impact Ratio (DIR) in the Dashboard during live testing.
- Document the impact of "Unfavorable-Only" mitigation on overall group parity.
