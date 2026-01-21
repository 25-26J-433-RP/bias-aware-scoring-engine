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

## 4. Mitigation Logic (Integration Phase)
We have prepared the integration pipeline to include a "Safety Valve" post-processing step using IBM AIF360 concepts.

- **Hook Implemented:** `apply_fairness_mitigation(score, flag)` in `sinhala_ml_v2.py`.
- **Trigger:** The system now accepts a `dyslexic_flag` from the upstream Classifier component.
- **Logic:** Currently in "Shadow Mode" (logging only). Future logic will apply re-weighing calibration if the fairness violation exceeds a threshold (e.g., SPD > 0.1).

## 5. Next Steps
- Validate the "Mock" contract with the Pattern Classifier team.
- Run full evaluation on the "Gold Standard" annotated dyslexic essay set once available.
- Finalize the re-weighing parameter in `apply_fairness_mitigation`.
