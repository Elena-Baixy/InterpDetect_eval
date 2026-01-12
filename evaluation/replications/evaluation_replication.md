# InterpDetect Replication Evaluation

## Overview

This document evaluates the replication of the InterpDetect experiment for detecting hallucinations in RAG systems using interpretability signals (ECS and PKS).

## Replication Process Summary

### What was replicated:
1. **Correlation Analysis**: Computed point-biserial correlations between ECS/PKS and hallucination labels
2. **Classifier Training**: Trained four classifiers (Logistic Regression, SVC, Random Forest, XGBoost)
3. **Model Evaluation**: Evaluated on both self-evaluation (Qwen) and proxy evaluation (GPT-4.1-mini) test sets

### Key Results:
- ECS shows negative correlation with hallucination (-0.2908)
- PKS shows positive correlation with hallucination (0.2806)
- All 448 attention heads show negative ECS correlation
- SVC achieved best validation F1 (76.01%, reported: 76.60%)
- Self-eval response F1: 73.42% (reported: 74.68%)
- Proxy-eval response F1: 76.78% (reported: 75.36%)

---

## Replication Evaluation — Binary Checklist

### RP1. Implementation Reconstructability

**PASS**

**Rationale**: The experiment can be reconstructed from the plan and code-walk (README) documentation. The plan clearly describes:
- The hypothesis and methodology
- The feature extraction approach (ECS via attention, PKS via FFN JS-divergence)
- The classifier training pipeline
- The evaluation metrics and expected results

The codebase provides clear implementations in `compute_scores.py`, `classifier.py`, and `predict.py`. While some parameters (like `max_depth=5`, test_size=0.1) had to be inferred from code defaults, these were not critical for understanding the experiment logic.

The pre-computed scores in the `datasets/` directory allowed for exact replication of the classifier training and evaluation phases without needing to re-run the score extraction (which would require significant compute resources).

---

### RP2. Environment Reproducibility

**PASS**

**Rationale**: The environment can be set up and run with standard Python packages:
- `scikit-learn` for classifiers
- `xgboost` for XGBoost classifier
- `pandas`, `numpy` for data manipulation
- `scipy` for statistical tests
- `matplotlib`, `seaborn` for visualization

Minor version warnings appeared when loading pre-trained models (scikit-learn 1.7.1 vs 1.8.0), but these did not affect functionality. All code executed successfully on the available CUDA environment (PyTorch 2.9.1+cu128).

The repository includes `requirements.txt` files for both main and baseline dependencies. The pre-computed datasets eliminate the need for TransformerLens/HuggingFace model loading for the replication.

---

### RP3. Determinism and Stability

**PASS**

**Rationale**: The replicated results are stable and consistent with reported values:

| Metric | Plan Reported | Replicated | Difference |
|--------|---------------|------------|------------|
| ECS-Hallucination Correlation | Negative | -0.2908 | Matches direction |
| PKS-Hallucination Correlation | Positive | +0.2806 | Matches direction |
| Best Val F1 (SVC) | 76.60% | 76.01% | -0.59% |
| Self-eval F1 (SVC) | 74.68% | 73.42% | -1.26% |
| Proxy-eval F1 (SVC) | 75.36% | 76.78% | +1.42% |

The small differences (<2 percentage points) are expected due to:
1. Random seed handling in train/test splits
2. Potential differences in class balancing
3. scikit-learn version differences

The qualitative findings (correlation directions, model ranking, generalization capability) are fully replicated. Running the replication script multiple times with fixed seeds produces identical results.

---

### RP4. Demo Presentation

**NA**

**Rationale**: The repository does not contain a separate demo. Instead, it provides:
1. Pre-computed score datasets for training and testing
2. Pre-trained model pickles
3. Command-line scripts for each pipeline stage

This is a full research implementation rather than a demo-only repository, so RP4 does not apply.

---

## Issues Encountered During Replication

1. **Pre-computed data**: The replication used pre-computed ECS/PKS scores rather than re-running the score extraction pipeline. This is acceptable as the score extraction requires loading the Qwen3-0.6B model and processing all examples, which is computationally intensive.

2. **Class balancing**: The balancing strategy (undersampling) was inferred from code rather than explicitly stated in the plan.

3. **Hyperparameters**: Some hyperparameters (max_depth=5, test_size=0.1) were inferred from code defaults.

4. **Model version warnings**: Pre-trained models were saved with scikit-learn 1.7.1 but loaded with 1.8.0, causing warnings (not errors).

5. **XGBoost warning**: The `use_label_encoder` parameter is deprecated but does not affect results.

---

## Summary

The InterpDetect replication was **successful**. All key findings from the plan were reproduced:

1. **Correlation analysis**: ECS shows negative correlation, PKS shows positive correlation, confirming the hypothesis
2. **Model selection**: SVC identified as best model with nearly identical validation F1
3. **Generalization**: Proxy-based evaluation shows the method works across models

The small numerical differences (1-2%) in F1 scores are within expected variance and do not affect the scientific conclusions. The repository provides sufficient documentation and code to reconstruct and verify the experiment.

### Checklist Summary

| Criterion | Result |
|-----------|--------|
| RP1. Implementation Reconstructability | **PASS** |
| RP2. Environment Reproducibility | **PASS** |
| RP3. Determinism and Stability | **PASS** |
| RP4. Demo Presentation | **NA** |
