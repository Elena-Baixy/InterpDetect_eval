# InterpDetect Replication Documentation

## Goal

This replication study aims to reproduce the key experiments from the InterpDetect paper, which develops a mechanistic interpretability-based hallucination detection method for Retrieval-Augmented Generation (RAG) systems. The method computes:

1. **External Context Scores (ECS)**: Measures how much attention heads focus on external context chunks
2. **Parametric Knowledge Scores (PKS)**: Measures JS divergence of vocabulary distributions before/after FFN layers

The hypothesis is that hallucinated responses show lower ECS (less external context utilization) and higher PKS (more parametric knowledge injection, especially in later layers).

## Data

### Training Data
- **Source**: Pre-computed ECS and PKS scores from the repository (`datasets/train/`)
- **Format**: 18 JSON files containing 1,800 examples with scores
- **Span-level samples**: 7,799 total (matching the plan's stated value)
- **Hallucination rate**: 43.51%

### Test Data
Two test sets were used:
1. **Qwen Self-evaluation** (`test_w_chunk_score_qwen06b.json`): 256 examples
   - For evaluating the method when the same model (Qwen3-0.6B) generates responses and computes signals
2. **GPT-4.1-mini Proxy evaluation** (`test_w_chunk_score_gpt41mini.json`): 166 examples
   - For evaluating generalization from Qwen3-0.6B signals to GPT-4.1-mini responses

### Features
- **ECS features**: 448 attention head scores (28 layers x 16 heads)
- **PKS features**: 28 layer scores
- **Total features**: 476

## Method

### 1. Correlation Analysis
Point-biserial correlation was computed between:
- ECS (sum of attention scores) vs hallucination labels
- PKS (sum of layer scores) vs hallucination labels
- Per-layer PKS vs hallucination labels
- Per-head ECS vs hallucination labels

### 2. Classifier Training
Four classifiers were trained on balanced data (undersampled majority class):
- Logistic Regression (max_iter=1000)
- Support Vector Classifier (SVC)
- Random Forest (max_depth=5)
- XGBoost (max_depth=5)

All models used StandardScaler preprocessing. Training/validation split was 90/10 with stratification.

### 3. Evaluation
Models were evaluated at two levels:
- **Span-level**: Direct prediction on individual response spans
- **Response-level**: Aggregated by OR (if any span is predicted hallucinated, the response is marked hallucinated)

## Results

### Correlation Analysis

| Metric | Result |
|--------|--------|
| ECS vs Hallucination | -0.2908 (p < 1e-150) |
| PKS vs Hallucination | 0.2806 (p < 1e-140) |
| Heads with negative ECS correlation | 448/448 (100%) |

**Key finding**: Later-layer FFNs (layers 18-26) show substantially stronger positive PKS correlation with hallucination, confirming that parametric knowledge injection increases in later layers for hallucinated responses.

### Classifier Training Results

| Model | Train F1 | Val F1 |
|-------|----------|--------|
| Logistic Regression | 78.78% | 72.78% |
| **SVC** | **82.04%** | **76.01%** |
| Random Forest | 77.84% | 74.78% |
| XGBoost | 99.82% | 74.82% |

**Best model**: SVC with validation F1 of 76.01%

Note: XGBoost shows clear overfitting (99.82% train vs 74.82% validation)

### Test Set Evaluation (Response-level)

#### Self-evaluation (Qwen responses)

| Model | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Logistic Regression | 61.05% | 90.63% | 72.96% |
| **SVC** | **61.70%** | **90.63%** | **73.42%** |
| Random Forest | 64.29% | 91.41% | 75.48% |
| XGBoost | 57.79% | 89.84% | 70.34% |

#### Proxy evaluation (GPT-4.1-mini responses)

| Model | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Logistic Regression | 61.36% | 97.59% | 75.35% |
| **SVC** | **63.28%** | **97.59%** | **76.78%** |
| Random Forest | 60.00% | 97.59% | 74.31% |
| XGBoost | 56.94% | 98.80% | 72.25% |

### Comparison with Plan

| Metric | Plan Reported | Replicated |
|--------|---------------|------------|
| ECS-Hallucination Correlation | Negative | Negative (-0.2908) |
| PKS-Hallucination Correlation | Positive (later layers) | Positive (0.2806) |
| Best Val F1 Model | SVC (76.60%) | SVC (76.01%) |
| Self-eval Response F1 (SVC) | 74.68% | 73.42% |
| Proxy-eval Response F1 (SVC) | 75.36% | 76.78% |

## Analysis

### Successfully Replicated Findings

1. **Correlation directions match**: ECS shows negative correlation (-0.2908), PKS shows positive correlation (0.2806), confirming the hypothesis that hallucinated responses use less external context and more parametric knowledge.

2. **All attention heads show negative ECS correlation**: 448/448 heads (100%) exhibit negative correlation with hallucination, matching the plan's claim.

3. **Later-layer PKS correlation pattern**: Layers 18-26 show the strongest positive correlations (0.22-0.32), confirming that later-layer FFNs inject more parametric knowledge for hallucinations.

4. **SVC selected as best model**: Both plan and replication identify SVC as having the highest validation F1, with very similar values (76.60% vs 76.01%).

5. **Test set performance**: Response-level F1 scores are within 1-2 percentage points of reported values:
   - Self-eval: 74.68% reported vs 73.42% replicated
   - Proxy-eval: 75.36% reported vs 76.78% replicated

### Minor Discrepancies

1. **Exact F1 values differ slightly**: This is expected due to random seed handling and potential differences in data splitting.

2. **XGBoost overfitting**: Both original and replication show XGBoost overfitting, but the exact degree may differ.

## Conclusion

The replication successfully reproduces all key findings from the InterpDetect paper:

- The correlation analysis confirms the hypothesized relationship between ECS/PKS and hallucinations
- The classifier training reproduces the model comparison with SVC as the best performer
- The test set evaluations show similar performance to reported values

The small numerical differences (1-2 percentage points) are within expected variance and do not affect the validity of the conclusions.
