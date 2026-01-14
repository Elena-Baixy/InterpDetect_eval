# InterpDetect Replication Documentation

## Goal

Replicate the InterpDetect experiment for detecting hallucinations in Retrieval-Augmented Generation (RAG) systems using mechanistic interpretability signals (ECS and PKS).

## Data

### Source
- **Repository**: InterpDetect (https://openreview.net/pdf?id=TZzBKwHLwF)
- **Model**: Qwen3-0.6B (smallest available model as per replication guidelines)

### Training Data
- **Location**: `/datasets/train/train3000_w_chunk_score_part*.json` (18 files)
- **Samples**: 1800 responses → 7799 spans
- **After balancing**: 6786 spans (3393 hallucinated, 3393 non-hallucinated)
- **Split**: 90% train, 10% validation

### Test Data
- **Location**: `/datasets/test/test_w_chunk_score_qwen06b.json`
- **Samples**: 256 responses → 975 spans
- **Distribution**: 699 non-hallucinated, 276 hallucinated

### Features
- **ECS (External Context Score)**: 448 features (28 layers × 16 attention heads)
  - Measures cosine similarity between response span and most-attended context chunk
- **PKS (Parametric Knowledge Score)**: 28 features (28 FFN layers)
  - Measures Jensen-Shannon divergence in vocabulary distribution before/after FFN

## Method

### 1. Score Computation (Pre-computed)
The repository provides pre-computed ECS and PKS scores using TransformerLens on Qwen3-0.6B:

**ECS Computation**:
1. For each response span, compute attention weights to all context chunks
2. Identify the context chunk with maximum attention
3. Compute cosine similarity using BGE embeddings

**PKS Computation**:
1. Get residual stream before (hook_resid_mid) and after (hook_resid_post) each FFN layer
2. Project both to vocabulary space using unembedding matrix W_U
3. Compute Jensen-Shannon divergence between softmax distributions

### 2. Classifier Training
Trained 4 classifiers using scikit-learn:
- **Logistic Regression**: Baseline linear classifier
- **SVC**: Support Vector Classifier (selected as best)
- **Random Forest**: Ensemble tree method (max_depth=5)
- **XGBoost**: Gradient boosting (max_depth=5)

**Preprocessing**: StandardScaler normalization
**Class Balancing**: Undersample majority class

### 3. Evaluation
- **Span-level**: Direct prediction on individual spans
- **Response-level**: OR aggregation (hallucinated if any span is hallucinated)

## Results

### Correlation Analysis
| Signal | Correlation (r) | p-value | Interpretation |
|--------|-----------------|---------|----------------|
| ECS    | -0.2987         | 6.87e-140 | Hallucinated responses use less external context |
| PKS    | +0.2768         | 1.32e-119 | Hallucinated responses have more parametric knowledge injection |

**Per-Layer PKS Correlation** (Top 5):
- Layer 24: r = 0.3179
- Layer 23: r = 0.3177
- Layer 21: r = 0.3147
- Layer 25: r = 0.2941
- Layer 20: r = 0.2579

### Classifier Performance (Validation Set)

| Algorithm    | Train F1 | Val F1  |
|--------------|----------|---------|
| LR           | 78.74%   | 72.40%  |
| **SVC**      | 82.04%   | **76.01%** |
| RandomForest | 77.84%   | 74.78%  |
| XGBoost      | 99.82%   | 74.82%  |

### Test Set Evaluation

| Metric           | Paper Reported | Pre-trained | Replicated |
|------------------|----------------|-------------|------------|
| Response-level F1| 74.68%         | 74.68%      | 73.42%     |
| Span-level F1    | N/A            | 64.94%      | 63.75%     |

## Analysis

### Key Findings Verified
1. **ECS-Hallucination Correlation**: Negative correlation confirmed (-0.2987), supporting the hypothesis that hallucinated responses utilize less external context.

2. **PKS-Hallucination Correlation**: Positive correlation confirmed (+0.2768), supporting the hypothesis that hallucinated responses involve more parametric knowledge injection.

3. **Later Layer Effect**: Layers 20-25 show the strongest PKS correlation with hallucination, consistent with the paper's claim that later FFN layers disproportionately inject parametric knowledge.

4. **SVC Selection**: SVC achieves best validation F1 (76.01%), closely matching the paper's reported 76.60%.

5. **XGBoost Overfitting**: XGBoost shows severe overfitting (Train: 99.82%, Val: 74.82%), as noted in the paper.

### Replication Discrepancy
- Response-level F1 is 1.26% lower than the pre-trained model
- This is likely due to:
  - Random variation in train/validation split
  - Slightly different hyperparameters
  - Class balancing randomization

## Conclusions

The replication successfully verifies:
1. The core mechanistic hypothesis (ECS/PKS correlation with hallucination)
2. The classifier training methodology
3. The evaluation approach (span → response aggregation)
4. The relative performance of different classifiers

The numerical results are within acceptable tolerance (< 2% difference) of reported values.
