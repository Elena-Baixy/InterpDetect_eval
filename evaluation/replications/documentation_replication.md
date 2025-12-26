# InterpDetect Replication Documentation

## Goal

Replicate the InterpDetect experiment for hallucination detection in Retrieval-Augmented Generation (RAG) systems using mechanistic interpretability signals. The experiment aims to demonstrate that:
1. External Context Score (ECS) and Parametric Knowledge Score (PKS) can predict hallucinations
2. A small proxy model (Qwen3-0.6b) can generalize to detect hallucinations in larger models (GPT-4.1-mini)

## Data

### Training Data
- **Source**: Pre-computed ECS and PKS scores from Qwen3-0.6b model
- **Size**: 1,800 response-level examples → 7,799 span-level samples
- **Features**: 
  - 448 ECS features (28 layers × 16 attention heads)
  - 28 PKS features (one per layer)
  - Total: 476 features
- **Labels**: Binary hallucination labels (0 = truthful, 1 = hallucinated)
- **Class Balance**: 4,406 non-hallucinated vs 3,393 hallucinated spans

### Test Data
1. **Self-evaluation (Qwen3-0.6b)**: 256 examples → 975 spans
2. **Proxy evaluation (GPT-4.1-mini)**: 166 examples → 1,105 spans

## Method

### Feature Engineering
1. **External Context Score (ECS)**: For each (layer, head) pair, compute cosine similarity between the response span embedding and the most-attended context chunk embedding
2. **Parametric Knowledge Score (PKS)**: For each FFN layer, compute Jensen-Shannon divergence between vocabulary distributions before and after the FFN layer

### Model Training
1. Balance classes by undersampling majority class
2. Split data 90/10 for train/validation
3. Standardize features using StandardScaler
4. Train four classifiers:
   - Logistic Regression
   - Support Vector Classifier (SVC)
   - Random Forest (max_depth=5)
   - XGBoost (max_depth=5)

### Evaluation
- **Span-level**: Direct prediction on individual response spans
- **Response-level**: Aggregate span predictions (OR logic - if any span is hallucinated, response is hallucinated)

## Results

### Training/Validation Performance (Span-Level)

| Model | Train F1 | Val F1 |
|-------|----------|--------|
| LR | 78.74% | 72.78% |
| SVC | 82.04% | 76.01% |
| RandomForest | 77.84% | 74.78% |
| XGBoost | 99.82% | 74.82% |

**Observation**: XGBoost shows clear overfitting (99.82% train vs 74.82% val)

### Self-Evaluation Results (Qwen3-0.6b, Response-Level)

| Model | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| LR | 61.05% | 90.62% | 72.96% |
| SVC | 61.70% | 90.62% | 73.42% |
| RandomForest | 64.29% | 91.41% | 75.48% |
| XGBoost | 57.79% | 89.84% | 70.34% |

### Proxy Evaluation Results (GPT-4.1-mini, Response-Level)

| Model | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| LR | 61.36% | 97.59% | 75.35% |
| SVC | 63.28% | 97.59% | 76.78% |
| RandomForest | 60.00% | 97.59% | 74.31% |
| XGBoost | 56.94% | 98.80% | 72.25% |

## Analysis

### Key Findings

1. **SVC achieves best validation F1**: Consistent with the plan's report that SVC achieved highest validation F1 (76.60% reported vs 76.01% replicated)

2. **XGBoost overfitting confirmed**: The plan noted XGBoost overfitted despite strong training performance - our results show 99.82% train F1 vs 74.82% val F1

3. **Proxy generalization validated**: The trained classifier on Qwen3-0.6b successfully generalizes to GPT-4.1-mini responses:
   - Self-evaluation F1: ~73-75%
   - Proxy evaluation F1: ~73-77%
   
4. **Numerical consistency**: Replicated F1 scores are within 2-3% of reported values, which is expected given:
   - Different random seeds for train/val splits
   - Class balancing randomness
   - Potential slight variations in data preprocessing

### Comparison with Reported Results

| Setting | Reported F1 | Replicated F1 | Difference |
|---------|-------------|---------------|------------|
| Self-eval (best) | 74.68% | 75.48% (RF) | +0.80% |
| Proxy-eval (best) | 75.36% | 76.78% (SVC) | +1.42% |

The replicated results are numerically consistent with the original findings.
