# InterpDetect Replication Documentation

## Goal
Replicate the InterpDetect hallucination detection experiment, which uses mechanistic interpretability signals from transformer models to detect hallucinations in RAG (Retrieval-Augmented Generation) systems.

## Data

### Source
- **Dataset**: RAGBench/FinQA dataset
- **Model for Signals**: Qwen3-0.6B (28 layers, 16 attention heads)
- **Pre-computed Scores**: Located in `datasets/train/` and `datasets/test/`

### Features
- **ECS (External Context Score)**: 448 features (28 layers × 16 heads)
  - Computed via attention weights and cosine similarity between response and context embeddings
- **PKS (Parametric Knowledge Score)**: 28 features (one per layer)
  - Computed via Jensen-Shannon divergence between vocabulary distributions before/after FFN layers

### Dataset Statistics
| Dataset | Examples | Spans | Class 0 | Class 1 |
|---------|----------|-------|---------|---------|
| Training | 1,800 | 7,799 | 4,406 | 3,393 |
| Test (Self) | 256 | 975 | 699 | 276 |
| Test (Proxy) | 166 | 1,105 | 835 | 270 |

## Method

### Classifier Training
1. **Preprocessing**:
   - Load pre-computed ECS/PKS scores from JSON files
   - Balance classes by undersampling majority class
   - 90/10 train/validation split with stratification
   - StandardScaler normalization

2. **Models Trained**:
   - Logistic Regression
   - Support Vector Classifier (SVC)
   - Random Forest (max_depth=5)
   - XGBoost (max_depth=5)

3. **Evaluation**:
   - Span-level metrics (precision, recall, F1)
   - Response-level metrics using OR aggregation

### Evaluation Settings
1. **Self-Evaluation**: Responses from Qwen3-0.6B, signals from Qwen3-0.6B
2. **Proxy-Based**: Responses from GPT-4.1-mini, signals from Qwen3-0.6B

## Results

### Classifier Comparison (Validation Set)
| Model | Train F1 | Val F1 |
|-------|----------|--------|
| Logistic Regression | 78.74% | 72.78% |
| SVC | 82.04% | 76.01% |
| Random Forest | 77.84% | 74.78% |
| XGBoost | 99.82% | 74.82% |

### Final Results Comparison
| Metric | Expected | Replicated | Pre-trained |
|--------|----------|------------|-------------|
| SVC Validation F1 | 76.60% | 76.01% | - |
| Self-Eval Response F1 | 74.68% | 73.42% | 74.68% |
| Proxy-Based Response F1 | 75.36% | 76.78% | 75.36% |

## Analysis

### Key Findings
1. **SVC achieves highest validation F1** (76.01%), confirming the original finding
2. **XGBoost overfits significantly** (train 99.8% vs val 74.8%), as noted in the original
3. **Proxy-based evaluation works** - signals from small model transfer to larger model responses
4. **Replicated results within 1.5%** of expected values
5. **Pre-trained models produce exact expected results**

### Reproducibility
- Results are fully deterministic with seed=42
- Zero variance across 3 consecutive runs
- sklearn version warnings (1.7.1 vs 1.5.2) do not affect functionality

### Deviations from Original
- Did not apply optional feature selection (DropConstantFeatures, SmartCorrelatedSelection)
- Minor differences in results due to random state handling in class balancing
