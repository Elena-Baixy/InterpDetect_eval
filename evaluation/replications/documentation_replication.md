# InterpDetect Replication Documentation

## Goal

Replicate the InterpDetect experiment that develops a mechanistic interpretability-based hallucination detection method for Retrieval-Augmented Generation (RAG) systems. The method computes External Context Scores (ECS) and Parametric Knowledge Scores (PKS) from a small proxy model (Qwen3-0.6B) to detect hallucinations in both self-generated and larger model (GPT-4.1-mini) responses.

## Data

### Training Data
- **Source**: Pre-computed ECS and PKS scores from 1,800 examples
- **Features**: 448 ECS features (28 layers × 16 attention heads) + 28 PKS features (one per FFN layer)
- **Samples**: 7,799 span-level samples after processing all response chunks
- **Balanced**: Undersampled to 3,393 samples per class (6,786 total)

### Test Data
1. **Qwen Self-Evaluation**: 256 responses, 975 span-level samples
2. **GPT Proxy-Based Evaluation**: 166 responses, 1,105 span-level samples

### Data Format
Each example contains:
- `prompt`: Input question with retrieved context
- `response`: Model-generated answer
- `scores`: List of chunk-level scores containing:
  - `prompt_attention_score`: Dictionary of ECS values per (layer, head)
  - `parameter_knowledge_scores`: Dictionary of PKS values per layer
  - `hallucination_label`: Binary label (0=truthful, 1=hallucinated)

## Method

### 1. External Context Score (ECS)
- For each response chunk and attention head, identify the most attended context chunk via attention weights
- Compute cosine similarity between response chunk and context chunk embeddings using BGE-base-en-v1.5
- Hypothesis: Lower ECS indicates less reliance on external context (correlates with hallucination)

### 2. Parametric Knowledge Score (PKS)
- Compute Jensen-Shannon divergence between vocabulary distributions before and after each FFN layer
- Uses the residual stream projections through the unembedding matrix
- Hypothesis: Higher PKS in later layers indicates more parametric knowledge injection (correlates with hallucination)

### 3. Classifier Training
- **Preprocessing**: StandardScaler normalization
- **Models**: Logistic Regression, SVC, Random Forest, XGBoost
- **Split**: 90% train, 10% validation with stratification
- **Class Balancing**: Undersampling majority class

### 4. Evaluation
- **Span-Level**: Direct prediction on response chunks
- **Response-Level**: OR aggregation (if any span is hallucinated, response is hallucinated)

## Results

### Correlation Analysis

| Feature Type | Correlation Direction | Mean Correlation |
|-------------|----------------------|------------------|
| ECS (all heads) | Negative | -0.23 |
| PKS (early layers 0-9) | Positive | 0.05 |
| PKS (late layers 20-27) | Positive | 0.24 |

**Key Finding**: All 448 attention head features show negative correlation with hallucination, confirming the hypothesis that hallucinated responses utilize less external context.

### Classifier Comparison

| Model | Train F1 | Validation F1 |
|-------|----------|---------------|
| Logistic Regression | 78.7% | 72.8% |
| SVC | 82.0% | 76.0% |
| Random Forest | 77.8% | 74.8% |
| XGBoost | 99.8% | 74.8% |

**Best Model**: SVC with highest validation F1 (76.0%)
**Note**: XGBoost shows significant overfitting (99.8% train vs 74.8% val)

### Detection Performance

| Experiment | Plan F1 | Replicated F1 | Difference |
|------------|---------|---------------|------------|
| Self-Evaluation (Qwen) | 74.68% | 73.42% | -1.26% |
| Proxy-Based (GPT) | 75.36% | 76.78% | +1.42% |

### Pre-trained Model Verification

Using the repository's pre-trained SVC model:
- Self-Evaluation: 74.68% (exact match with plan)
- Proxy-Based: 75.36% (exact match with plan)

## Analysis

### Strengths
1. **Reproducible**: Results are deterministic with fixed random seeds (zero variance across runs)
2. **Numerically Consistent**: Replicated results within ~1.5% of reported values
3. **Hypothesis Validated**: Correlation analysis confirms ECS/PKS relationships with hallucination
4. **Generalization Confirmed**: Proxy-based evaluation shows small model signals generalize to larger models

### Limitations
1. Minor sklearn version mismatch warnings when loading pre-trained models (1.7.1 vs 1.5.2)
2. Small differences in replicated vs reported F1 scores likely due to:
   - Different random splits during replication
   - Using reimplemented pipeline vs original code

### Ambiguities Encountered
1. The exact preprocessing for feature selection (DropConstantFeatures, SmartCorrelatedSelection) was not applied in the replication as it appeared optional in the original code
2. Training data loaded from 18 batch files (1,800 examples) vs the plan mentioning 7,799 span-level samples - this is consistent as each example has multiple response chunks

## Conclusion

The replication successfully reproduces the InterpDetect experiment's key findings:
1. ECS and PKS signals correlate with hallucination as hypothesized
2. SVC classifier achieves best performance without overfitting
3. Self-evaluation and proxy-based detection both work effectively
4. Results are stable and reproducible with proper seed control
