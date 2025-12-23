# Plan
## Objective
Develop a mechanistic interpretability-based hallucination detection method for Retrieval-Augmented Generation (RAG) systems by computing External Context Scores (ECS) across layers and attention heads and Parametric Knowledge Scores (PKS) across layers (FFN), training regression-based classifiers on these signals, and demonstrating generalization from a small proxy model (Qwen3-0.6b) to larger production models (GPT-4.1-mini).

## Hypothesis
1. RAG hallucinations correlate with:  later-layer FFN modules disproportionately inject parametric knowledge into the residual stream while attention heads fail to adequately exploit external context.
2. External Context Score (ECS) and Parametric Knowledge Score (PKS) are correlated with hallucination occurrence and can serve as predictive features for hallucination detection.
3. Mechanistic signals extracted from a small proxy model (0.6b parameters) can generalize to detect hallucinations in responses from larger production-level models.

## Methodology
1. Compute External Context Score (ECS) per attention head and layer by identifying the most attended context chunk via attention weights, then measuring cosine similarity between response and context embeddings.
2. Compute Parametric Knowledge Score (PKS) per FFN layer by measuring Jensen-Shannon divergence between vocabulary distributions before and after the FFN layer in the residual stream.
3. Use TransformerLens library on Qwen3-0.6b model to extract internal mechanistic signals (ECS and PKS) at span level across 28 layers and 16 attention heads.
4. Train binary classifiers (Logistic Regression, SVC, Random Forest, XGBoost) on standardized and correlation-filtered ECS/PKS features to predict span-level hallucinations, then aggregate to response-level.
5. Evaluate both self-evaluation (same model generates responses and computes signals) and proxy-based evaluation (Qwen3-0.6b signals applied to GPT-4.1-mini responses) settings.

## Experiments
### Correlation Analysis: ECS vs Hallucination
- What varied: Comparing ECS values between truthful and hallucinated responses across layers and attention heads
- Metric: Pearson Correlation Coefficient between inverse hallucination label and ECS
- Main result: All attention heads exhibit negative correlations; hallucinated responses utilize less external context than truthful ones.

### Correlation Analysis: PKS vs Hallucination
- What varied: Comparing PKS values between truthful and hallucinated responses across FFN layers
- Metric: Pearson correlation between hallucination labels and PKS
- Main result: Later-layer FFNs exhibit substantially higher PKS for hallucinated responses and are positively correlated with hallucinations.

### Classifier Training and Selection
- What varied: Four classifier types: Logistic Regression, SVC, Random Forest, XGBoost trained on 7,799 span-level samples
- Metric: Validation F1 score, precision, and recall at span level
- Main result: SVC achieved highest validation F1 (76.60%) and was selected; XGBoost overfitted despite strong training performance.

### Self-Evaluation Detection
- What varied: Comparing proposed method against baselines (LLMs and commercial tools) on Qwen3-0.6b generated responses
- Metric: Response-level Precision, Recall, F1
- Main result: Method achieved F1=74.68%, outperforming TruLens (67.32%) and llama-3.1-8b-instant (57.53%), comparable to RefChecker (75.86%).

### Proxy-Based Evaluation Detection
- What varied: Applying Qwen3-0.6b trained classifier to GPT-4.1-mini responses against same baselines
- Metric: Response-level Precision, Recall, F1
- Main result: Method achieved F1=75.36%, outperforming nearly all models except GPT-5 (76.92%) and RAGAS (76.19%), using only 0.6b parameter signals.