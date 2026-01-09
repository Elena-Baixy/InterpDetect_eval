# Documentation Evaluation Summary

## InterpDetect Replication - Documentation Evaluation

**Evaluation Date**: 2026-01-08

---

## Results Comparison

The replicated documentation faithfully reproduces the core experimental results from the original InterpDetect project. Key metrics comparison:

| Metric | Original | Replicated | Difference |
|--------|----------|------------|------------|
| SVC Validation F1 | 76.60% | 76.0% | -0.60% |
| Self-Evaluation F1 (Qwen) | 74.68% | 73.42% | -1.26% |
| Proxy-Based F1 (GPT) | 75.36% | 76.78% | +1.42% |

The small differences (within ~1.5%) are attributable to different random splits during replication training. Critically, when using the repository's **pre-trained SVC model**, the replication achieves exact matches: 74.68% for self-evaluation and 75.36% for proxy-based evaluation.

All qualitative trends match: ECS shows negative correlation with hallucination across all 448 attention heads, and PKS shows positive correlation in later layers.

---

## Conclusions Comparison

The replicated documentation presents conclusions fully consistent with the original:

1. **ECS Correlation**: Both confirm that hallucinated responses utilize less external context (negative ECS correlation)
2. **PKS Correlation**: Both confirm later-layer FFNs inject more parametric knowledge for hallucinations (positive PKS correlation)
3. **Model Selection**: Both identify SVC as the best classifier and note XGBoost overfitting
4. **Generalization**: Both confirm that small proxy model (Qwen3-0.6B) signals generalize to larger models (GPT-4.1-mini)
5. **Effectiveness**: Both confirm the method outperforms or matches several baseline approaches

The replicated documentation adds additional detail on reproducibility and determinism, which supports rather than contradicts the original claims.

---

## External or Hallucinated Information

No external references, invented findings, or hallucinated details were introduced in the replicated documentation. All claims are traceable to:

- **plan.md**: Methodology, hypotheses, and expected results
- **CodeWalkthrough.md**: Project structure, usage, and implementation details
- **Source code** (e.g., scripts/compute_scores.py): Implementation specifics like BGE-base-en-v1.5 for embeddings

Technical details discovered during replication (e.g., specific correlation values, sklearn version warnings) represent legitimate replication outputs rather than external information.

---

## Evaluation Checklist

| Criterion | Status | Notes |
|-----------|--------|-------|
| **DE1: Result Fidelity** | PASS | Results within ~1.5% tolerance; pre-trained model achieves exact match |
| **DE2: Conclusion Consistency** | PASS | All conclusions consistent; no contradictions |
| **DE3: No External Information** | PASS | All claims traceable to original docs or code |

---

## Final Verdict

**PASS**

The replicated documentation faithfully reproduces both the results and conclusions of the original InterpDetect experiment. All three evaluation criteria (DE1-DE3) are satisfied.
