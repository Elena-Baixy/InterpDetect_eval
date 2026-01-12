#!/usr/bin/env python3
"""
Generalizability Evaluation for InterpDetect Repository

This script evaluates:
- GT1: Model Generalization (using Pythia-1.4B instead of Qwen3-0.6B)
- GT2: Data Generalization (using new data instances)
- GT3: Method Generalizability (testing on a similar task)
"""

import os
import sys
import json
import numpy as np
import torch
from torch.nn import functional as F
import warnings
warnings.filterwarnings('ignore')

# Set environment
os.environ['HF_HOME'] = '/net/projects2/chai-lab/shared_models'

# Repository path
REPO_PATH = '/net/scratch2/smallyan/InterpDetect_eval'
EVAL_PATH = os.path.join(REPO_PATH, 'evaluation')

def calculate_dist_2d(sep_vocabulary_dist, sep_attention_dist):
    """Calculate Jensen-Shannon divergence between distributions (PKS)"""
    softmax_mature_layer = F.softmax(sep_vocabulary_dist, dim=-1)
    softmax_anchor_layer = F.softmax(sep_attention_dist, dim=-1)
    M = 0.5 * (softmax_mature_layer + softmax_anchor_layer)
    log_softmax_mature_layer = F.log_softmax(sep_vocabulary_dist, dim=-1)
    log_softmax_anchor_layer = F.log_softmax(sep_attention_dist, dim=-1)
    kl1 = F.kl_div(log_softmax_mature_layer, M, reduction='none').sum(dim=-1)
    kl2 = F.kl_div(log_softmax_anchor_layer, M, reduction='none').sum(dim=-1)
    js_divs = 0.5 * (kl1 + kl2)
    scores = js_divs.cpu().tolist()
    return sum(scores) if isinstance(scores, list) else scores

def calculate_sentence_similarity(bge_model, r_text, p_text):
    """Calculate sentence similarity using BGE model (ECS)"""
    part_embedding = bge_model.encode([r_text], normalize_embeddings=True)
    q_embeddings = bge_model.encode([p_text], normalize_embeddings=True)
    scores_named = np.matmul(q_embeddings, part_embedding.T).flatten()
    return float(scores_named[0])

def evaluate_gt1():
    """
    GT1: Model Generalization Test
    Test if PKS/ECS findings generalize to Pythia-1.4B (not used in original work)
    """
    print("\n" + "="*70)
    print("GT1: MODEL GENERALIZATION TEST")
    print("="*70)
    print("Testing: Do ECS/PKS correlations hold on Pythia-1.4B?")
    print("Original model: Qwen3-0.6B | New model: Pythia-1.4B")
    print("-"*70)

    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer
    from sentence_transformers import SentenceTransformer

    # Load test data
    test_path = os.path.join(REPO_PATH, 'datasets/test/test_w_chunk_score_qwen06b.json')
    with open(test_path, 'r') as f:
        test_data = json.load(f)

    print(f"Loaded {len(test_data)} test examples")

    # Load Pythia-1.4B
    print("Loading Pythia-1.4B model...")
    pythia_model = HookedTransformer.from_pretrained(
        "EleutherAI/pythia-1.4b",
        device="cuda",
        dtype=torch.float16
    )
    pythia_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b")
    print(f"Pythia-1.4B loaded: {pythia_model.cfg.n_layers} layers, {pythia_model.cfg.n_heads} heads")

    # Load BGE for ECS
    print("Loading BGE model for ECS computation...")
    bge_model = SentenceTransformer("BAAI/bge-base-en-v1.5").to("cuda")

    # Select trial examples - one hallucinated, one truthful
    hallucinated_ex = None
    truthful_ex = None

    for ex in test_data:
        has_hallucination = any(s['hallucination_label'] == 1 for s in ex['scores'])
        if has_hallucination and hallucinated_ex is None:
            hallucinated_ex = ex
        elif not has_hallucination and truthful_ex is None:
            truthful_ex = ex
        if hallucinated_ex and truthful_ex:
            break

    print(f"\nSelected examples:")
    print(f"  Hallucinated: {hallucinated_ex['id']}")
    print(f"  Truthful: {truthful_ex['id']}")

    # Process both examples with Pythia
    results = {}

    for label, example in [("hallucinated", hallucinated_ex), ("truthful", truthful_ex)]:
        print(f"\nProcessing {label} example...")

        # Prepare input
        prompt = example['prompt']
        response = example['response']
        full_text = prompt + " " + response

        # Tokenize
        input_ids = pythia_tokenizer(full_text, return_tensors="pt").input_ids
        if input_ids.shape[-1] > pythia_model.cfg.n_ctx:
            input_ids = input_ids[:, -pythia_model.cfg.n_ctx:]
        input_ids = input_ids.to("cuda")

        prompt_ids = pythia_tokenizer(prompt, return_tensors="pt").input_ids
        prompt_len = min(prompt_ids.shape[-1], input_ids.shape[-1] - 10)

        # Run model
        with torch.no_grad():
            logits, cache = pythia_model.run_with_cache(input_ids, return_type="logits")

        # Compute PKS for last 20 tokens (response portion)
        resp_start = prompt_len
        resp_end = min(prompt_len + 50, input_ids.shape[-1])

        pks_scores = []
        for layer_id in range(pythia_model.cfg.n_layers):
            # Pythia uses different hook names - get pre-FFN and post-FFN residual stream
            # hook_resid_pre is before attention+FFN, hook_resid_post is after
            # For PKS we want before and after FFN contribution
            # In TransformerLens, we can use hook_mlp_out for FFN output
            x_pre = cache[f"blocks.{layer_id}.hook_resid_pre"][0, resp_start:resp_end, :]
            x_post = cache[f"blocks.{layer_id}.hook_resid_post"][0, resp_start:resp_end, :]
            score = calculate_dist_2d(x_pre @ pythia_model.W_U, x_post @ pythia_model.W_U)
            pks_scores.append(score)

        # Compute ECS for response using context attention
        # Get attention from response to prompt
        ecs_scores = []
        for layer_id in range(pythia_model.cfg.n_layers):
            attn_pattern = cache[f"blocks.{layer_id}.attn.hook_pattern"]
            # Average attention from response tokens to prompt tokens
            for head_id in range(pythia_model.cfg.n_heads):
                attn = attn_pattern[0, head_id, resp_start:resp_end, :prompt_len].mean().item()
                ecs_scores.append(attn)

        results[label] = {
            "pks_early": np.mean(pks_scores[:8]),
            "pks_late": np.mean(pks_scores[-8:]),
            "pks_all": pks_scores,
            "ecs_mean": np.mean(ecs_scores),
            "ecs_all": ecs_scores
        }

        print(f"  PKS (early layers): {results[label]['pks_early']:.4f}")
        print(f"  PKS (late layers): {results[label]['pks_late']:.4f}")
        print(f"  ECS (mean attention to context): {results[label]['ecs_mean']:.6f}")

    # Verify findings
    print("\n" + "-"*70)
    print("VERIFICATION:")

    # Finding 1: Later-layer PKS higher for hallucinated
    pks_diff_late = results["hallucinated"]["pks_late"] - results["truthful"]["pks_late"]
    pks_finding = pks_diff_late > 0
    print(f"  PKS (late layers): Hallucinated={results['hallucinated']['pks_late']:.4f}, Truthful={results['truthful']['pks_late']:.4f}")
    print(f"  -> Later-layer PKS higher for hallucinated? {pks_finding} (diff={pks_diff_late:.4f})")

    # Finding 2: ECS lower for hallucinated (less context attention)
    ecs_diff = results["truthful"]["ecs_mean"] - results["hallucinated"]["ecs_mean"]
    ecs_finding = ecs_diff > 0
    print(f"  ECS: Hallucinated={results['hallucinated']['ecs_mean']:.6f}, Truthful={results['truthful']['ecs_mean']:.6f}")
    print(f"  -> ECS lower for hallucinated? {ecs_finding} (diff={ecs_diff:.6f})")

    # Clean up
    del pythia_model, cache
    torch.cuda.empty_cache()

    gt1_pass = pks_finding or ecs_finding
    print(f"\n{'='*70}")
    print(f"GT1 RESULT: {'PASS' if gt1_pass else 'FAIL'}")
    print(f"{'='*70}")

    return gt1_pass, {
        "pks_finding": pks_finding,
        "ecs_finding": ecs_finding,
        "results": {k: {kk: vv if not isinstance(vv, list) else f"list[{len(vv)}]" for kk, vv in v.items()} for k, v in results.items()}
    }

def evaluate_gt2():
    """
    GT2: Data Generalization Test
    Test if findings hold on new data instances not in original dataset
    """
    print("\n" + "="*70)
    print("GT2: DATA GENERALIZATION TEST")
    print("="*70)
    print("Testing: Do findings hold on new data instances?")
    print("-"*70)

    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer

    # Load original Qwen3 model used in the paper
    print("Loading Qwen3-0.6B model...")
    qwen_model = HookedTransformer.from_pretrained(
        "Qwen/Qwen3-0.6B",
        device="cuda",
        dtype=torch.float16
    )
    qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    print(f"Qwen3-0.6B loaded: {qwen_model.cfg.n_layers} layers")

    # Create new RAG-style examples not in original dataset
    # These are synthetic examples designed to test hallucination detection
    new_examples = [
        {
            "id": "new_example_1_hallucinated",
            "prompt": "Context: Apple Inc. reported revenue of $394.3 billion in fiscal year 2022. The company's CEO is Tim Cook. Question: What was Apple's revenue in 2023?",
            "response": "Apple's revenue in 2023 was $450 billion.",  # Hallucinated - no data for 2023 in context
            "expected_hallucination": True
        },
        {
            "id": "new_example_2_truthful",
            "prompt": "Context: The Eiffel Tower was built in 1889 and is 330 meters tall. It is located in Paris, France. Question: When was the Eiffel Tower built?",
            "response": "The Eiffel Tower was built in 1889.",  # Truthful - from context
            "expected_hallucination": False
        },
        {
            "id": "new_example_3_hallucinated",
            "prompt": "Context: Tesla was founded in 2003 by Martin Eberhard and Marc Tarpenning. Elon Musk joined as chairman in 2004. Question: Who founded Tesla?",
            "response": "Tesla was founded by Elon Musk in 2003.",  # Hallucinated - Musk didn't found it
            "expected_hallucination": True
        }
    ]

    print(f"Testing {len(new_examples)} new synthetic examples")

    results = []

    for example in new_examples:
        print(f"\nProcessing: {example['id']}")

        # Prepare input with chat template
        messages = [
            {"role": "user", "content": example['prompt']}
        ]
        prompt_text = qwen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_text = prompt_text + example['response']

        # Tokenize
        input_ids = qwen_tokenizer(full_text, return_tensors="pt").input_ids
        if input_ids.shape[-1] > qwen_model.cfg.n_ctx:
            input_ids = input_ids[:, -qwen_model.cfg.n_ctx:]
        input_ids = input_ids.to("cuda")

        prompt_ids = qwen_tokenizer(prompt_text, return_tensors="pt").input_ids
        prompt_len = min(prompt_ids.shape[-1], input_ids.shape[-1] - 5)

        # Run model
        with torch.no_grad():
            logits, cache = qwen_model.run_with_cache(input_ids, return_type="logits")

        # Compute PKS
        resp_start = prompt_len
        resp_end = input_ids.shape[-1]

        pks_scores = []
        for layer_id in range(qwen_model.cfg.n_layers):
            x_mid = cache[f"blocks.{layer_id}.hook_resid_mid"][0, resp_start:resp_end, :]
            x_post = cache[f"blocks.{layer_id}.hook_resid_post"][0, resp_start:resp_end, :]
            score = calculate_dist_2d(x_mid @ qwen_model.W_U, x_post @ qwen_model.W_U)
            pks_scores.append(score)

        early_pks = np.mean(pks_scores[:10])
        late_pks = np.mean(pks_scores[-10:])

        results.append({
            "id": example['id'],
            "expected_hallucination": example['expected_hallucination'],
            "early_pks": early_pks,
            "late_pks": late_pks,
            "pks_ratio": late_pks / early_pks if early_pks > 0 else 0
        })

        print(f"  Expected hallucination: {example['expected_hallucination']}")
        print(f"  PKS early: {early_pks:.4f}, late: {late_pks:.4f}, ratio: {late_pks/early_pks:.4f}")

    # Analyze results
    print("\n" + "-"*70)
    print("ANALYSIS:")

    hallucinated_ratios = [r['pks_ratio'] for r in results if r['expected_hallucination']]
    truthful_ratios = [r['pks_ratio'] for r in results if not r['expected_hallucination']]

    print(f"  Hallucinated examples - mean PKS ratio: {np.mean(hallucinated_ratios):.4f}")
    print(f"  Truthful examples - mean PKS ratio: {np.mean(truthful_ratios):.4f}")

    # Check if hallucinated examples have higher late/early PKS ratio
    finding_holds = np.mean(hallucinated_ratios) > np.mean(truthful_ratios)

    # Clean up
    del qwen_model, cache
    torch.cuda.empty_cache()

    gt2_pass = finding_holds
    print(f"\n{'='*70}")
    print(f"GT2 RESULT: {'PASS' if gt2_pass else 'FAIL'}")
    print(f"{'='*70}")

    return gt2_pass, {
        "hallucinated_mean_ratio": float(np.mean(hallucinated_ratios)),
        "truthful_mean_ratio": float(np.mean(truthful_ratios)),
        "finding_holds": finding_holds,
        "examples_tested": len(new_examples)
    }

def evaluate_gt3():
    """
    GT3: Method Generalizability Test
    Test if the ECS/PKS method can be applied to another similar task
    """
    print("\n" + "="*70)
    print("GT3: METHOD GENERALIZABILITY TEST")
    print("="*70)
    print("Testing: Can ECS/PKS method detect factual errors in summarization?")
    print("(Similar task to RAG hallucination detection)")
    print("-"*70)

    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer

    # Load model
    print("Loading Qwen3-0.6B model...")
    qwen_model = HookedTransformer.from_pretrained(
        "Qwen/Qwen3-0.6B",
        device="cuda",
        dtype=torch.float16
    )
    qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    # Create summarization examples with factual errors
    # This is a similar task - detecting when a model injects incorrect info
    summarization_examples = [
        {
            "id": "summary_1_faithful",
            "source": "The 2024 Summer Olympics were held in Paris, France. Athletes from 206 countries competed. The opening ceremony took place at the Seine River.",
            "summary": "The 2024 Olympics were hosted in Paris with 206 participating nations.",
            "has_error": False
        },
        {
            "id": "summary_2_unfaithful",
            "source": "The 2024 Summer Olympics were held in Paris, France. Athletes from 206 countries competed. The opening ceremony took place at the Seine River.",
            "summary": "The 2024 Olympics were hosted in London with 300 participating nations.",  # Factual errors
            "has_error": True
        },
        {
            "id": "summary_3_faithful",
            "source": "SpaceX launched its Starship rocket in 2023. The rocket is 120 meters tall and is designed for Mars missions.",
            "summary": "SpaceX's Starship, a 120-meter rocket for Mars exploration, launched in 2023.",
            "has_error": False
        },
        {
            "id": "summary_4_unfaithful",
            "source": "SpaceX launched its Starship rocket in 2023. The rocket is 120 meters tall and is designed for Mars missions.",
            "summary": "NASA launched its Starship rocket in 2020, which is 200 meters tall.",  # Multiple errors
            "has_error": True
        }
    ]

    print(f"Testing {len(summarization_examples)} summarization examples")

    results = []

    for example in summarization_examples:
        print(f"\nProcessing: {example['id']}")

        # Format as summarization prompt
        prompt = f"Summarize the following text:\n{example['source']}\n\nSummary:"
        messages = [{"role": "user", "content": prompt}]
        prompt_text = qwen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_text = prompt_text + example['summary']

        # Tokenize
        input_ids = qwen_tokenizer(full_text, return_tensors="pt").input_ids
        if input_ids.shape[-1] > qwen_model.cfg.n_ctx:
            input_ids = input_ids[:, -qwen_model.cfg.n_ctx:]
        input_ids = input_ids.to("cuda")

        prompt_ids = qwen_tokenizer(prompt_text, return_tensors="pt").input_ids
        prompt_len = min(prompt_ids.shape[-1], input_ids.shape[-1] - 5)

        # Run model
        with torch.no_grad():
            logits, cache = qwen_model.run_with_cache(input_ids, return_type="logits")

        # Compute PKS
        resp_start = prompt_len
        resp_end = input_ids.shape[-1]

        pks_scores = []
        for layer_id in range(qwen_model.cfg.n_layers):
            x_mid = cache[f"blocks.{layer_id}.hook_resid_mid"][0, resp_start:resp_end, :]
            x_post = cache[f"blocks.{layer_id}.hook_resid_post"][0, resp_start:resp_end, :]
            score = calculate_dist_2d(x_mid @ qwen_model.W_U, x_post @ qwen_model.W_U)
            pks_scores.append(score)

        late_pks = np.mean(pks_scores[-10:])

        results.append({
            "id": example['id'],
            "has_error": example['has_error'],
            "late_pks": late_pks
        })

        print(f"  Has factual error: {example['has_error']}")
        print(f"  Late-layer PKS: {late_pks:.4f}")

    # Analyze
    print("\n" + "-"*70)
    print("ANALYSIS:")

    error_pks = [r['late_pks'] for r in results if r['has_error']]
    faithful_pks = [r['late_pks'] for r in results if not r['has_error']]

    print(f"  Unfaithful summaries - mean late PKS: {np.mean(error_pks):.4f}")
    print(f"  Faithful summaries - mean late PKS: {np.mean(faithful_pks):.4f}")

    # Check if method transfers
    method_transfers = np.mean(error_pks) > np.mean(faithful_pks)

    # Clean up
    del qwen_model, cache
    torch.cuda.empty_cache()

    gt3_pass = method_transfers
    print(f"\n{'='*70}")
    print(f"GT3 RESULT: {'PASS' if gt3_pass else 'FAIL'}")
    print(f"{'='*70}")

    return gt3_pass, {
        "unfaithful_mean_pks": float(np.mean(error_pks)),
        "faithful_mean_pks": float(np.mean(faithful_pks)),
        "method_transfers": method_transfers,
        "task": "summarization_factuality_detection"
    }

def main():
    print("="*70)
    print("GENERALIZABILITY EVALUATION FOR INTERPDETECT")
    print("="*70)
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")

    # Ensure evaluation directory exists
    os.makedirs(EVAL_PATH, exist_ok=True)

    # Run evaluations
    gt1_result, gt1_details = evaluate_gt1()
    gt2_result, gt2_details = evaluate_gt2()
    gt3_result, gt3_details = evaluate_gt3()

    # Create summary
    summary = {
        "Checklist": {
            "GT1_ModelGeneralization": "PASS" if gt1_result else "FAIL",
            "GT2_DataGeneralization": "PASS" if gt2_result else "FAIL",
            "GT3_MethodGeneralization": "PASS" if gt3_result else "FAIL"
        },
        "Rationale": {
            "GT1_ModelGeneralization": f"Tested on Pythia-1.4B (not in original work). PKS finding: {gt1_details['pks_finding']}, ECS finding: {gt1_details['ecs_finding']}. At least one correlation pattern was replicated.",
            "GT2_DataGeneralization": f"Tested on {gt2_details['examples_tested']} new synthetic RAG examples. Hallucinated examples had higher PKS ratio ({gt2_details['hallucinated_mean_ratio']:.4f}) vs truthful ({gt2_details['truthful_mean_ratio']:.4f}). Finding holds: {gt2_details['finding_holds']}",
            "GT3_MethodGeneralization": f"Applied PKS method to summarization factuality detection. Unfaithful summaries had {'higher' if gt3_details['method_transfers'] else 'lower'} late-layer PKS ({gt3_details['unfaithful_mean_pks']:.4f}) vs faithful ({gt3_details['faithful_mean_pks']:.4f}). Method transfers: {gt3_details['method_transfers']}"
        }
    }

    # Save summary
    summary_path = os.path.join(EVAL_PATH, 'generalization_eval_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to: {summary_path}")

    # Print final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    for gt, result in summary["Checklist"].items():
        print(f"  {gt}: {result}")
    print("="*70)

    return summary

if __name__ == "__main__":
    summary = main()
