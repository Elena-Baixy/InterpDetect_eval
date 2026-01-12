#!/usr/bin/env python3
"""
InterpDetect Replication Study

This script replicates the InterpDetect experiment: detecting hallucinations
in RAG systems using interpretability signals (ECS and PKS).

Goals:
1. Reproduce correlation analysis between ECS/PKS and hallucination labels
2. Train classifiers (Logistic Regression, SVC, Random Forest, XGBoost)
3. Evaluate predictions at span and response level
4. Compare with the reported results in plan.md
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import pickle
from scipy.stats import pointbiserialr, pearsonr
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# Try to import xgboost, skip if not available
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    print("Warning: XGBoost not available, will skip XGBoost model")
    HAS_XGBOOST = False

import torch

# Set working directory
os.chdir('/net/scratch2/smallyan/InterpDetect_eval')

# Create output directory
os.makedirs('evaluation/replications', exist_ok=True)

print("=" * 80)
print("InterpDetect Replication Study")
print("=" * 80)
print(f"\nWorking directory: {os.getcwd()}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch version: {torch.__version__}")

# ============================================================================
# 1. Load and Explore Data
# ============================================================================
print("\n" + "=" * 80)
print("1. LOADING DATA")
print("=" * 80)

def load_training_data(folder_path):
    """Load training data from JSON files in the specified folder"""
    data = []
    json_files = sorted(glob.glob(os.path.join(folder_path, "*.json")))
    print(f"Found {len(json_files)} JSON files in {folder_path}")

    for file_path in json_files:
        with open(file_path, "r") as f:
            batch = json.load(f)
            data.extend(batch)

    print(f"Loaded {len(data)} examples total")
    return data

def load_test_data(file_path):
    """Load test data from single JSON file"""
    with open(file_path, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} test examples from {file_path}")
    return data

# Load training data
train_data = load_training_data("datasets/train")

# Load test data (Qwen-generated responses for self-evaluation)
test_data_qwen = load_test_data("datasets/test/test_w_chunk_score_qwen06b.json")

# Load test data (GPT-4.1-mini responses for proxy evaluation)
test_data_gpt = load_test_data("datasets/test/test_w_chunk_score_gpt41mini.json")

# Examine the data structure
sample = train_data[0]
print(f"\nData structure:")
print(f"  Keys in example: {list(sample.keys())}")
print(f"  Number of response spans (scores): {len(sample['scores'])}")
score_sample = sample['scores'][0]
print(f"  Attention head scores: {len(score_sample['prompt_attention_score'])}")
print(f"  PKS layer scores: {len(score_sample['parameter_knowledge_scores'])}")

# ============================================================================
# 2. Correlation Analysis
# ============================================================================
print("\n" + "=" * 80)
print("2. CORRELATION ANALYSIS")
print("=" * 80)

def extract_scores_and_labels(data):
    """Extract ECS, PKS scores and hallucination labels from data"""
    ecs_scores = []  # External Context Scores (attention-based)
    pks_scores = []  # Parametric Knowledge Scores
    labels = []

    # Also collect per-layer/head scores for detailed analysis
    attention_by_head = {}  # key: (layer, head), value: list of scores
    pks_by_layer = {}  # key: layer, value: list of scores

    for example in data:
        for score_item in example['scores']:
            # Sum all attention scores (ECS)
            attn_scores = score_item['prompt_attention_score']
            ecs_sum = sum(attn_scores.values())
            ecs_scores.append(ecs_sum)

            # Sum all PKS scores
            pks_vals = score_item['parameter_knowledge_scores']
            pks_sum = sum(pks_vals.values())
            pks_scores.append(pks_sum)

            labels.append(score_item['hallucination_label'])

            # Collect per-head attention scores
            for key, val in attn_scores.items():
                if key not in attention_by_head:
                    attention_by_head[key] = []
                attention_by_head[key].append(val)

            # Collect per-layer PKS scores
            for key, val in pks_vals.items():
                if key not in pks_by_layer:
                    pks_by_layer[key] = []
                pks_by_layer[key].append(val)

    return {
        'ecs_scores': np.array(ecs_scores),
        'pks_scores': np.array(pks_scores),
        'labels': np.array(labels),
        'attention_by_head': attention_by_head,
        'pks_by_layer': pks_by_layer
    }

# Extract scores from training data
train_scores = extract_scores_and_labels(train_data)
print(f"Total span-level samples: {len(train_scores['labels'])}")
print(f"Hallucination rate: {train_scores['labels'].mean():.2%}")

# Overall correlation analysis
ecs_corr, ecs_pval = pointbiserialr(train_scores['labels'], train_scores['ecs_scores'])
pks_corr, pks_pval = pointbiserialr(train_scores['labels'], train_scores['pks_scores'])

print(f"\nECS (External Context Score) vs Hallucination:")
print(f"  Point-biserial correlation: {ecs_corr:.4f} (p={ecs_pval:.2e})")
print(f"  Interpretation: {'Negative' if ecs_corr < 0 else 'Positive'} correlation")

print(f"\nPKS (Parametric Knowledge Score) vs Hallucination:")
print(f"  Point-biserial correlation: {pks_corr:.4f} (p={pks_pval:.2e})")
print(f"  Interpretation: {'Positive' if pks_corr > 0 else 'Negative'} correlation")

# Per-layer PKS correlation analysis
print("\nPer-layer PKS correlation with hallucination:")
layer_correlations = []
layer_names = sorted(train_scores['pks_by_layer'].keys(),
                    key=lambda x: int(x.split('_')[1]))

for layer_name in layer_names:
    layer_scores = np.array(train_scores['pks_by_layer'][layer_name])
    corr, pval = pointbiserialr(train_scores['labels'], layer_scores)
    layer_correlations.append({'layer': layer_name, 'correlation': corr, 'p_value': pval})

layer_df = pd.DataFrame(layer_correlations)
print(layer_df.to_string())

# Plot layer correlations
fig, ax = plt.subplots(figsize=(12, 5))
layer_nums = [int(l.split('_')[1]) for l in layer_df['layer']]
ax.bar(layer_nums, layer_df['correlation'],
       color=['red' if c > 0 else 'blue' for c in layer_df['correlation']])
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('Layer')
ax.set_ylabel('Correlation with Hallucination')
ax.set_title('PKS Correlation with Hallucination by Layer\n(Positive = more PKS in hallucinated spans)')
plt.tight_layout()
plt.savefig('evaluation/replications/pks_correlation_by_layer.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: evaluation/replications/pks_correlation_by_layer.png")

# Per-head ECS correlation analysis
print("\nComputing per-head ECS correlations...")
head_correlations = []
for head_key in train_scores['attention_by_head'].keys():
    head_scores = np.array(train_scores['attention_by_head'][head_key])
    corr, pval = pointbiserialr(train_scores['labels'], head_scores)
    # Parse layer and head from key like "(0, 0)"
    layer, head = eval(head_key)
    head_correlations.append({
        'layer': layer,
        'head': head,
        'correlation': corr,
        'p_value': pval
    })

head_df = pd.DataFrame(head_correlations)
head_df = head_df.sort_values(['layer', 'head'])

# Create heatmap of correlations
n_layers = head_df['layer'].max() + 1
n_heads = head_df['head'].max() + 1
corr_matrix = np.zeros((n_layers, n_heads))

for _, row in head_df.iterrows():
    corr_matrix[int(row['layer']), int(row['head'])] = row['correlation']

fig, ax = plt.subplots(figsize=(14, 10))
im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.15, vmax=0.15)
ax.set_xlabel('Head')
ax.set_ylabel('Layer')
ax.set_title('ECS Correlation with Hallucination by Layer and Head\n(Negative = less context attention in hallucinated spans)')
plt.colorbar(im, ax=ax, label='Correlation')
plt.tight_layout()
plt.savefig('evaluation/replications/ecs_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: evaluation/replications/ecs_correlation_heatmap.png")

print(f"\nECS Summary:")
print(f"  Mean correlation across all heads: {head_df['correlation'].mean():.4f}")
print(f"  Negative correlation heads: {(head_df['correlation'] < 0).sum()} / {len(head_df)}")

# ============================================================================
# 3. Classifier Training
# ============================================================================
print("\n" + "=" * 80)
print("3. CLASSIFIER TRAINING")
print("=" * 80)

def prepare_dataframe(data):
    """Convert data to DataFrame with all features"""
    # Get column names from first example
    first_score = data[0]['scores'][0]
    attention_cols = list(first_score['prompt_attention_score'].keys())
    pks_cols = list(first_score['parameter_knowledge_scores'].keys())

    records = []
    for i, example in enumerate(data):
        for j, score in enumerate(example['scores']):
            record = {
                'identifier': f"response_{i}_item_{j}",
                'hallucination_label': score['hallucination_label']
            }
            # Add attention scores
            for col in attention_cols:
                record[col] = score['prompt_attention_score'][col]
            # Add PKS scores
            for col in pks_cols:
                record[col] = score['parameter_knowledge_scores'][col]
            records.append(record)

    df = pd.DataFrame(records)
    return df, attention_cols, pks_cols

# Prepare training data
train_df, attention_cols, pks_cols = prepare_dataframe(train_data)
print(f"Training DataFrame shape: {train_df.shape}")
print(f"Features: {len(attention_cols)} attention + {len(pks_cols)} PKS = {len(attention_cols) + len(pks_cols)} total")
print(f"\nClass distribution:")
print(train_df['hallucination_label'].value_counts())

# Balance classes by undersampling
min_class_count = train_df['hallucination_label'].value_counts().min()
balanced_df = (
    train_df.groupby('hallucination_label', group_keys=False)
    .apply(lambda x: x.sample(min_class_count, random_state=42))
)
print(f"\nAfter balancing: {len(balanced_df)} samples")
print(balanced_df['hallucination_label'].value_counts())

# Split into train/validation
feature_cols = attention_cols + pks_cols
X = balanced_df[feature_cols]
y = balanced_df['hallucination_label']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

print(f"\nTrain set: {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples")

# Train classifiers
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'SVC': SVC(),
    'RandomForest': RandomForestClassifier(max_depth=5, random_state=42),
}

if HAS_XGBOOST:
    models['XGBoost'] = XGBClassifier(max_depth=5, random_state=42,
                                       use_label_encoder=False, eval_metric='logloss')

trained_models = {}
results = []

print("\nTraining classifiers...")
print("-" * 80)

for name, model in models.items():
    print(f"\nTraining {name}...")

    # Create pipeline with scaler
    clf = make_pipeline(StandardScaler(), model)
    clf.fit(X_train, y_train)

    # Training metrics
    train_pred = clf.predict(X_train)
    train_p, train_r, train_f, _ = precision_recall_fscore_support(
        y_train, train_pred, average='binary'
    )

    # Validation metrics
    val_pred = clf.predict(X_val)
    val_p, val_r, val_f, _ = precision_recall_fscore_support(
        y_val, val_pred, average='binary'
    )

    trained_models[name] = clf
    results.append({
        'Model': name,
        'Train_Precision': train_p,
        'Train_Recall': train_r,
        'Train_F1': train_f,
        'Val_Precision': val_p,
        'Val_Recall': val_r,
        'Val_F1': val_f
    })

    print(f"  Train: P={train_p:.4f}, R={train_r:.4f}, F1={train_f:.4f}")
    print(f"  Val:   P={val_p:.4f}, R={val_r:.4f}, F1={val_f:.4f}")

# Summary table
results_df = pd.DataFrame(results)
print("\n" + "-" * 80)
print("MODEL COMPARISON SUMMARY")
print("-" * 80)
print(results_df.to_string(index=False))

# Visualize model comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x = np.arange(len(results_df))
width = 0.25

axes[0].bar(x - width, results_df['Train_Precision'], width, label='Precision')
axes[0].bar(x, results_df['Train_Recall'], width, label='Recall')
axes[0].bar(x + width, results_df['Train_F1'], width, label='F1')
axes[0].set_xticks(x)
axes[0].set_xticklabels(results_df['Model'], rotation=45, ha='right')
axes[0].set_ylabel('Score')
axes[0].set_title('Training Metrics')
axes[0].legend()
axes[0].set_ylim(0, 1)

axes[1].bar(x - width, results_df['Val_Precision'], width, label='Precision')
axes[1].bar(x, results_df['Val_Recall'], width, label='Recall')
axes[1].bar(x + width, results_df['Val_F1'], width, label='F1')
axes[1].set_xticks(x)
axes[1].set_xticklabels(results_df['Model'], rotation=45, ha='right')
axes[1].set_ylabel('Score')
axes[1].set_title('Validation Metrics')
axes[1].legend()
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('evaluation/replications/model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: evaluation/replications/model_comparison.png")

# ============================================================================
# 4. Test Set Evaluation
# ============================================================================
print("\n" + "=" * 80)
print("4. TEST SET EVALUATION")
print("=" * 80)

def evaluate_on_test(model, test_data, model_name="Model", dataset_name="Test"):
    """Evaluate model on test data at both span and response level"""
    # Prepare test dataframe
    test_df, _, _ = prepare_dataframe(test_data)

    # Get features
    feature_cols = [c for c in test_df.columns if c not in ['identifier', 'hallucination_label']]
    X_test = test_df[feature_cols]
    y_test = test_df['hallucination_label']

    # Make predictions
    y_pred = model.predict(X_test)
    test_df['pred'] = y_pred

    # Span-level metrics
    span_precision = precision_score(y_test, y_pred)
    span_recall = recall_score(y_test, y_pred)
    span_f1 = f1_score(y_test, y_pred)

    # Response-level metrics (aggregate by OR)
    test_df['response_id'] = test_df['identifier'].str.extract(r'(response_\d+)_item_\d+')
    response_agg = test_df.groupby('response_id').agg({
        'pred': 'max',
        'hallucination_label': 'max'
    }).reset_index()

    resp_precision = precision_score(response_agg['hallucination_label'], response_agg['pred'])
    resp_recall = recall_score(response_agg['hallucination_label'], response_agg['pred'])
    resp_f1 = f1_score(response_agg['hallucination_label'], response_agg['pred'])

    return {
        'model': model_name,
        'dataset': dataset_name,
        'span_precision': span_precision,
        'span_recall': span_recall,
        'span_f1': span_f1,
        'response_precision': resp_precision,
        'response_recall': resp_recall,
        'response_f1': resp_f1,
        'n_responses': len(response_agg)
    }

# Evaluate all models on both test sets
all_results = []

for model_name, model in trained_models.items():
    # Self-evaluation (Qwen)
    qwen_results = evaluate_on_test(model, test_data_qwen, model_name, "Qwen (Self-eval)")
    all_results.append(qwen_results)

    # Proxy evaluation (GPT-4.1-mini)
    gpt_results = evaluate_on_test(model, test_data_gpt, model_name, "GPT-4.1-mini (Proxy)")
    all_results.append(gpt_results)

test_results_df = pd.DataFrame(all_results)

print("\n--- Self-Evaluation (Qwen-generated responses) ---")
qwen_df = test_results_df[test_results_df['dataset'] == 'Qwen (Self-eval)'][[
    'model', 'span_f1', 'response_precision', 'response_recall', 'response_f1'
]]
print(qwen_df.to_string(index=False))

print("\n--- Proxy Evaluation (GPT-4.1-mini responses) ---")
gpt_df = test_results_df[test_results_df['dataset'] == 'GPT-4.1-mini (Proxy)'][[
    'model', 'span_f1', 'response_precision', 'response_recall', 'response_f1'
]]
print(gpt_df.to_string(index=False))

# ============================================================================
# 5. Comparison with Pre-trained Models
# ============================================================================
print("\n" + "=" * 80)
print("5. COMPARISON WITH PRE-TRAINED MODELS FROM REPOSITORY")
print("=" * 80)

pretrained_results = []

for model_file in sorted(glob.glob('trained_models/model_*.pickle')):
    model_name = os.path.basename(model_file).replace('model_', '').replace('_3000.pickle', '')

    with open(model_file, 'rb') as f:
        pretrained_model = pickle.load(f)

    # Evaluate on both test sets
    qwen_res = evaluate_on_test(pretrained_model, test_data_qwen, f"PreTrained_{model_name}", "Qwen")
    gpt_res = evaluate_on_test(pretrained_model, test_data_gpt, f"PreTrained_{model_name}", "GPT-4.1-mini")

    pretrained_results.append(qwen_res)
    pretrained_results.append(gpt_res)

pretrained_df = pd.DataFrame(pretrained_results)

print("\n--- Pre-trained Model Results on Qwen Test Set ---")
qwen_pretrained = pretrained_df[pretrained_df['dataset'] == 'Qwen'][[
    'model', 'response_precision', 'response_recall', 'response_f1'
]]
print(qwen_pretrained.to_string(index=False))

print("\n--- Pre-trained Model Results on GPT-4.1-mini Test Set ---")
gpt_pretrained = pretrained_df[pretrained_df['dataset'] == 'GPT-4.1-mini'][[
    'model', 'response_precision', 'response_recall', 'response_f1'
]]
print(gpt_pretrained.to_string(index=False))

# ============================================================================
# 6. Results Summary
# ============================================================================
print("\n" + "=" * 80)
print("6. REPLICATION SUMMARY")
print("=" * 80)

print("\n1. CORRELATION ANALYSIS")
print("-" * 50)
print(f"   ECS vs Hallucination: {ecs_corr:.4f} (expected: negative)")
print(f"   PKS vs Hallucination: {pks_corr:.4f} (expected: positive for later layers)")
print(f"\n   Plan claims: 'All attention heads exhibit negative correlations'")
print(f"   Replicated: {(head_df['correlation'] < 0).sum()}/{len(head_df)} heads show negative correlation")

print("\n2. CLASSIFIER TRAINING")
print("-" * 50)
print(f"   Plan reports SVC achieved highest validation F1: 76.60%")
best_model = results_df.loc[results_df['Val_F1'].idxmax()]
print(f"   Replicated best model: {best_model['Model']} with Val F1: {best_model['Val_F1']*100:.2f}%")

print("\n3. SELF-EVALUATION (Qwen responses)")
print("-" * 50)
print(f"   Plan reports method F1: 74.68%")
svc_qwen = test_results_df[(test_results_df['model'] == 'SVC') &
                           (test_results_df['dataset'] == 'Qwen (Self-eval)')].iloc[0]
print(f"   Replicated SVC Response-level F1: {svc_qwen['response_f1']*100:.2f}%")

print("\n4. PROXY EVALUATION (GPT-4.1-mini responses)")
print("-" * 50)
print(f"   Plan reports method F1: 75.36%")
svc_gpt = test_results_df[(test_results_df['model'] == 'SVC') &
                          (test_results_df['dataset'] == 'GPT-4.1-mini (Proxy)')].iloc[0]
print(f"   Replicated SVC Response-level F1: {svc_gpt['response_f1']*100:.2f}%")

# Final comparison table
comparison_data = {
    'Metric': [
        'ECS-Hallucination Correlation (sign)',
        'PKS-Hallucination Correlation (sign)',
        'Best Val F1 Model',
        'Self-eval Response F1 (SVC)',
        'Proxy-eval Response F1 (SVC)'
    ],
    'Plan_Reported': [
        'Negative',
        'Positive (later layers)',
        'SVC (76.60%)',
        '74.68%',
        '75.36%'
    ],
    'Replicated': [
        f"{'Negative' if ecs_corr < 0 else 'Positive'} ({ecs_corr:.4f})",
        f"{'Positive' if pks_corr > 0 else 'Negative'} ({pks_corr:.4f})",
        f"{best_model['Model']} ({best_model['Val_F1']*100:.2f}%)",
        f"{svc_qwen['response_f1']*100:.2f}%",
        f"{svc_gpt['response_f1']*100:.2f}%"
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n" + "=" * 80)
print("FINAL COMPARISON TABLE")
print("=" * 80)
print(comparison_df.to_string(index=False))

# Save comparison
comparison_df.to_csv('evaluation/replications/comparison_results.csv', index=False)
print("\nSaved: evaluation/replications/comparison_results.csv")

# Save all results
all_results_data = {
    'training_results': results_df.to_dict('records'),
    'test_results': test_results_df.to_dict('records'),
    'pretrained_results': pretrained_df.to_dict('records'),
    'correlation_analysis': {
        'ecs_correlation': ecs_corr,
        'ecs_pvalue': ecs_pval,
        'pks_correlation': pks_corr,
        'pks_pvalue': pks_pval,
        'negative_ecs_heads': int((head_df['correlation'] < 0).sum()),
        'total_heads': len(head_df)
    }
}

with open('evaluation/replications/all_results.json', 'w') as f:
    json.dump(all_results_data, f, indent=2)
print("Saved: evaluation/replications/all_results.json")

# Document issues
issues = [
    "1. Data files are pre-computed scores (ECS/PKS), not raw data - full pipeline from score extraction was not re-run.",
    "2. Class balancing strategy (undersampling) was inferred from code, not explicitly stated in plan.",
    "3. Test/validation split ratio (0.1) was inferred from code defaults.",
    "4. Exact hyperparameters for classifiers (e.g., max_depth=5 for RF/XGBoost) were inferred from code.",
    f"5. The plan mentions '7,799 span-level samples' for training; actual loaded samples: {len(train_scores['labels'])}."
]

print("\n" + "=" * 80)
print("ISSUES AND OBSERVATIONS DURING REPLICATION")
print("=" * 80)
for issue in issues:
    print(f"\n{issue}")

with open('evaluation/replications/replication_issues.txt', 'w') as f:
    f.write("\n".join(issues))
print("\nSaved: evaluation/replications/replication_issues.txt")

print("\n" + "=" * 80)
print("REPLICATION COMPLETE")
print("=" * 80)
print("\nGenerated files:")
print("  - evaluation/replications/pks_correlation_by_layer.png")
print("  - evaluation/replications/ecs_correlation_heatmap.png")
print("  - evaluation/replications/model_comparison.png")
print("  - evaluation/replications/comparison_results.csv")
print("  - evaluation/replications/all_results.json")
print("  - evaluation/replications/replication_issues.txt")
