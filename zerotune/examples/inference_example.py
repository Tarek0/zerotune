"""
Example demonstrating ZeroTune for hyperparameter optimization on a standard dataset.

This example:
1. Loads the breast cancer dataset
2. Uses ZeroTune to optimize hyperparameters for multiple model types
3. Evaluates and compares the models with optimized hyperparameters
4. Compares against default hyperparameters
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Import ZeroTune
from zerotune import ZeroTune
from zerotune.core import CONFIG
from zerotune.core.utils import safe_json_serialize, save_json

# Load the breast cancer dataset
print("Loading breast cancer dataset...")
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print(f"Dataset shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")
print(f"Class distribution: {y.value_counts().to_dict()}")

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=CONFIG["defaults"]["test_size"], 
    random_state=CONFIG["defaults"]["random_state"]
)

# Create output directory
os.makedirs("output", exist_ok=True)

# Try different model types
model_types = CONFIG["supported"]["models"]
results = {}
default_results = {}

print("\n=== Optimizing Hyperparameters ===")

for model_type in model_types:
    print(f"\n--- {model_type.upper()} ---")
    
    # Initialize ZeroTune
    zt = ZeroTune(model_type=model_type, kb_path=f"output/{model_type}_kb.json")
    
    # Optimize hyperparameters
    print(f"Optimizing hyperparameters for {model_type}...")
    best_params, best_score, model = zt.optimize(
        X_train, y_train, n_iter=5, verbose=True  # Reduced iterations for example
    )
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    except:
        roc_auc = None
    
    # Store results
    results[model_type] = {
        "params": best_params,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "model": model
    }
    
    print(f"\nResults for {model_type}:")
    print(f"Accuracy: {accuracy:.4f}")
    if roc_auc:
        print(f"ROC AUC: {roc_auc:.4f}")
    print("Best hyperparameters:", best_params)

    # Compare with default model
    print("\nComparing with default hyperparameters...")
    if model_type == "decision_tree":
        default_model = DecisionTreeClassifier(random_state=CONFIG["defaults"]["random_state"])
    elif model_type == "random_forest":
        default_model = RandomForestClassifier(random_state=CONFIG["defaults"]["random_state"])
    elif model_type == "xgboost":
        default_model = XGBClassifier(
            random_state=CONFIG["defaults"]["random_state"],
            enable_categorical=True
        )
    
    default_model.fit(X_train, y_train)
    default_y_pred = default_model.predict(X_test)
    default_accuracy = accuracy_score(y_test, default_y_pred)
    
    try:
        default_y_pred_proba = default_model.predict_proba(X_test)[:, 1]
        default_roc_auc = roc_auc_score(y_test, default_y_pred_proba)
    except:
        default_roc_auc = None
    
    default_results[model_type] = {
        "accuracy": default_accuracy,
        "roc_auc": default_roc_auc
    }
    
    print(f"Default model accuracy: {default_accuracy:.4f}")
    if default_roc_auc:
        print(f"Default model ROC AUC: {default_roc_auc:.4f}")
    
    # Calculate improvement
    acc_improvement = ((accuracy - default_accuracy) / default_accuracy) * 100
    print(f"Accuracy improvement: {acc_improvement:.2f}%")
    
    if roc_auc and default_roc_auc:
        auc_improvement = ((roc_auc - default_roc_auc) / default_roc_auc) * 100
        print(f"ROC AUC improvement: {auc_improvement:.2f}%")

# Save results to JSON file
output_results = {
    "optimized": {model: {
        "params": results[model]["params"],
        "accuracy": results[model]["accuracy"],
        "roc_auc": results[model]["roc_auc"]
    } for model in model_types},
    "default": default_results
}
save_json(output_results, "output/comparison_results.json")
print("\nResults saved to output/comparison_results.json")

# Print comparison table
print("\n\n=== Model Comparison ===\n")
print(f"{'Model Type':<15} {'Optimized Accuracy':<20} {'Default Accuracy':<20} {'Improvement':<15}")
print("-" * 70)

for model_type in model_types:
    opt_acc = results[model_type]['accuracy']
    def_acc = default_results[model_type]['accuracy']
    imp = ((opt_acc - def_acc) / def_acc) * 100
    print(f"{model_type:<15} {opt_acc:<20.4f} {def_acc:<20.4f} {imp:<15.2f}%")

if all(results[mt]['roc_auc'] is not None for mt in model_types):
    print("\n")
    print(f"{'Model Type':<15} {'Optimized ROC AUC':<20} {'Default ROC AUC':<20} {'Improvement':<15}")
    print("-" * 70)
    
    for model_type in model_types:
        opt_auc = results[model_type]['roc_auc']
        def_auc = default_results[model_type]['roc_auc']
        imp = ((opt_auc - def_auc) / def_auc) * 100
        print(f"{model_type:<15} {opt_auc:<20.4f} {def_auc:<20.4f} {imp:<15.2f}%")

# Find the best model
best_model_type = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
print(f"\nBest model type: {best_model_type}")
print(f"Best model accuracy: {results[best_model_type]['accuracy']:.4f}")
if results[best_model_type]['roc_auc']:
    print(f"Best model ROC AUC: {results[best_model_type]['roc_auc']:.4f}")

# Print detailed classification report for the best model
print("\nClassification report for the best model:")
y_pred = results[best_model_type]['model'].predict(X_test)
print(classification_report(y_test, y_pred)) 