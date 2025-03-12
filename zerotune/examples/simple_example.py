"""
Simple example script demonstrating ZeroTune functionality.

This example:
1. Creates a synthetic dataset
2. Calculates dataset meta-features
3. Sets up parameter configuration for a decision tree
4. Runs Optuna HPO 
5. Compares with random hyperparameter search

You can use this as a template for your own experiments.
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Import ZeroTune
import sys
sys.path.append("../../")  # Add parent directory to path
from zerotune import (
    calculate_dataset_meta_parameters,
    optuna_hpo,
    random_hyperparameter_evaluation
)

# Create output directory
os.makedirs("output", exist_ok=True)

# Generate a synthetic dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    flip_y=0.05,
    random_state=42
)

# Convert to pandas DataFrame/Series for better handling
X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
y = pd.Series(y)

# Calculate dataset meta-parameters
meta_params = calculate_dataset_meta_parameters(X, y)
print("Dataset meta-parameters:")
for key, value in meta_params.items():
    print(f"  {key}: {value}")

# Define parameter configuration for DecisionTreeClassifier
param_config = {
    "max_depth": {
        "percentage_splits": [0.25, 0.5, 0.7, 0.8, 0.9, 0.999], 
        "param_type": "int", 
        "dependency": "n_samples"
    },
    "min_samples_split": {
        "percentage_splits": [0.005, 0.01, 0.02, 0.05, 0.1], 
        "param_type": "float"
    },
    "min_samples_leaf": {
        "percentage_splits": [0.005, 0.01, 0.02, 0.05, 0.1], 
        "param_type": "float"
    },
    "max_features": {
        "percentage_splits": [0.5, 0.7, 0.8, 0.9, 0.99], 
        "param_type": "float"
    }
}

# Define which meta-parameters to use
selected_meta_params = ["n_samples", "n_features", "n_highly_target_corr"]

# Run random hyperparameter search
print("\nRunning random hyperparameter search...")
rand_hyperparams, rand_perf, rand_scores = random_hyperparameter_evaluation(
    X, y,
    meta_params=selected_meta_params,
    param_config=param_config,
    model_name="DecisionTreeClassifier4Param",
    random_seed=42,
    n_seeds=5
)

print(f"Random hyperparameter performance: {rand_perf:.4f}")
print(f"Random hyperparameters: {rand_hyperparams}")

# Run Optuna HPO
print("\nRunning Optuna hyperparameter optimization...")
optuna_results = optuna_hpo(
    X, y,
    meta_params=selected_meta_params,
    param_config=param_config,
    n_trials=50,
    n_seeds=5,
    model_name="DecisionTreeClassifier4Param"
)

print(f"Optuna best performance: {optuna_results['average_best_perf']:.4f}")
print(f"Optuna best hyperparameters: {optuna_results['all_results'][0]['best_hyperparams']}")

# Save results to CSV
optuna_results['combined_trials_df'].to_csv("output/optuna_trials.csv", index=False)
print("\nResults saved to output/optuna_trials.csv")

# Comparison
print("\nPerformance Comparison:")
print(f"Random Search: {rand_perf:.4f}")
print(f"Optuna HPO:    {optuna_results['average_best_perf']:.4f}")
print(f"Improvement:   {(optuna_results['average_best_perf'] - rand_perf) / rand_perf * 100:.2f}%") 