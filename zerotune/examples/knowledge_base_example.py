"""
Example script demonstrating how to build a ZeroTune knowledge base and train a custom model.

This example:
1. Creates a new knowledge base
2. Adds synthetic datasets
3. Compiles and saves the knowledge base
4. Trains a ZeroTune model
5. Uses the trained model to make predictions on a new dataset
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

# Import ZeroTune
import sys
sys.path.append("../../")  # Add parent directory to path
from zerotune import (
    KnowledgeBase,
    CustomZeroTunePredictor
)

# Create output directory
output_dir = "kb_example_output"
os.makedirs(output_dir, exist_ok=True)

# Knowledge base name
kb_name = "example_decision_tree_kb"

print("Building ZeroTune knowledge base...")

# Create a new knowledge base
kb = KnowledgeBase(name=kb_name, save_dir=output_dir)

# Generate a small number of synthetic datasets for the example
# In a real-world scenario, you would use more datasets
print("\nGenerating synthetic datasets...")
n_datasets = 10  # Use a small number for the example (use 100+ for real applications)
results = kb.add_multiple_synthetic_datasets(n_datasets=n_datasets, random_seed=42)

print(f"Added {len(results)} synthetic datasets to the knowledge base")

# Compile and save the knowledge base
print("\nCompiling and saving knowledge base...")
kb.compile_knowledge_base()
kb.save()

print(f"Knowledge base saved to {os.path.join(output_dir, kb_name)}")

# Define features and target parameters for training
dataset_features = ["n_samples", "n_features", "n_highly_target_corr", "imbalance_ratio"]
target_params = ["params_max_depth", "params_min_samples_split", "params_min_samples_leaf", "params_max_features"]

# Train a ZeroTune model
print("\nTraining ZeroTune model...")
model, score = kb.train_model(
    dataset_features=dataset_features, 
    target_params=target_params,
    n_iter=50  # Using a smaller number for the example
)

print(f"Model trained with MSE: {score}")

# Parameter configuration for DecisionTreeClassifier
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

# Test the trained model on a new dataset
print("\nTesting the trained model on a new dataset...")
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Create a custom predictor with our trained model
model_path = os.path.join(output_dir, kb_name, "zerotune_model.joblib")
predictor = CustomZeroTunePredictor(model_path=model_path, param_config=param_config)

# Get hyperparameter predictions
print("\nPredicting hyperparameters...")
hyperparams = predictor.predict(X, y)
print("Predicted hyperparameters:")
for param, value in hyperparams.items():
    print(f"  - {param}: {value}")

# Evaluate the model with predicted hyperparameters
print("\nEvaluating model with predicted hyperparameters...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = DecisionTreeClassifier(**hyperparams, random_state=42)
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv)
print(f"ROC AUC with ZeroTune hyperparameters: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

# Compare with default hyperparameters
print("\nComparing with default hyperparameters...")
default_model = DecisionTreeClassifier(random_state=42)
default_scores = cross_val_score(default_model, X, y, scoring='roc_auc', cv=cv)
print(f"ROC AUC with default hyperparameters: {np.mean(default_scores):.4f} ± {np.std(default_scores):.4f}")

# Calculate improvement
improvement = (np.mean(scores) - np.mean(default_scores)) / np.mean(default_scores) * 100
print(f"\nImprovement: {improvement:.2f}%")

print("\nNotes:")
print("1. This example uses a small knowledge base for demonstration.")
print("2. For real-world applications, use more datasets (100+) for better results.")
print("3. The trained model is saved at:", model_path) 