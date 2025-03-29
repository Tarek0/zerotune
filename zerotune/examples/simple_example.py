"""
Simple example script demonstrating ZeroTune functionality.

This example:
1. Creates a synthetic dataset
2. Initializes ZeroTune with a specific model type
3. Optimizes hyperparameters
4. Evaluates the model with optimized hyperparameters

You can use this as a template for your own experiments.
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

# Import ZeroTune
from zerotune import ZeroTune
from zerotune.core import CONFIG  # Import the config module

# Create output directory
os.makedirs("output", exist_ok=True)

# Generate a synthetic dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    flip_y=0.05,
    random_state=CONFIG["defaults"]["random_state"]  # Use config random state
)

# Convert to pandas DataFrame/Series for better handling
X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
y = pd.Series(y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=CONFIG["defaults"]["test_size"], random_state=CONFIG["defaults"]["random_state"]  # Use config values
)

print(f"Dataset shape: {X.shape}")
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Try different model types
model_types = CONFIG["supported"]["models"]  # Use models from config

results = {}

for model_type in model_types:
    print(f"\n\n=== Optimizing {model_type} ===\n")
    
    # Initialize ZeroTune with the specified model type
    zt = ZeroTune(model_type=model_type, kb_path="output/kb.json")
    
    # Optimize hyperparameters
    print(f"Optimizing hyperparameters for {model_type}...")
    best_params, best_score, model = zt.optimize(
        X_train, y_train, n_iter=5, verbose=True  # Reduce iterations for example
    )
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    # Store results
    results[model_type] = {
        "best_params": best_params,
        "best_score": best_score,
        "test_accuracy": test_accuracy,
        "model": model
    }
    
    # Print results
    print(f"\nResults for {model_type}:")
    print(f"Best validation score: {best_score:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print("\nBest hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Print classification report
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

# Compare results
print("\n\n=== Comparison of Model Types ===\n")
print(f"{'Model Type':<15} {'Validation Score':<20} {'Test Accuracy':<15}")
print("-" * 50)

for model_type, result in results.items():
    print(f"{model_type:<15} {result['best_score']:<20.4f} {result['test_accuracy']:<15.4f}")

# Find the best model
best_model_type = max(results.items(), key=lambda x: x[1]['test_accuracy'])[0]
print(f"\nBest model type: {best_model_type}")
print(f"Best model test accuracy: {results[best_model_type]['test_accuracy']:.4f}") 