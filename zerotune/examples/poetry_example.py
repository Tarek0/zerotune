"""
Example demonstrating how Poetry simplifies dependency management in ZeroTune.

This example:
1. Imports dependencies that were automatically installed by Poetry
2. Shows how to access resources within the package
3. Demonstrates a simple ZeroTune workflow
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

# Check for Poetry environment
try:
    import poetry_version
    print("Running in Poetry environment ✓")
except ImportError:
    print("Not running in Poetry environment. Some features may not work properly.")

# Import ZeroTune (no need for sys.path manipulation with Poetry)
from zerotune import (
    calculate_dataset_meta_parameters,
    generate_random_params,
    relative2absolute_dict
)

# Generate synthetic data
from sklearn.datasets import make_classification

print("Generating synthetic dataset...")
X, y = make_classification(
    n_samples=500,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    random_state=42
)

# Convert to pandas DataFrame/Series (as expected by ZeroTune functions)
X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
y_series = pd.Series(y)

# Calculate dataset meta-parameters
print("\nCalculating dataset meta-parameters...")
meta_params = calculate_dataset_meta_parameters(X_df, y_series)

# Show key meta-parameters
print("\nKey meta-parameters:")
key_params = ["n_samples", "n_features", "n_highly_target_corr", "imbalance_ratio"]
for param in key_params:
    print(f"  {param}: {meta_params[param]}")

# Define parameter configuration
param_config = {
    "max_depth": {
        "percentage_splits": [0.25, 0.5, 0.7, 0.8, 0.9], 
        "param_type": "int", 
        "dependency": "n_samples"
    },
    "min_samples_split": {
        "percentage_splits": [0.005, 0.01, 0.02, 0.05], 
        "param_type": "float"
    }
}

# Generate random hyperparameters
print("\nGenerating random hyperparameters...")
random_params = generate_random_params(param_config, random_seed=42)
print("Random relative parameters:", random_params)

# Convert to absolute parameters
absolute_params = relative2absolute_dict(param_config, meta_params, random_params)
print("Converted absolute parameters:", absolute_params)

# Demonstrate Poetry's access to package resources
models_dir = Path(__file__).parent.parent / "models"
print(f"\nPackage resources directory at: {models_dir}")
print(f"Directory exists: {models_dir.exists()}")

print("\nWith Poetry, you can:")
print("- Easily manage dependencies")
print("- Access package resources")
print("- Have reproducible builds with the lock file")
print("- Separate development and production dependencies")
print("- Have a standardized development workflow") 