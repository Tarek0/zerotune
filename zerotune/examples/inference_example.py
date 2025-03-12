"""
Simple example demonstrating ZeroTune inference with pre-trained models.

This example:
1. Loads the breast cancer dataset
2. Gets predictions from a pre-trained ZeroTune model
3. Creates and evaluates a model with the predicted hyperparameters
4. Compares against default hyperparameters
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

# Import ZeroTune - two approaches provided:
# Approach 1: Use proper package import if installed
try:
    from zerotune import (
        ZeroTunePredictor,
        get_available_models
    )
    print("Using installed ZeroTune package")
# Approach 2: Use relative import for development
except ImportError:
    import sys
    # Add the parent directory of the current file to the path
    module_path = str(Path(__file__).resolve().parent.parent.parent)
    if module_path not in sys.path:
        sys.path.insert(0, module_path)
    from zerotune import (
        ZeroTunePredictor,
        get_available_models
    )
    print("Using development ZeroTune import")

# Load the breast cancer dataset
print("Loading breast cancer dataset...")
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print(f"Dataset shape: {X.shape}")

# List available pre-trained models
print("\nAvailable pre-trained models:")
available_models = get_available_models()
for name, description in available_models.items():
    print(f"  - {name}: {description}")

# Create ZeroTune predictor
try:
    print("\nCreating ZeroTune predictor...")
    
    # First check if we need to rename the model file to match expected name
    model_dir = Path(__file__).resolve().parent.parent / "models"
    expected_file = model_dir / "decision_tree_classifier.joblib"
    actual_file = model_dir / "ZeroTune_DecisionTreeClassifier4Param.joblib"
    
    if not expected_file.exists() and actual_file.exists():
        print(f"Note: Found model file with different name. Using {actual_file.name}")
        # Create a temporary symlink or copy in a real application
        # For this example, we'll just inform the user
    
    predictor = ZeroTunePredictor(model_name="decision_tree")
    
    # Get model info
    model_info = predictor.get_model_info()
    print("Model info:")
    print(f"  - Name: {model_info['name']}")
    print(f"  - Description: {model_info['description']}")
    print(f"  - Dataset features: {model_info['dataset_features']}")
    print(f"  - Target params: {model_info['target_params']}")
    
    # Predict hyperparameters
    print("\nPredicting hyperparameters...")
    hyperparams = predictor.predict(X, y)
    print("Predicted hyperparameters:")
    for param, value in hyperparams.items():
        print(f"  - {param}: {value}")
    
    # Evaluate model with predicted hyperparameters
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

except Exception as e:
    print(f"\nError: {e}")
    print("\nNote: This example requires pre-trained models to be available.")
    print("If the model file wasn't found, ensure you have the following file in place:")
    print(f"  - {expected_file}")
    print("\nAlternatively, you need to update the model name in predictors.py to match your actual file:")
    print(f"  - Update model_file: 'decision_tree_classifier.joblib' to '{actual_file.name}'")
    print("\nIf you don't have pre-trained models, you can train your own using the KnowledgeBase class.")
    print("See the knowledge_base_example.py for instructions.") 