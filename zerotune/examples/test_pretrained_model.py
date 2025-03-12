"""
Test script for the pretrained Decision Tree Classifier model.

This script:
1. Loads the pretrained model
2. Applies it to a sample dataset
3. Validates that the predictions make sense
4. Uses the predictions to train a Decision Tree Classifier
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
import joblib
from pathlib import Path

# Import ZeroTune
import sys
sys.path.append("../../")  # Add parent directory to path
from zerotune import (
    ZeroTunePredictor,
    get_available_models
)

print("Testing Pretrained Decision Tree Classifier Model")
print("=" * 50)

# Check if the model file exists
model_path = Path(__file__).parent.parent / "models" / "decision_tree_classifier.joblib"
print(f"Checking for model at: {model_path}")

if not model_path.exists():
    print(f"ERROR: Model file not found at {model_path}")
    print("Please ensure you've added the pretrained model to the models directory.")
    sys.exit(1)

print(f"✓ Model file found")

# Try loading the model directly with joblib to check format
try:
    model_data = joblib.load(model_path)
    print("Model contents:")
    print(f"- Keys: {list(model_data.keys())}")
    if 'model' in model_data:
        print(f"- Model type: {type(model_data['model'])}")
    if 'dataset_features' in model_data:
        print(f"- Dataset features: {model_data['dataset_features']}")
    if 'target_params' in model_data:
        print(f"- Target parameters: {model_data['target_params']}")
    print("✓ Model format is valid")
except Exception as e:
    print(f"ERROR loading model: {str(e)}")
    print("The model file may be corrupted or in an unexpected format.")
    sys.exit(1)

# Now try loading with ZeroTunePredictor
print("\nTesting model through ZeroTunePredictor...")
try:
    # List available models
    available_models = get_available_models()
    print(f"Available models: {available_models}")
    
    # Create predictor
    predictor = ZeroTunePredictor(model_name="decision_tree")
    print("✓ Successfully created ZeroTunePredictor")
    
    # Get model info
    model_info = predictor.get_model_info()
    print("\nModel info:")
    for key, value in model_info.items():
        print(f"- {key}: {value}")
except Exception as e:
    print(f"ERROR with ZeroTunePredictor: {str(e)}")
    sys.exit(1)

# Test prediction on a real dataset
print("\nTesting predictions on breast cancer dataset...")
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print(f"Dataset shape: {X.shape}")

try:
    # Predict hyperparameters
    hyperparams = predictor.predict(X, y)
    print("\nPredicted hyperparameters:")
    for param, value in hyperparams.items():
        print(f"- {param}: {value}")
    
    # Validate that hyperparameters are reasonable
    expected_params = ['max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features']
    for param in expected_params:
        if param not in hyperparams:
            print(f"WARNING: Expected parameter '{param}' not found in predictions")
    
    # Test using the hyperparameters
    print("\nEvaluating Decision Tree with predicted hyperparameters...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = DecisionTreeClassifier(**hyperparams, random_state=42)
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv)
    
    print(f"ROC AUC with ZeroTune hyperparameters: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    # Compare with default hyperparameters
    default_model = DecisionTreeClassifier(random_state=42)
    default_scores = cross_val_score(default_model, X, y, scoring='roc_auc', cv=cv)
    print(f"ROC AUC with default hyperparameters: {np.mean(default_scores):.4f} ± {np.std(default_scores):.4f}")
    
    # Calculate improvement
    improvement = (np.mean(scores) - np.mean(default_scores)) / np.mean(default_scores) * 100
    print(f"\nImprovement: {improvement:.2f}%")
    
    if improvement > 0:
        print("✓ Pretrained model successfully improved performance!")
    else:
        print("WARNING: Pretrained model did not improve performance over defaults")
        
except Exception as e:
    print(f"ERROR during prediction: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 50)
print("Pretrained model test completed successfully!") 