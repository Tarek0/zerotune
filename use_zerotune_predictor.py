"""
Example script demonstrating how to use a trained ZeroTune model to predict hyperparameters.

This script:
1. Loads a sample dataset (breast cancer)
2. Sets up a CustomZeroTunePredictor with your trained model
3. Predicts optimal hyperparameters for a Decision Tree
4. Trains a model with those hyperparameters
5. Evaluates the model performance against default hyperparameters
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

# Try to import matplotlib, but handle missing dependencies
try:
    import matplotlib.pyplot as plt
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Visualization will not be available: {str(e)}")
    print("To fix, try: poetry add 'importlib_resources<6.0.0' 'matplotlib<3.10'")
    VISUALIZATION_AVAILABLE = False

# Import ZeroTune
try:
    from zerotune import CustomZeroTunePredictor
except ImportError:
    import sys
    # Add parent directory to path if needed
    sys.path.insert(0, ".")
    from zerotune import CustomZeroTunePredictor

# Load the breast cancer dataset as an example
print("Loading breast cancer dataset...")
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print(f"Dataset shape: {X.shape}")
print(f"Class distribution: {np.bincount(y)}")

# Define parameter configuration for Decision Tree
print("\nSetting up parameter configuration...")
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

# Path to your trained model
model_path = "./zerotune_kb/openml_custom_kb_2/models/zerotune_model.joblib"
if not Path(model_path).exists():
    print(f"Warning: Model file not found at {model_path}")
    print("You may need to update the path to your trained model.")
    model_path = input("Enter the path to your trained model (or press Enter to use the default): ") or model_path

# Create a predictor with your custom model
print(f"\nCreating ZeroTune predictor with model at: {model_path}")
try:
    predictor = CustomZeroTunePredictor(model_path=model_path, param_config=param_config)
    
    # Predict hyperparameters for the dataset
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
    
    # Visualize the results with a bar chart
    if VISUALIZATION_AVAILABLE:
        try:
            plt.figure(figsize=(10, 6))
            bars = plt.bar(['Default', 'ZeroTune'], 
                         [np.mean(default_scores), np.mean(scores)],
                         yerr=[np.std(default_scores), np.std(scores)],
                         capsize=10, alpha=0.7, color=['lightgray', 'green'])
            
            # Add labels and title
            plt.ylabel('ROC AUC Score')
            plt.title('Performance Comparison: Default vs ZeroTune Hyperparameters')
            plt.ylim(0.9, 1.0)  # Adjust as needed based on your scores
            
            # Add value annotations
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                        f'{height:.4f}', ha='center', va='bottom')
            
            # Add improvement annotation
            plt.annotate(f'+{improvement:.2f}%', 
                        xy=(1, np.mean(scores)), 
                        xytext=(1.2, np.mean(scores) + 0.02),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7),
                        fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('zerotune_performance_comparison.png')
            plt.show()
            print("Performance comparison plot saved as 'zerotune_performance_comparison.png'")
        except Exception as e:
            print(f"Could not create visualization: {e}")
    else:
        print("\nVisualization skipped due to missing dependencies.")
        print("Performance Summary:")
        print(f"  Default:  {np.mean(default_scores):.4f} ± {np.std(default_scores):.4f}")
        print(f"  ZeroTune: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        print(f"  Improvement: {improvement:.2f}%")
    
except Exception as e:
    print(f"Error: {e}")
    print("\nTroubleshooting tips:")
    print("1. Make sure the model file exists at the specified path")
    print("2. Check that you have all required dependencies installed")
    print("3. Ensure you're using Python 3.8 or newer")
    print("4. Try running 'poetry install' to set up dependencies") 