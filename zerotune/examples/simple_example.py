"""
Simple example demonstrating ZeroTune's hyperparameter optimization capabilities.

This example shows both zero-shot (instant) and iterative (warm-started) HPO modes.
"""

import os
import sys
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Add parent directory to path if running directly
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from zerotune import ZeroTune
from zerotune.core.config import CONFIG


def create_synthetic_dataset(
    n_samples: int = 1000,
    n_features: int = 20,
    n_informative: int = 10,
    n_redundant: int = 5,
    n_classes: int = 2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create a synthetic classification dataset.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_informative: Number of informative features
        n_redundant: Number of redundant features
        n_classes: Number of classes
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X, y) where X is a DataFrame of features and y is a Series of targets
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        random_state=random_state
    )
    return pd.DataFrame(X), pd.Series(y)


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluate a model on test data.
    
    Args:
        model: Trained model with predict method
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Tuple of (accuracy, classification_report)
    """
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    report = classification_report(y_test, y_pred, output_dict=True)
    return accuracy, report


def main() -> None:
    """Main execution function for the example."""
    # Create output directory
    os.makedirs("output", exist_ok=True)

    # Generate synthetic dataset
    print("Generating synthetic dataset...")
    X, y = create_synthetic_dataset()
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=CONFIG["defaults"]["test_size"], 
        random_state=CONFIG["defaults"]["random_state"]
    )

    print(f"Dataset shape: {X.shape}")
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Try different model types
    model_types: List[str] = CONFIG["supported"]["models"]
    results: Dict[str, Dict[str, Any]] = {}
    zero_shot_results: Dict[str, Dict[str, Any]] = {}

    for model_type in model_types:
        print(f"\n\n=== Testing {model_type} ===")
        
        # Initialize ZeroTune with the specified model type
        zt = ZeroTune(model_type=model_type, kb_path="output/kb.json")
        
        # 1. Zero-shot HPO (instant prediction)
        print(f"\n1. Zero-shot HPO for {model_type}...")
        try:
            best_params_zero, best_score_zero, model_zero = zt.optimize(
                X_train, y_train,
                n_iter=10,  # n_iter is ignored in zero-shot mode
                verbose=True,
                optimize=False  # Use zero-shot mode
            )
            
            # Evaluate zero-shot model
            test_accuracy_zero, test_report_zero = evaluate_model(model_zero, X_test, y_test)
            
            # Store zero-shot results
            zero_shot_results[model_type] = {
                "best_params": best_params_zero,
                "best_score": best_score_zero,
                "test_accuracy": test_accuracy_zero,
                "test_report": test_report_zero,
                "model": model_zero
            }
            
            print(f"\nZero-shot results for {model_type}:")
            print(f"Best validation score: {best_score_zero:.4f}")
            print(f"Test accuracy: {test_accuracy_zero:.4f}")
            print("\nBest hyperparameters:")
            for param, value in best_params_zero.items():
                print(f"  {param}: {value}")
        except Exception as e:
            print(f"Error in zero-shot mode: {str(e)}")
            zero_shot_results[model_type] = None
        
        # 2. Iterative HPO (warm-started)
        print(f"\n2. Iterative HPO for {model_type}...")
        try:
            best_params, best_score, model = zt.optimize(
                X_train, y_train,
                n_iter=10,  # Run 10 iterations
                verbose=True,
                optimize=True  # Use iterative mode (default)
            )
            
            # Evaluate iterative model
            test_accuracy, test_report = evaluate_model(model, X_test, y_test)
            
            # Store iterative results
            results[model_type] = {
                "best_params": best_params,
                "best_score": best_score,
                "test_accuracy": test_accuracy,
                "test_report": test_report,
                "model": model
            }
            
            print(f"\nIterative results for {model_type}:")
            print(f"Best validation score: {best_score:.4f}")
            print(f"Test accuracy: {test_accuracy:.4f}")
            print("\nBest hyperparameters:")
            for param, value in best_params.items():
                print(f"  {param}: {value}")
        except Exception as e:
            print(f"Error in iterative mode: {str(e)}")
            results[model_type] = None

    # Compare results
    print("\n\n=== Comparison of HPO Modes ===\n")
    print(f"{'Model Type':<15} {'Zero-shot Score':<20} {'Iterative Score':<20} {'Improvement':<15}")
    print("-" * 70)

    for model_type in model_types:
        if model_type in zero_shot_results and zero_shot_results[model_type] is not None and \
           model_type in results and results[model_type] is not None:
            zero_score = zero_shot_results[model_type]['test_accuracy']
            iter_score = results[model_type]['test_accuracy']
            improvement = iter_score - zero_score
            print(f"{model_type:<15} {zero_score:<20.4f} {iter_score:<20.4f} {improvement:+.4f}")

    # Find the best model overall
    best_model_type = None
    best_score = -1
    best_mode = None
    
    for model_type in model_types:
        if model_type in results and results[model_type] is not None:
            iter_score = results[model_type]['test_accuracy']
            if iter_score > best_score:
                best_score = iter_score
                best_model_type = model_type
                best_mode = "iterative"
        
        if model_type in zero_shot_results and zero_shot_results[model_type] is not None:
            zero_score = zero_shot_results[model_type]['test_accuracy']
            if zero_score > best_score:
                best_score = zero_score
                best_model_type = model_type
                best_mode = "zero-shot"
    
    if best_model_type:
        print(f"\nBest overall model: {best_model_type} ({best_mode} mode)")
        print(f"Best test accuracy: {best_score:.4f}")


if __name__ == "__main__":
    main() 