"""
Real-world example demonstrating ZeroTune's hyperparameter optimization capabilities.

This example shows both zero-shot (instant) and iterative (warm-started) HPO modes
using a real-world dataset from OpenML.

Example usage:
    python inference_example.py --dataset 40981 --model xgboost
    python inference_example.py --dataset 40981 --model xgboost --no-optimize
"""

import os
import sys
import argparse
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Add parent directory to path if running directly
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from zerotune import ZeroTune
from zerotune.core.config import CONFIG


def load_openml_dataset(
    dataset_id: int,
    target_column: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.Series, str]:
    """
    Load a dataset from OpenML.
    
    Args:
        dataset_id: OpenML dataset ID
        target_column: Name of the target column (if None, will use the last column)
        
    Returns:
        Tuple of (X, y, dataset_name) where:
        - X is a DataFrame of features
        - y is a Series of targets
        - dataset_name is the name of the dataset
    """
    try:
        import openml
    except ImportError:
        raise ImportError(
            "OpenML package not found. Please install it with: pip install openml"
        )
    
    # Load dataset
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe",
        target=target_column
    )
    
    # Handle categorical features
    for col in X.columns:
        if categorical_indicator[attribute_names.index(col)]:
            X[col] = X[col].astype("category")
    
    # Handle categorical target
    if y.dtype == "object" or y.dtype.name == "category":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index, name=y.name)
    
    return X, y, dataset.name


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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run ZeroTune on a real-world dataset from OpenML"
    )
    parser.add_argument(
        "--dataset",
        type=int,
        required=True,
        help="OpenML dataset ID"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=CONFIG["supported"]["models"],
        default="xgboost",
        help="Model type to use"
    )
    parser.add_argument(
        "--target",
        type=str,
        help="Name of the target column (if None, will use the last column)"
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Use zero-shot mode (no iterative optimization)"
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=10,
        help="Number of optimization iterations (ignored in zero-shot mode)"
    )
    return parser.parse_args()


def main() -> None:
    """Main execution function for the example."""
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Load dataset
    print(f"\nLoading dataset {args.dataset} from OpenML...")
    try:
        X, y, dataset_name = load_openml_dataset(args.dataset, args.target)
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        sys.exit(1)
    
    print(f"Dataset: {dataset_name}")
    print(f"Features shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CONFIG["defaults"]["test_size"],
        random_state=CONFIG["defaults"]["random_state"]
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Initialize ZeroTune
    print(f"\nInitializing ZeroTune with {args.model}...")
    zt = ZeroTune(model_type=args.model, kb_path="output/kb.json")
    
    # Run hyperparameter optimization
    mode = "zero-shot" if args.no_optimize else "iterative"
    print(f"\nRunning {mode} hyperparameter optimization...")
    
    try:
        best_params, best_score, model = zt.optimize(
            X_train, y_train,
            n_iter=args.n_iter,
            verbose=True,
            optimize=not args.no_optimize
        )
        
        # Evaluate on test set
        test_accuracy, test_report = evaluate_model(model, X_test, y_test)
        
        # Print results
        print(f"\nResults for {args.model} ({mode} mode):")
        print(f"Best validation score: {best_score:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        print("\nBest hyperparameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        
        # Save results
        results = {
            "dataset": dataset_name,
            "dataset_id": args.dataset,
            "model_type": args.model,
            "mode": mode,
            "best_params": best_params,
            "best_score": best_score,
            "test_accuracy": test_accuracy,
            "test_report": test_report
        }
        
        # Save to JSON
        import json
        output_file = f"output/{dataset_name}_{args.model}_{mode}_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")
        
        # Save to text file for easy reading
        output_txt = f"output/{dataset_name}_{args.model}_{mode}_results.txt"
        with open(output_txt, "w") as f:
            f.write(f"Dataset: {dataset_name} (ID: {args.dataset})\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Mode: {mode}\n\n")
            f.write(f"Best validation score: {best_score:.4f}\n")
            f.write(f"Test accuracy: {test_accuracy:.4f}\n\n")
            f.write("Best hyperparameters:\n")
            for param, value in best_params.items():
                f.write(f"  {param}: {value}\n")
            f.write("\nClassification report:\n")
            f.write(classification_report(y_test, model.predict(X_test)))
        print(f"Detailed results saved to {output_txt}")
        
    except Exception as e:
        print(f"Error during optimization: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 