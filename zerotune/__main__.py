#!/usr/bin/env python
"""
ZeroTune command-line interface for hyperparameter optimization.
"""

import os
import sys
import argparse
import datetime
from typing import List, Optional, Dict, Any, Union
import pandas as pd
import numpy as np

# Add the parent directory to sys.path if running directly
if __name__ == "__main__":
    # This allows the script to run directly
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from the new modular structure
from zerotune import (
    ZeroTune,
    ModelConfigs,
    fetch_open_ml_data,
    prepare_data,
    get_dataset_ids,
    get_recommended_datasets
)
from zerotune.core import CONFIG
from zerotune.core.utils import save_json, load_json


def main() -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="ZeroTune: Zero-shot hyperparameter optimization")
    subparsers = parser.add_subparsers(dest="command", help="Command to run", required=True)
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict hyperparameters for a dataset")
    predict_parser.add_argument("--dataset-id", type=int, help="OpenML dataset ID")
    predict_parser.add_argument("--data-path", type=str, help="Path to custom dataset CSV file")
    predict_parser.add_argument("--target", type=str, help="Target column name for custom dataset")
    predict_parser.add_argument("--model-type", type=str, choices=["decision_tree", "random_forest", "xgboost"], default="random_forest", help="Model type to optimize")
    predict_parser.add_argument("--zero-shot", action="store_true", help="Use zero-shot mode instead of iterative")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model with optimized hyperparameters")
    train_parser.add_argument("--dataset-id", type=int, required=True, help="OpenML dataset ID")
    train_parser.add_argument("--model-type", type=str, choices=["decision_tree", "random_forest", "xgboost"], default="random_forest", help="Model type to optimize")
    train_parser.add_argument("--zero-shot", action="store_true", help="Use zero-shot mode instead of iterative")
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run a demo optimization")
    demo_parser.add_argument("--model-type", type=str, choices=["decision_tree", "random_forest", "xgboost"], default="random_forest", help="Model type to optimize")
    demo_parser.add_argument("--zero-shot", action="store_true", help="Use zero-shot mode instead of iterative")
    
    # Datasets command
    datasets_parser = subparsers.add_parser("datasets", help="List available datasets")
    datasets_parser.add_argument("--category", type=str, choices=["binary", "multiclass", "all"], default="all", help="Dataset category to list")
    
    try:
        args = parser.parse_args()
        
        if args.command == "predict":
            if args.dataset_id is not None:
                return predict_openml(args.dataset_id, args.model_type, args.zero_shot)
            elif args.data_path is not None and args.target is not None:
                return predict_custom(args.data_path, args.target, args.model_type, args.zero_shot)
            else:
                print("Error: Either --dataset-id or both --data-path and --target must be provided")
                return 1
                
        elif args.command == "train":
            return train_model(args.dataset_id, args.model_type, args.zero_shot)
            
        elif args.command == "demo":
            return run_demo(args.model_type, args.zero_shot)
            
        elif args.command == "datasets":
            try:
                from zerotune.core.data_loading import load_dataset_catalog, get_dataset_ids
                print("Listing available datasets...")
                catalog = load_dataset_catalog()
                dataset_ids = get_dataset_ids(category=args.category)
                if dataset_ids:
                    print(f"\nFound {len(dataset_ids)} datasets:")
                    for dataset_id in dataset_ids:
                        print(f"- Dataset ID: {dataset_id}")
                else:
                    print("No datasets found.")
                return 0
            except Exception as e:
                print(f"Error listing datasets: {str(e)}")
                return 1
                
        else:
            print(f"Unknown command: {args.command}")
            return 1
            
    except SystemExit as e:
        return e.code
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

def predict_openml(dataset_id: int, model_type: str, zero_shot: bool) -> int:
    """Predict hyperparameters for an OpenML dataset."""
    try:
        print(f"Fetching OpenML dataset {dataset_id}...")
        df, target_name, dataset_name = fetch_open_ml_data(dataset_id)
        X, y = prepare_data(df, target_name)
        print(f"Dataset: {dataset_name}")
        
        # Initialize ZeroTune
        print(f"Initializing ZeroTune with model type: {model_type}")
        zt = ZeroTune(model_type=model_type)
        
        # Optimize hyperparameters
        if zero_shot:
            print("Predicting hyperparameters using zero-shot mode...")
        else:
            print("Optimizing hyperparameters using iterative mode...")
        best_params, best_score, model = zt.optimize(
            X, y,
            n_iter=10 if not zero_shot else 0,
            verbose=True,
            optimize=not zero_shot
        )
        
        # Display results
        print("\n=== Best Hyperparameters ===")
        for param, value in best_params.items():
            print(f"{param}: {value}")
        print(f"\nScore: {best_score}")
        
        return 0
    except Exception as e:
        print(f"Error predicting hyperparameters: {str(e)}")
        return 1

def predict_custom(data_path: str, target: str, model_type: str, zero_shot: bool) -> int:
    """Predict hyperparameters for a custom dataset."""
    try:
        print(f"Loading dataset from {data_path}...")
        df = pd.read_csv(data_path)
        if target not in df.columns:
            print(f"Error: Target column '{target}' not found in dataset")
            return 1
        
        X = df.drop(target, axis=1)
        y = df[target]
        dataset_name = os.path.basename(data_path)
        print(f"Dataset: {dataset_name}")
        
        # Initialize ZeroTune
        print(f"Initializing ZeroTune with model type: {model_type}")
        zt = ZeroTune(model_type=model_type)
        
        # Optimize hyperparameters
        if zero_shot:
            print("Predicting hyperparameters using zero-shot mode...")
        else:
            print("Optimizing hyperparameters using iterative mode...")
        best_params, best_score, model = zt.optimize(
            X, y,
            n_iter=10 if not zero_shot else 0,
            verbose=True,
            optimize=not zero_shot
        )
        
        # Display results
        print("\n=== Best Hyperparameters ===")
        for param, value in best_params.items():
            print(f"{param}: {value}")
        print(f"\nScore: {best_score}")
        
        return 0
    except Exception as e:
        print(f"Error predicting hyperparameters: {str(e)}")
        return 1

def train_model(dataset_id: int, model_type: str, zero_shot: bool) -> int:
    """Train a model with optimized hyperparameters."""
    try:
        print(f"Fetching OpenML dataset {dataset_id}...")
        df, target_name, dataset_name = fetch_open_ml_data(dataset_id)
        X, y = prepare_data(df, target_name)
        print(f"Dataset: {dataset_name}")
        
        # Initialize ZeroTune
        print(f"Initializing ZeroTune with model type: {model_type}")
        zt = ZeroTune(model_type=model_type)
        
        # Optimize hyperparameters
        if zero_shot:
            print("Predicting hyperparameters using zero-shot mode...")
        else:
            print("Optimizing hyperparameters using iterative mode...")
        best_params, best_score, model = zt.optimize(
            X, y,
            n_iter=10 if not zero_shot else 0,
            verbose=True,
            optimize=not zero_shot
        )
        
        # Train final model
        print("\nTraining final model with best hyperparameters...")
        model.fit(X, y)
        
        # Evaluate model
        y_pred = model.predict(X)
        accuracy = (y_pred == y).mean()
        print(f"\nTraining accuracy: {accuracy:.4f}")
        
        return 0
    except Exception as e:
        print(f"Error training model: {str(e)}")
        return 1

def run_demo(model_type: str, zero_shot: bool) -> int:
    """Run a demo optimization."""
    try:
        # Use a small, well-known dataset for the demo
        dataset_id = 31  # credit-g dataset
        print(f"Running demo with model type: {model_type}")
        print(f"Using dataset ID: {dataset_id}")
        
        return predict_openml(dataset_id, model_type, zero_shot)
    except Exception as e:
        print(f"Error running demo: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 