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


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for ZeroTune CLI.
    
    Args:
        args: Command line arguments (defaults to sys.argv if None)
        
    Returns:
        int: Exit code
    """
    parser = argparse.ArgumentParser(
        description="ZeroTune: Hyperparameter optimization using meta-learning"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict hyperparameters for a dataset")
    predict_parser.add_argument(
        "--dataset", required=True, 
        help="Dataset to predict for. Can be an OpenML ID, path to CSV file, or 'custom'"
    )
    predict_parser.add_argument(
        "--model", default=CONFIG["defaults"]["model_type"],
        choices=CONFIG["supported"]["models"],
        help=f"Model to predict hyperparameters for (default: {CONFIG['defaults']['model_type']})"
    )
    predict_parser.add_argument(
        "--output-dir", dest="output_dir", default="./output/",
        help="Output directory where models are stored. You can provide a custom path like './output/my_experiment/'."
    )
    predict_parser.add_argument(
        "--n-iter", dest="n_iter", type=int, default=CONFIG["defaults"]["n_iter"],
        help=f"Number of iterations for optimization (default: {CONFIG['defaults']['n_iter']})"
    )
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Build a knowledge base for ZeroTune")
    train_parser.add_argument(
        "--datasets", nargs="+", type=int, required=True,
        help="List of OpenML dataset IDs to use for training"
    )
    train_parser.add_argument(
        "--model", default=CONFIG["defaults"]["model_type"],
        choices=CONFIG["supported"]["models"],
        help=f"Model to train for (default: {CONFIG['defaults']['model_type']})"
    )
    train_parser.add_argument(
        "--n-iter", dest="n_iter", type=int, default=CONFIG["defaults"]["n_iter"],
        help=f"Number of iterations for each dataset (default: {CONFIG['defaults']['n_iter']})"
    )
    train_parser.add_argument(
        "--output-dir", dest="output_dir",
        help="Output directory (defaults to ./output/YYYYMMDD_HHMMSS/). You can provide a custom name like './output/my_experiment/'."
    )
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run a demonstration of ZeroTune")
    demo_parser.add_argument(
        "--dataset-id", dest="dataset_id", type=int, default=40981,
        help="OpenML dataset ID to use for the demo (default: 40981)"
    )
    demo_parser.add_argument(
        "--model", default=CONFIG["defaults"]["model_type"],
        choices=CONFIG["supported"]["models"],
        help=f"Model to optimize in the demo (default: {CONFIG['defaults']['model_type']})"
    )
    
    # Datasets command
    datasets_parser = subparsers.add_parser("datasets", help="List available datasets from the catalog")
    
    # Create a mutually exclusive group for listing vs recommendations
    datasets_display_group = datasets_parser.add_mutually_exclusive_group()
    datasets_display_group.add_argument(
        "--list", action="store_true",
        help="List all available datasets instead of showing recommendations"
    )
    datasets_display_group.add_argument(
        "--default", action="store_true",
        help="Show recommended datasets (this is the default behavior)"
    )
    
    datasets_parser.add_argument(
        "--category", choices=["binary", "multi-class", "all"],
        default="all", help="Filter datasets by category"
    )
    datasets_parser.add_argument(
        "--classes", type=int, help="Filter by number of classes"
    )
    datasets_parser.add_argument(
        "--count", type=int, default=5,
        help="Number of recommended datasets to show"
    )
    
    args = parser.parse_args(args)
    
    # Set experiment ID and paths
    if hasattr(args, "output_dir") and args.output_dir:
        output_root = args.output_dir
        if not output_root.endswith("/"):
            output_root += "/"
        # If only a name is provided without a path, put it in the output directory
        if not os.path.dirname(output_root):
            output_root = f"./output/{output_root}"
    else:
        exp_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = f"./output/{exp_id}/"
    
    os.makedirs(output_root, exist_ok=True)
    
    # Execute the chosen command
    if args.command == "predict":
        try:
            if args.dataset.isdigit():
                # OpenML dataset ID
                dataset_id = int(args.dataset)
                print(f"Fetching OpenML dataset {dataset_id}...")
                df, target_name, dataset_name = fetch_open_ml_data(dataset_id)
                X, y = prepare_data(df, target_name)
                print(f"Dataset: {dataset_name}")
            elif args.dataset == "custom":
                # Example custom dataset
                print("Using example custom dataset...")
                # Generate a simple synthetic dataset
                from sklearn.datasets import make_classification
                X, y = make_classification(
                    n_samples=1000, n_features=20, n_informative=10,
                    n_redundant=5, n_classes=2, random_state=CONFIG["defaults"]["random_state"]
                )
                X = pd.DataFrame(X)
                dataset_name = "custom_dataset"
            else:
                # Path to CSV file
                print(f"Loading dataset from {args.dataset}...")
                df = pd.read_csv(args.dataset)
                # Assume last column is target
                target_name = df.columns[-1]
                X = df.drop(target_name, axis=1)
                y = df[target_name]
                dataset_name = os.path.basename(args.dataset)
            
            # Initialize ZeroTune with model type
            print(f"Initializing ZeroTune with model type: {args.model}")
            zt = ZeroTune(model_type=args.model, kb_path=os.path.join(output_root, "kb.json"))
            
            # Optimize hyperparameters
            print("Optimizing hyperparameters...")
            best_params, best_score, model = zt.optimize(
                X, y, 
                n_iter=args.n_iter, 
                verbose=True
            )
            
            # Display results
            print("\n=== Best Hyperparameters ===")
            for param, value in best_params.items():
                print(f"{param}: {value}")
            print(f"\nScore: {best_score}")
            
            # Save results
            results = {
                "dataset": dataset_name,
                "model": args.model,
                "score": best_score,
                "hyperparameters": best_params,
                "timestamp": datetime.datetime.now().isoformat()
            }
            results_file = os.path.join(output_root, f"{dataset_name}_{args.model}_results.json")
            save_json(results, results_file)
            
            # Also save a readable text file
            txt_file = os.path.join(output_root, f"{dataset_name}_{args.model}_results.txt")
            with open(txt_file, 'w') as f:
                f.write(f"Dataset: {dataset_name}\n")
                f.write(f"Model: {args.model}\n")
                f.write(f"Score: {best_score}\n\n")
                f.write("Hyperparameters:\n")
                for param, value in best_params.items():
                    f.write(f"{param}: {value}\n")
            
            print(f"\nResults saved to {results_file} and {txt_file}")
            return 0
            
        except Exception as e:
            print(f"Error predicting hyperparameters: {str(e)}")
            import traceback
            traceback.print_exc()
            return 1
            
    elif args.command == "train":
        # Build a knowledge base
        try:
            print(f"Building knowledge base for model type: {args.model}")
            
            # Initialize ZeroTune
            kb_path = os.path.join(output_root, "kb.json")
            zt = ZeroTune(model_type=args.model, kb_path=kb_path)
            
            # Build knowledge base
            print(f"Using {len(args.datasets)} datasets with {args.n_iter} iterations each")
            knowledge_base = zt.build_knowledge_base(
                dataset_ids=args.datasets,
                n_iter=args.n_iter,
                verbose=True
            )
            
            print("\nKnowledge base built successfully!")
            print(f"Knowledge base saved to: {kb_path}")
            
            # Print summary
            print(f"\nDatasets in knowledge base: {len(knowledge_base.get('datasets', []))}")
            print(f"Total results: {len(knowledge_base.get('results', []))}")
            
            return 0
        except Exception as e:
            print(f"Error building knowledge base: {str(e)}")
            import traceback
            traceback.print_exc()
            return 1
    
    elif args.command == "demo":
        # Run a demonstration
        try:
            # Use the Australian credit dataset as a demo
            dataset_id = args.dataset_id
            model_type = args.model
            
            print("\n=== ZeroTune Demo ===")
            print(f"Model type: {model_type}")
            print(f"Dataset ID: {dataset_id}")
            print("\nStep 1: Loading dataset...")
            
            # Fetch and prepare the dataset
            data, target_name, dataset_name = fetch_open_ml_data(dataset_id)
            print(f"✓ Dataset loaded: {dataset_name}")
            print(f"  - Target variable: {target_name}")
            
            # Prepare the data for modeling
            X, y = prepare_data(data, target_name)
            # Convert y to pandas Series if it's not already
            if not isinstance(y, pd.Series):
                y = pd.Series(y)
            
            print(f"\nStep 2: Data preparation")
            print(f"✓ Features shape: {X.shape}")
            print(f"✓ Target distribution: {dict(y.value_counts())}")
            
            # Initialize ZeroTune
            print("\nStep 3: Initializing ZeroTune")
            zt = ZeroTune(model_type=model_type)
            print(f"✓ ZeroTune initialized with {model_type}")
            print(f"  - Supported hyperparameters: {', '.join(zt.model_config['param_config'].keys())}")
            
            # Run optimization
            print("\nStep 4: Running hyperparameter optimization")
            print("This step will:")
            print("  1. Extract dataset meta-features")
            print("  2. Find similar datasets in the knowledge base")
            print("  3. Use meta-learning to guide the search")
            print("  4. Optimize hyperparameters (5 iterations for demo)")
            
            best_params, best_score, model = zt.optimize(
                X, y,
                n_iter=5,  # Reduced iterations for demo
                verbose=True
            )
            
            print("\nStep 5: Results")
            print("Best hyperparameters found:")
            for param, value in best_params.items():
                print(f"  - {param}: {value}")
            print(f"\nModel performance:")
            print(f"  - Validation score: {best_score:.4f}")
            
            # Additional model information
            print("\nModel details:")
            print(f"  - Type: {model.__class__.__name__}")
            print(f"  - Features used: {X.shape[1]}")
            print(f"  - Training samples: {X.shape[0]}")
            
            print("\n✓ Demo completed successfully!")
            print("\nNext steps:")
            print("1. Try different model types: --model decision_tree or --model random_forest")
            print("2. Use your own dataset: zerotune predict --dataset path/to/your/data.csv")
            print("3. Build a knowledge base: zerotune train --datasets 31 38 44")
            return 0
        except Exception as e:
            print(f"\nError running demo: {str(e)}")
            import traceback
            traceback.print_exc()
            return 1
    
    elif args.command == "datasets":
        # Display recommended datasets by default or list all datasets if --list flag is used
        if args.list:
            print("Listing available datasets...")
            
            # Get dataset IDs based on category
            category = None if args.category == "all" else args.category
            dataset_ids = get_dataset_ids(category=category, classes=args.classes)
            
            if not dataset_ids:
                print("No datasets found matching the criteria.")
                return 0
            
            # Group datasets by category for better readability
            datasets_by_category = {}
            catalog = load_dataset_catalog()
            
            for cat in catalog.keys():
                datasets_in_cat = []
                for dataset_name, dataset_info in catalog[cat].items():
                    if isinstance(dataset_info, dict) and 'id' in dataset_info:
                        dataset_id = dataset_info['id']
                    else:
                        dataset_id = dataset_info
                    
                    if int(dataset_id) in dataset_ids:
                        datasets_in_cat.append((dataset_name, int(dataset_id)))
                
                if datasets_in_cat:
                    datasets_by_category[cat] = datasets_in_cat
            
            # Print datasets by category
            for cat, datasets in datasets_by_category.items():
                print(f"\n{cat.upper()} DATASETS:")
                print(f"{'Dataset Name':<30} {'ID':<10}")
                print("-" * 40)
                for name, id in datasets:
                    print(f"{name:<30} {id:<10}")
            
            print(f"\nTotal datasets: {len(dataset_ids)}")
        else:
            # Show recommended datasets
            print("Recommended datasets for training:")
            datasets = get_recommended_datasets(n_datasets=args.count)
            
            if not datasets:
                print("No recommended datasets found.")
                return 0
            
            # Fetch and display dataset information
            print(f"{'Dataset ID':<10} {'Dataset Name':<30} {'Samples':<10} {'Features':<10}")
            print("-" * 60)
            
            for dataset_id in datasets:
                try:
                    import openml
                    dataset = openml.datasets.get_dataset(dataset_id)
                    print(f"{dataset_id:<10} {dataset.name:<30} {dataset.qualities['NumberOfInstances']:<10} {dataset.qualities['NumberOfFeatures']:<10}")
                except Exception as e:
                    print(f"{dataset_id:<10} Error retrieving dataset info: {str(e)}")
            
            # Provide instructions for next steps
            print("\nTo build a knowledge base with these datasets:")
            dataset_ids_str = " ".join(map(str, datasets))
            print(f"zerotune train --datasets {dataset_ids_str} --model {CONFIG['defaults']['model_type']}")
        
        return 0
        
    else:
        # No command specified, show help
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 