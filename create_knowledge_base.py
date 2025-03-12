#!/usr/bin/env python
"""
ZeroTune Knowledge Base Creation Utility

This script builds a knowledge base for ZeroTune with configurable dataset sources:
1. Synthetic datasets - Generated with scikit-learn's make_classification
2. OpenML datasets - Real-world datasets from the OpenML repository
3. Sklearn datasets - Built-in datasets from scikit-learn (like breast cancer)

Usage:
    python create_knowledge_base.py [options]

Options:
    --name NAME             Name for the knowledge base (default: my_knowledge_base)
    --synthetic             Include synthetic datasets (default: False)
    --synthetic-count N     Number of synthetic datasets to create (default: 20)
    --openml                Include OpenML datasets (default: False)
    --openml-ids IDS        Comma-separated list of OpenML dataset IDs to use
    --openml-config FILE    JSON file containing OpenML dataset IDs and names to use
    --sklearn               Include built-in scikit-learn datasets (default: False)
    --hpo-iterations N      Number of HPO iterations for model training (default: 100)
    --no-test               Skip testing the model after training
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from timeit import default_timer as timer
import warnings
from sklearn.preprocessing import LabelEncoder

# Try to import matplotlib, but it's optional
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not found. Performance visualization will be skipped.")

# Import ZeroTune
from zerotune import (
    KnowledgeBase,
    CustomZeroTunePredictor
)

# List of OpenML datasets to include in the knowledge base
# These are classification datasets with different characteristics
DEFAULT_OPENML_DATASETS = [
    31,     # credit-g
    1494,   # qsar-biodeg
    1510,   # wdbc
    40975,  # car
]

# Available scikit-learn datasets
SKLEARN_DATASETS = [
    ("breast_cancer", load_breast_cancer)
]

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create a ZeroTune knowledge base')
    parser.add_argument('--name', type=str, default='my_knowledge_base',
                        help='Name for the knowledge base')
    parser.add_argument('--synthetic', action='store_true',
                        help='Include synthetic datasets')
    parser.add_argument('--synthetic-count', type=int, default=20,
                        help='Number of synthetic datasets to create (default: 20)')
    parser.add_argument('--openml', action='store_true',
                        help='Include OpenML datasets')
    parser.add_argument('--openml-ids', type=str,
                        help='Comma-separated list of OpenML dataset IDs to use')
    parser.add_argument('--openml-config', type=str,
                        help='JSON file containing OpenML dataset IDs and names to use')
    parser.add_argument('--sklearn', action='store_true',
                        help='Include built-in scikit-learn datasets')
    parser.add_argument('--hpo-iterations', type=int, default=100,
                        help='Number of HPO iterations for model training (more iterations = better model, but slower)')
    parser.add_argument('--no-test', action='store_true', dest='skip_test',
                        help='Skip testing the model after training')
    return parser.parse_args()

def create_knowledge_base(name, save_dir="./zerotune_kb"):
    """Create a new knowledge base."""
    print(f"\nCreating knowledge base: {name}")
    kb = KnowledgeBase(name=name, save_dir=save_dir)
    print(f"Knowledge base will be saved to: {os.path.join(save_dir, name)}")
    return kb

def get_openml_dataset_ids(args):
    """Determine which OpenML dataset IDs to use based on arguments.
    
    Args:
        args: Command-line arguments
        
    Returns:
        List of dataset IDs to use, dictionary of dataset info (if available)
    """
    datasets_info = {}
    
    # Check if specific dataset IDs were provided via command line
    if args.openml_ids:
        try:
            dataset_ids = [int(id_str.strip()) for id_str in args.openml_ids.split(',')]
            print(f"Using {len(dataset_ids)} OpenML datasets from command line argument")
            return dataset_ids, datasets_info
        except ValueError:
            print("Warning: Invalid OpenML dataset IDs provided. Format should be comma-separated integers.")
            print("Falling back to default datasets.")
    
    # Check if a config file was provided
    if args.openml_config:
        try:
            with open(args.openml_config, 'r') as f:
                config = json.load(f)
            
            if isinstance(config, list):
                # Simple list of dataset IDs
                dataset_ids = [int(id) for id in config]
                print(f"Using {len(dataset_ids)} OpenML datasets from config file")
                return dataset_ids, datasets_info
            elif isinstance(config, dict):
                # Dictionary with dataset IDs as keys and names/metadata as values
                dataset_ids = []
                for id_str, info in config.items():
                    try:
                        dataset_id = int(id_str)
                        dataset_ids.append(dataset_id)
                        
                        # If info is a string, it's just the name
                        if isinstance(info, str):
                            datasets_info[dataset_id] = info
                        # If info is a dict, it contains metadata
                        elif isinstance(info, dict) and 'name' in info:
                            datasets_info[dataset_id] = info['name']
                    except ValueError:
                        print(f"Warning: Invalid dataset ID in config: {id_str}")
                
                print(f"Using {len(dataset_ids)} OpenML datasets from config file")
                return dataset_ids, datasets_info
            else:
                print("Warning: Invalid OpenML config format. Should be a list or dictionary.")
                print("Falling back to default datasets.")
        except Exception as e:
            print(f"Error reading OpenML config file: {str(e)}")
            print("Falling back to default datasets.")
    
    # Use the default list if no specific IDs were provided
    return DEFAULT_OPENML_DATASETS, datasets_info

def add_openml_datasets(kb, dataset_ids=None, provided_datasets_info=None):
    """Add OpenML datasets to the knowledge base.
    
    Args:
        kb: KnowledgeBase instance
        dataset_ids: Optional list of specific dataset IDs to use
        provided_datasets_info: Optional dictionary mapping dataset IDs to names
    """
    # Determine which datasets to use
    if dataset_ids is None:
        dataset_ids = DEFAULT_OPENML_DATASETS
    
    print(f"\nAdding {len(dataset_ids)} OpenML datasets...")
    
    # Get OpenML dataset info to display names alongside IDs
    datasets_info = provided_datasets_info or {}
    
    if not provided_datasets_info:
        try:
            import openml
            for dataset_id in dataset_ids:
                if dataset_id not in datasets_info:
                    try:
                        # Use basic get_dataset without extra parameters for compatibility
                        dataset = openml.datasets.get_dataset(dataset_id, download_data=False)
                        datasets_info[dataset_id] = dataset.name
                    except:
                        datasets_info[dataset_id] = f"Dataset-{dataset_id}"
        except ImportError:
            print("OpenML package not found. Will use generic dataset names.")
            datasets_info = {dataset_id: f"Dataset-{dataset_id}" for dataset_id in dataset_ids}
    
    # Keep track of successful and failed datasets
    successful = []
    failed = []
    
    # Add datasets with progress tracking
    for i, dataset_id in enumerate(dataset_ids):
        dataset_name = datasets_info.get(dataset_id, f"Dataset-{dataset_id}")
        print(f"  [{i+1}/{len(dataset_ids)}] Adding {dataset_name} (ID: {dataset_id})...")
        
        try:
            start = timer()
            try:
                # Try to import here in case not installed
                import openml
                import pandas as pd
                import numpy as np
                
                # Download the dataset (without dataset_format parameter)
                dataset = openml.datasets.get_dataset(dataset_id, download_data=True)
                
                # Get the default target attribute
                target_attribute = dataset.default_target_attribute
                print(f"    Target attribute: {target_attribute}")
                
                # Get the data - IMPORTANT: must specify target directly
                X, y, categorical_indicator, attribute_names = dataset.get_data(target=target_attribute)
                
                # Convert to pandas DataFrame/Series
                if not isinstance(X, pd.DataFrame):
                    if attribute_names is not None:
                        X = pd.DataFrame(X, columns=attribute_names)
                    else:
                        X = pd.DataFrame(X)
                
                # Ensure y is a pandas Series
                if not isinstance(y, pd.Series):
                    y = pd.Series(y, name=target_attribute)
                
                # Handle categorical target values by converting to integers
                if pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y) or pd.api.types.is_string_dtype(y):
                    print(f"    Converting categorical target to integers...")
                    print(f"    Original target values (sample): {y.head().tolist()}")
                    
                    # Convert categorical labels to integers
                    label_encoder = LabelEncoder()
                    encoded_values = label_encoder.fit_transform(y)
                    y = pd.Series(encoded_values, name=target_attribute)
                    
                    # Show the mapping
                    mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
                    print(f"    Mapping: {mapping}")
                    print(f"    Converted target values (sample): {y.head().tolist()}")
                
                # Add to knowledge base
                dataset_info = kb.add_dataset(X, y, dataset_name=dataset_name)
            except ImportError:
                print("  ✗ OpenML package not found. Please install with 'pip install openml'")
                raise
            except Exception as e:
                print(f"  ✗ Error fetching dataset: {str(e)}")
                raise
                
            end = timer()
            
            print(f"  ✓ Added {dataset_name} ({end-start:.2f}s)")
            print(f"    Features: {dataset_info['n_features']}, Samples: {dataset_info['n_samples']}")
            successful.append((dataset_id, dataset_name))
        except Exception as e:
            print(f"  ✗ Failed to add {dataset_name}: {str(e)}")
            failed.append((dataset_id, dataset_name))
    
    # Print summary
    print(f"\nAdded {len(successful)} out of {len(dataset_ids)} OpenML datasets")
    if failed:
        print(f"The following datasets failed and were skipped:")
        for dataset_id, dataset_name in failed:
            print(f"  - {dataset_name} (ID: {dataset_id})")
    
    return kb

def add_synthetic_datasets(kb, n_datasets=20):
    """Add synthetic datasets to the knowledge base."""
    print(f"\nAdding {n_datasets} synthetic datasets...")
    
    start = timer()
    results = kb.add_multiple_synthetic_datasets(n_datasets=n_datasets, random_seed=42, min_informative=3)
    end = timer()
    
    print(f"✓ Added {len(results)} synthetic datasets ({end-start:.2f}s)")
    
    # Print summary statistics of the synthetic datasets
    n_features = [result['n_features'] for result in results]
    n_samples = [result['n_samples'] for result in results]
    
    print(f"  Features: min={min(n_features)}, max={max(n_features)}, avg={np.mean(n_features):.1f}")
    print(f"  Samples: min={min(n_samples)}, max={max(n_samples)}, avg={np.mean(n_samples):.1f}")
    
    # Make sure all required columns are present in the knowledge base by doing extra HPO
    try:
        print("\nRunning additional HPO to ensure all required hyperparameters are present...")
        for i, result in enumerate(results):
            dataset_id = result.get('dataset_id')
            dataset_name = result.get('name', f'synthetic_{dataset_id}')
            
            # Load the dataset
            dataset_dir = os.path.join(kb.kb_dir, 'datasets', dataset_name)
            X = pd.read_csv(os.path.join(dataset_dir, 'X.csv'))
            y = pd.read_csv(os.path.join(dataset_dir, 'y.csv'))
            y = y.iloc[:, 0]  # Get the first column from y DataFrame
            
            if i == 0:  # Only do this for the first dataset to save time
                # Set up parameter configuration
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
                
                # Calculate dataset meta-parameters
                meta_params = ["n_samples", "n_features", "n_highly_target_corr", "imbalance_ratio"]
                
                # Run HPO with more trials to ensure we get all hyperparameters
                from zerotune import optuna_hpo
                hpo_results = optuna_hpo(
                    X, y,
                    meta_params=meta_params,
                    param_config=param_config,
                    n_trials=30,  # Use more trials for better exploration
                    n_seeds=2,    # Use multiple seeds
                    model_name="DecisionTreeClassifier4Param"
                )
                
                print(f"  Extra HPO completed for {dataset_name}")
    except Exception as e:
        print(f"Warning: Extra HPO step failed, but we'll continue anyway: {str(e)}")
    
    return kb

def add_sklearn_datasets(kb):
    """Add scikit-learn datasets to the knowledge base."""
    print("\nAdding scikit-learn datasets...")
    
    for dataset_name, dataset_loader in SKLEARN_DATASETS:
        try:
            print(f"  Adding {dataset_name}...")
            
            # Load the dataset
            data = dataset_loader()
            X = pd.DataFrame(data.data, columns=data.feature_names if hasattr(data, 'feature_names') else None)
            y = pd.Series(data.target)
            
            # Add the dataset to the knowledge base
            start = timer()
            dataset_info = kb.add_dataset(X, y, dataset_name=dataset_name)
            end = timer()
            
            print(f"  ✓ Added {dataset_name} ({end-start:.2f}s)")
            print(f"    Features: {dataset_info['n_features']}, Samples: {dataset_info['n_samples']}")
            
        except Exception as e:
            print(f"  ✗ Failed to add {dataset_name}: {str(e)}")
    
    return kb

def compile_and_save_kb(kb):
    """Compile and save the knowledge base."""
    print("\nCompiling knowledge base...")
    start = timer()
    kb.compile_knowledge_base()
    end = timer()
    print(f"✓ Knowledge base compiled ({end-start:.2f}s)")
    
    # Ensure all required columns are present for model training
    required_columns = ["params_max_depth", "params_min_samples_split", 
                        "params_min_samples_leaf", "params_max_features"]
    
    if kb.kb is not None:
        missing_columns = [col for col in required_columns if col not in kb.kb.columns]
        if missing_columns:
            print(f"Adding missing columns to knowledge base: {missing_columns}")
            for col in missing_columns:
                if col == "params_max_depth":
                    kb.kb[col] = 10  # Default value
                elif col == "params_min_samples_split":
                    kb.kb[col] = 2   # Default value
                elif col == "params_min_samples_leaf":
                    kb.kb[col] = 1   # Default value
                elif col == "params_max_features":
                    kb.kb[col] = 0.8  # Default value 
    
    print("\nSaving knowledge base...")
    start = timer()
    kb.save()
    end = timer()
    print(f"✓ Knowledge base saved ({end-start:.2f}s)")
    
    return kb

def train_model(kb, n_iter=100):
    """Train a ZeroTune model using the knowledge base.
    
    Args:
        kb: KnowledgeBase instance
        n_iter: Number of hyperparameter optimization iterations (more = better but slower)
    
    Returns:
        Tuple of (trained model, feature columns, target columns)
    """
    print(f"\nTraining ZeroTune model with {n_iter} HPO iterations...")
    
    # Check if the knowledge base is empty
    if kb.kb is None or len(kb.kb) == 0:
        raise ValueError("Knowledge base is empty. No datasets were successfully added.")
    
    # Load the knowledge base
    dataset_features = ["n_samples", "n_features", "n_highly_target_corr", "imbalance_ratio"]
    target_params = ["params_max_depth", "params_min_samples_split", "params_min_samples_leaf", "params_max_features"]
    
    # Check if the knowledge base has the required columns
    missing_columns = []
    for column in dataset_features + target_params:
        if column not in kb.kb.columns:
            missing_columns.append(column)
    
    if missing_columns:
        print("Knowledge base is missing required columns:")
        for col in missing_columns:
            print(f"  - {col}")
        print("\nAvailable columns:")
        for col in kb.kb.columns:
            print(f"  - {col}")
        raise ValueError(f"Knowledge base has missing columns. At least one dataset must be successfully added.")
    
    # Check if there are enough datasets for training
    if len(kb.kb) < 2:
        raise ValueError(f"Knowledge base only has {len(kb.kb)} datasets. At least 2 are required for training.")
    
    start = timer()
    
    # Train the model
    model, score = kb.train_model(
        dataset_features=dataset_features,
        target_params=target_params,
        n_iter=n_iter
    )
    
    end = timer()
    print(f"Model training completed in {end-start:.2f} seconds")
    print(f"Model score: {score:.4f}")
    
    return model, dataset_features, target_params

def test_model(kb_name, save_dir="./zerotune_kb"):
    """Test the trained model on a new dataset."""
    print("\nTesting model on a new dataset...")
    
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
    
    # Path to the saved model
    model_path = os.path.join(save_dir, kb_name, "models", "zerotune_model.joblib")
    
    # Create a custom predictor with the trained model
    try:
        predictor = CustomZeroTunePredictor(
            model_path=model_path,
            param_config=param_config
        )
        print(f"✓ Loaded model from {model_path}")
    except Exception as e:
        print(f"✗ Failed to load model: {str(e)}")
        return
    
    # Generate a test dataset
    print("\nGenerating test dataset...")
    X, y = make_classification(
        n_samples=2000,
        n_features=30,
        n_informative=15, 
        n_redundant=10,
        n_classes=2,
        class_sep=0.8,
        random_state=100  # Different seed than training
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y = pd.Series(y)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=100
    )
    
    # Predict hyperparameters
    print("\nPredicting hyperparameters...")
    hyperparams = predictor.predict(X, y)
    print("Predicted hyperparameters:")
    for param, value in hyperparams.items():
        print(f"  - {param}: {value}")
    
    # Compare with default hyperparameters
    print("\nComparing with default hyperparameters...")
    
    # Create models with predicted and default hyperparameters
    predicted_model = DecisionTreeClassifier(**hyperparams, random_state=42)
    default_model = DecisionTreeClassifier(random_state=42)
    
    # Train and evaluate with cross-validation
    cv = 5
    scoring = 'roc_auc'  # Use ROC AUC as a more robust metric
    predicted_scores = cross_val_score(predicted_model, X, y, cv=cv, scoring=scoring)
    default_scores = cross_val_score(default_model, X, y, cv=cv, scoring=scoring)
    
    # Calculate mean scores
    predicted_mean = np.mean(predicted_scores)
    default_mean = np.mean(default_scores)
    
    print(f"{scoring.upper()} with ZeroTune hyperparameters: {predicted_mean:.4f} ± {np.std(predicted_scores):.4f}")
    print(f"{scoring.upper()} with default hyperparameters: {default_mean:.4f} ± {np.std(default_scores):.4f}")
    
    # Calculate improvement
    improvement = (predicted_mean - default_mean) / default_mean * 100
    print(f"\nImprovement: {improvement:.2f}%")
    
    # Plot comparison if matplotlib is available
    if HAS_MATPLOTLIB:
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.join(save_dir, kb_name), exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        plt.bar(['Default', 'ZeroTune'], [default_mean, predicted_mean], yerr=[np.std(default_scores), np.std(predicted_scores)])
        plt.ylim(0.5, 1.0)
        plt.ylabel(scoring.upper())
        plt.title('Model Performance Comparison')
        plt.savefig(os.path.join(save_dir, kb_name, 'performance_comparison.png'))
        print(f"\nPerformance comparison chart saved to {os.path.join(save_dir, kb_name, 'performance_comparison.png')}")
    else:
        print("\nNote: Performance visualization skipped because matplotlib is not available.")
    
    return hyperparams, predicted_mean, default_mean

def main():
    """Main function to run the entire process."""
    # Filter some common warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
    
    print("=" * 80)
    print("ZeroTune Knowledge Base Creation Utility")
    print("=" * 80)
    
    # Parse command line arguments
    args = parse_args()
    
    # Check if at least one dataset source is enabled
    if not args.synthetic and not args.openml and not args.openml_ids and not args.openml_config and not args.sklearn:
        print("ERROR: You must enable at least one dataset source:")
        print("  - Synthetic datasets (--synthetic)")
        print("  - OpenML datasets (--openml, --openml-ids, or --openml-config)")
        print("  - Scikit-learn datasets (--sklearn)")
        return
    
    # Display configuration
    print("Configuration:")
    print(f"  Knowledge base name: {args.name}")
    print(f"  Synthetic datasets: {'Enabled (' + str(args.synthetic_count) + ' datasets)' if args.synthetic else 'Disabled'}")
    
    # Determine OpenML status for display
    openml_status = "Disabled"
    if args.openml or args.openml_ids or args.openml_config:
        if args.openml_config:
            openml_status = f"Enabled (from config file: {args.openml_config})"
        elif args.openml_ids:
            openml_status = f"Enabled (custom IDs: {args.openml_ids})"
        else:
            openml_status = f"Enabled (Default list - {len(DEFAULT_OPENML_DATASETS)} datasets)"
    
    print(f"  OpenML datasets: {openml_status}")
    print(f"  Scikit-learn datasets: {'Enabled' if args.sklearn else 'Disabled'}")
    print(f"  HPO training iterations: {args.hpo_iterations}")
    print(f"  Test after training: {'No' if args.skip_test else 'Yes'}")
    print("=" * 80)
    
    # Track overall time
    overall_start = timer()
    
    # Create the knowledge base
    kb = create_knowledge_base(args.name)
    
    # Count successful datasets
    dataset_count = 0
    
    # Add sklearn datasets if enabled
    if args.sklearn:
        kb = add_sklearn_datasets(kb)
        # Update dataset count (approximate since we don't track this precisely)
        dataset_count += len(SKLEARN_DATASETS)
    
    # Add OpenML datasets if enabled
    if args.openml or args.openml_ids or args.openml_config:
        # Get the list of dataset IDs to use
        dataset_ids, datasets_info = get_openml_dataset_ids(args)
        kb = add_openml_datasets(kb, 
                                dataset_ids=dataset_ids,
                                provided_datasets_info=datasets_info)
        # Update dataset count
        # We're just guessing here since we don't track the actual count
        if kb.kb is not None and hasattr(kb.kb, 'shape'):
            dataset_count = len(kb.kb)
    
    # Add synthetic datasets if enabled or if we don't have any datasets yet
    if args.synthetic or dataset_count == 0:
        # If we're adding synthetic datasets because no other datasets were added,
        # print a message to explain what's happening
        if dataset_count == 0 and not args.synthetic:
            print("\nNo datasets were successfully added from OpenML. Adding 10 synthetic datasets as a fallback.")
            synthetic_count = 10  # Use 10 as default fallback
            kb = add_synthetic_datasets(kb, synthetic_count)
            dataset_count += synthetic_count
        else:
            kb = add_synthetic_datasets(kb, args.synthetic_count)
            dataset_count += args.synthetic_count
    
    # Compile and save the knowledge base
    kb = compile_and_save_kb(kb)
    
    # Check if we have enough data before training
    if dataset_count == 0:
        print("\nERROR: No datasets were successfully added to the knowledge base.")
        print("Cannot train a model without data. Please try again with different dataset sources.")
        return
    
    try:
        # Train a model
        model, features, targets = train_model(kb, n_iter=args.hpo_iterations)
        
        # Test the model if not skipped
        if not args.skip_test:
            test_model(args.name)
    except ValueError as e:
        print(f"\nERROR in model training: {str(e)}")
        print("The knowledge base may not have enough data or the right columns for training.")
        print("Try adding more datasets or different dataset sources.")
    except Exception as e:
        print(f"\nERROR: {str(e)}")
    
    # Print overall time
    overall_end = timer()
    total_minutes = (overall_end - overall_start) / 60
    print(f"\nTotal time: {total_minutes:.2f} minutes")
    print("\nDone!")
    
    # Print info about the knowledge base location
    print("\nYour knowledge base is ready to use!")
    print(f"Location: ./zerotune_kb/{args.name}/")
    print(f"Trained model: ./zerotune_kb/{args.name}/models/zerotune_model.joblib")
    
    # Save dataset configuration for reproducibility
    config_path = os.path.join("./zerotune_kb", args.name, "kb_config.json")
    try:
        # Create a configuration summary
        config = {
            "name": args.name,
            "synthetic_enabled": args.synthetic,
            "synthetic_count": args.synthetic_count if args.synthetic else 0,
            "openml_enabled": bool(args.openml or args.openml_ids or args.openml_config),
            "sklearn_datasets": args.sklearn,
            "hpo_iterations": args.hpo_iterations,
            "created_on": pd.Timestamp.now().isoformat()
        }
        
        # Include OpenML dataset IDs if they were used
        if args.openml or args.openml_ids or args.openml_config:
            dataset_ids, _ = get_openml_dataset_ids(args)
            config["openml_dataset_ids"] = dataset_ids
        
        # Save the configuration
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Configuration saved to {config_path}")
    except Exception as e:
        print(f"Note: Could not save configuration: {str(e)}")
    
    # Print usage instructions
    print("\nTo use this knowledge base and model in your code:")
    print("```python")
    print("from zerotune import CustomZeroTunePredictor")
    print()
    print("# Define parameter configuration")
    print("param_config = {")
    print('    "max_depth": {')
    print('        "percentage_splits": [0.25, 0.5, 0.7, 0.8, 0.9, 0.999],')
    print('        "param_type": "int",')
    print('        "dependency": "n_samples"')
    print("    },")
    print('    "min_samples_split": {')
    print('        "percentage_splits": [0.005, 0.01, 0.02, 0.05, 0.1],')
    print('        "param_type": "float"')
    print("    },")
    print('    "min_samples_leaf": {')
    print('        "percentage_splits": [0.005, 0.01, 0.02, 0.05, 0.1],')
    print('        "param_type": "float"')
    print("    },")
    print('    "max_features": {')
    print('        "percentage_splits": [0.5, 0.7, 0.8, 0.9, 0.99],')
    print('        "param_type": "float"')
    print("    }")
    print("}")
    print()
    print("# Create a predictor with your custom model")
    print(f'predictor = CustomZeroTunePredictor(model_path="./zerotune_kb/{args.name}/models/zerotune_model.joblib", param_config=param_config)')
    print()
    print("# Predict hyperparameters for a new dataset")
    print("hyperparams = predictor.predict(X, y)")
    print("```")

if __name__ == "__main__":
    main() 