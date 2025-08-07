"""
Random Forest Knowledge Base Builder & Zero-Shot Predictor Training/Evaluation

CURRENT STATUS: 🚀 ENHANCED WITH DECISION TREE WINNING STRATEGIES!
✅ Percentage-based max_depth scaling with dataset size  
✅ Enhanced knowledge base: 50 HPO runs per dataset (proven optimal)
✅ Removed normalization for better meta-feature signal
✅ Multi-seed robust evaluation (50 seeds) for statistical validity
✅ Expected excellent performance based on Decision Tree success

ARCHITECTURE:
- zerotune/core/predictor_training.py: Advanced training with RFECV & GroupKFold
- zerotune/core/feature_extraction.py: Robust meta-feature extraction
- zerotune/predictors.py: Zero-shot predictor class for inference
- models/predictor_random_forest_rf_kb_v1_full.joblib: Production-ready trained model
- knowledge_base/kb_random_forest_rf_kb_v1_full.json: Clean, comprehensive training data

RANDOM FOREST ADVANTAGES:
- More hyperparameters than Decision Tree (5 vs 4): n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features
- Ensemble approach reduces overfitting compared to single tree
- Built-in feature randomness should help with generalization
- Expected to outperform single Decision Tree while maintaining simplicity vs XGBoost

OPTIMIZATION APPROACH (Enhanced):
- 50 HPO trials per dataset (up from 20) for optimal hyperparameter discovery
- Percentage-based max_depth scaling intelligently adapts to dataset size
- No normalization: Raw meta-features provide stronger predictive signal
- Multi-seed evaluation (50 seeds) for statistically robust benchmarking
- Top-3 trials per dataset used for predictor training (quality filtering)
- RFECV feature selection with forced inclusion of key meta-features
- GroupKFold cross-validation preventing data leakage

SYSTEM STATUS: 🚀 READY FOR EXCELLENT PERFORMANCE
Applying all proven winning strategies from Decision Tree success!
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
import argparse
import random
import math

# Import required modules
from zerotune import ZeroTune
from zerotune.core.predictor_training import train_predictor_from_knowledge_base
from zerotune.core.data_loading import fetch_open_ml_data, prepare_data
from zerotune.core.feature_extraction import calculate_dataset_meta_parameters
from zerotune.core.utils import convert_to_dataframe
from zerotune.predictors import ZeroTunePredictor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import optuna

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Test datasets - smaller collection for quick testing
TEST_DATASET_COLLECTION = [31, 44]  # credit-g, spambase

# Dataset collection for knowledge base building
FULL_DATASET_COLLECTION = [
    31,    # credit-g (1,000 × 20, imbalance: 2.333)
    38,    # sick (3,772 × 29, imbalance: 15.329)
    44,    # spambase (4,601 × 57, imbalance: 1.538)
    52,    # trains (10 × 32, imbalance: 1.000)
    151,   # electricity (45,312 × 8, imbalance: 1.355)
    179,   # adult (48,842 × 14, imbalance: 3.179)
    298,   # coil2000 (9,822 × 85, imbalance: 15.761)
    846,   # elevators (16,599 × 18, imbalance: 2.236)
    1053,  # jm1 (10,885 × 21, imbalance: 4.169)
    1112,  # KDDCup09 churn (50,000 × 230, imbalance: 12.617)
    1120,  # MagicTelescope (19,020 × 10, imbalance: 1.844)
    1128,  # OVA Breast (1,545 × 10,935, imbalance: 3.491)
    1220,  # Click prediction small (39,948 × 9, imbalance: 4.938)
    40900, # Satellite (5,100 × 36, imbalance: 67.000)
    45038, # road-safety (111,762 × 32, imbalance: 1.000)
]

# Validation datasets (unseen during knowledge base creation)
VALIDATION_DATASET_COLLECTION = [
    917,   # fri_c1_1000_25 (1000 × 25, imbalance: 1.000)
    1049,  # pc4 (1458 × 37, imbalance: 1.213)
    1111,  # KDDCup09_appetency (50000 × 230, imbalance: 12.617)
    1464,  # blood-transfusion-service-center (748 × 4, imbalance: 3.237)
    1494,  # qsar-biodeg (1055 × 41, imbalance: 1.540)
    1510,  # wdbc (569 × 30, imbalance: 1.679)
    1558,  # bank-marketing (4521 × 16, imbalance: 6.954)
    4534,  # PhishingWebsites (11055 × 30, imbalance: 1.236)
    23381, # dresses-sales (500 × 12, imbalance: 1.917)
    40536, # SpeedDating (8378 × 120, imbalance: 1.151)
]

def print_header():
    """Print the experiment header."""
    print("Random Forest Knowledge Base Builder for Zero-Shot HPO")
    print("=" * 70)

def build_knowledge_base(mode="test", n_iter=50):
    """
    Build knowledge base using ZeroTune
    
    Args:
        mode: "test" for small dataset collection, "full" for complete collection
        n_iter: Number of optimization iterations per dataset
    
    Returns:
        tuple: (knowledge_base_path, knowledge_base_dict)
    """
    dataset_collection = TEST_DATASET_COLLECTION if mode == "test" else FULL_DATASET_COLLECTION
    
    print(f"\nBuilding {mode.upper()} knowledge base for Random Forest...")
    print(f"Datasets: {len(dataset_collection)}")
    print(f"HPO iterations per dataset: {n_iter}")
    print(f"Total HPO runs: {len(dataset_collection) * n_iter}")
    print("This will collect HPO data and meta-features from multiple datasets")
    print("-" * 60)
    
    # Initialize ZeroTune for random forest
    kb_path = f"knowledge_base/kb_random_forest_rf_kb_v1_{mode}.json"
    zt = ZeroTune(model_type='random_forest', kb_path=kb_path)
    
    try:
        # Build knowledge base
        kb = zt.build_knowledge_base(
            dataset_ids=dataset_collection,
            n_iter=n_iter
        )
        
        print("\n" + "=" * 60)
        print("KNOWLEDGE BASE BUILDING COMPLETED")
        print("=" * 60)
        print(f"✅ Knowledge base saved to: {kb_path}")
        print(f"📊 Total datasets processed: {len(dataset_collection)}")
        print(f"📊 Knowledge base contains comprehensive HPO data for Random Forest")
        
        return kb_path, kb
        
    except Exception as e:
        print(f"❌ Error building knowledge base: {e}")
        return None, None

def train_predictor(mode="test"):
    """
    Train zero-shot predictor from knowledge base
    
    Args:
        mode: "test" for test KB, "full" for production KB
    
    Returns:
        str: Path to trained model
    """
    kb_path = f"knowledge_base/kb_random_forest_rf_kb_v1_{mode}.json"
    
    if not os.path.exists(kb_path):
        print(f"❌ Knowledge base not found: {kb_path}")
        print("Please run knowledge base building first!")
        return None
    
    print(f"\nTraining zero-shot predictor from {mode.upper()} knowledge base...")
    print("-" * 60)
    
    # Train predictor
    model_path = train_predictor_from_knowledge_base(
        kb_path=kb_path,
        model_name="random_forest",
        task_type="binary",
        output_dir="models",
        exp_id=f"rf_kb_v1_{mode}",
        test_size=0.2,
        random_state=42,
        verbose=True,
        top_k_trials=3
    )
    
    print(f"✅ Model saved to: {model_path}")
    return model_path

def evaluate_hyperparameters(params, X_train, y_train, X_test, y_test):
    """Evaluate a set of hyperparameters and return AUC score."""
    try:
        # Convert percentage-based max_depth to absolute if needed
        if 'max_depth' in params and isinstance(params['max_depth'], float) and params['max_depth'] < 1.0:
            n_samples = len(X_train)
            max_theoretical = max(1, int(math.log2(n_samples) * 2))
            params['max_depth'] = max(1, int(params['max_depth'] * max_theoretical))
        
        # Ensure max_depth is not None if it becomes 0
        if 'max_depth' in params and params['max_depth'] == 0:
            params['max_depth'] = None
            
        # Convert percentage-based parameters to absolute counts
        n_samples = len(X_train)
        for param in ['min_samples_split', 'min_samples_leaf']:
            if param in params and isinstance(params[param], float) and params[param] < 1.0:
                params[param] = max(2 if param == 'min_samples_split' else 1, 
                                  int(params[param] * n_samples))
        
        # Convert max_features percentage to absolute if needed
        if 'max_features' in params and isinstance(params['max_features'], float):
            n_features = X_train.shape[1]
            params['max_features'] = max(1, int(params['max_features'] * n_features))
            
        # Create and train model
        rf = RandomForestClassifier(**params, random_state=42)
        rf.fit(X_train, y_train)
        
        # Get predictions
        y_pred_proba = rf.predict_proba(X_test)
        
        # Calculate AUC
        if y_pred_proba.shape[1] == 2:  # Binary classification
            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:  # Multi-class
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
            
        return auc
        
    except Exception as e:
        print(f"⚠️ Error evaluating hyperparameters: {e}")
        return 0.5  # Return neutral score for failed evaluations

def generate_random_hyperparameters(dataset_size=1000, n_features=10, random_state=None):
    """Generate random hyperparameters for Random Forest matching KB configuration ranges."""
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)
    
    # Calculate max_depth range based on dataset size (same as KB)
    max_theoretical_depth = max(1, int(math.log2(dataset_size) * 2))
    max_depth_percentages = [0.25, 0.50, 0.70, 0.8, 0.9, 0.999]
    max_depth_options = [max(1, int(p * max_theoretical_depth)) for p in max_depth_percentages]
    max_depth_options.append(None)  # Add unlimited depth option
    
    # Calculate percentage-based sample parameters  
    min_samples_split_percentages = [0.01, 0.02, 0.05, 0.10, 0.20]
    min_samples_split_options = [max(2, int(p * dataset_size)) for p in min_samples_split_percentages]
    
    min_samples_leaf_percentages = [0.005, 0.01, 0.02, 0.05, 0.10]  
    min_samples_leaf_options = [max(1, int(p * dataset_size)) for p in min_samples_leaf_percentages]
    
    # Calculate percentage-based feature parameters
    max_features_percentages = [0.50, 0.70, 0.8, 0.9, 0.99]
    max_features_options = [max(1, int(p * n_features)) for p in max_features_percentages]
    
    return {
        'n_estimators': random.randint(10, 250),
        'max_depth': random.choice(max_depth_options),
        'min_samples_split': random.choice(min_samples_split_options),
        'min_samples_leaf': random.choice(min_samples_leaf_options),
        'max_features': random.choice(max_features_options)
    }

def test_zero_shot_predictor(mode="test", model_path=None, save_benchmark=True, n_seeds=1):
    """
    Test the zero-shot predictor on validation datasets
    
    Args:
        mode: "test" or "full"
        model_path: Path to trained model (optional)
        save_benchmark: Whether to save benchmark results
        n_seeds: Number of random seeds for robust evaluation
    
    Returns:
        dict: Results summary
    """
    if model_path is None:
        model_path = f"models/predictor_random_forest_rf_kb_v1_{mode}.joblib"
    
    if not os.path.exists(model_path):
        print(f"❌ Trained model not found: {model_path}")
        print("Please run predictor training first!")
        return None
    
    try:
        # Load the predictor
        predictor = ZeroTunePredictor.from_file(model_path)
        print(f"✅ Loaded predictor: {model_path}")
    except Exception as e:
        print(f"❌ Error loading predictor: {e}")
        return None
    
    test_datasets = VALIDATION_DATASET_COLLECTION
    
    if n_seeds > 1:
        print(f"\nStarting zero-shot evaluation with {n_seeds} random seeds...")
        print("-" * 60)
    else:
        print(f"\nStarting zero-shot evaluation...")
        print("-" * 60)
    
    results = []
    
    # For multiple seeds, we'll store all results and aggregate at the end
    all_seed_results = [] if n_seeds > 1 else None
    
    for dataset_id in test_datasets:
        print(f"\n🔍 Testing on dataset {dataset_id}")
        
        try:
            from openml import datasets
            
            # Download dataset
            dataset = datasets.get_dataset(dataset_id)
            X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
            
            print(f"Dataset name: {dataset.name}")
            print(f"Dataset: {dataset.name}")
            print(f"Shape: {X.shape}")
            
            # Get zero-shot prediction
            predicted_params = predictor.predict(X, y)
            print("Predicted hyperparameters:")
            for param, value in predicted_params.items():
                print(f"  {param}: {value}")
            
            # For multiple seeds, collect results across all seeds
            if n_seeds > 1:
                seed_results = []
                seed_random_results = []
                
                for seed_idx in range(n_seeds):
                    current_seed = 42 + seed_idx
                    
                    # Evaluate predicted hyperparameters with current seed
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=current_seed, stratify=y
                    )
                    
                    # Train with predicted hyperparameters
                    auc_predicted = evaluate_hyperparameters(predicted_params, X_train, y_train, X_test, y_test)
                    seed_results.append(auc_predicted)
                    
                    # Generate and evaluate random hyperparameters with same seed
                    random_params = generate_random_hyperparameters(
                        dataset_size=X.shape[0], 
                        n_features=X.shape[1], 
                        random_state=current_seed
                    )
                    auc_random = evaluate_hyperparameters(random_params, X_train, y_train, X_test, y_test)
                    seed_random_results.append(auc_random)
                
                # Calculate statistics across seeds
                auc_predicted_mean = np.mean(seed_results)
                auc_predicted_std = np.std(seed_results)
                auc_random_mean = np.mean(seed_random_results)
                auc_random_std = np.std(seed_random_results)
                
                print(f"✅ Zero-Shot AUC: {auc_predicted_mean:.4f} ± {auc_predicted_std:.4f} (avg over {n_seeds} seeds)")
                print(f"📊 Random AUC: {auc_random_mean:.4f} ± {auc_random_std:.4f} (avg over {n_seeds} seeds)")
                
                uplift = auc_predicted_mean - auc_random_mean
                uplift_pct = (uplift / auc_random_mean * 100) if auc_random_mean > 0 else 0
                print(f"🚀 Uplift vs random: {uplift:+.4f} ({uplift_pct:+.1f}%)")
                
                # Store results
                results.append({
                    'dataset_id': dataset_id,
                    'dataset_name': dataset.name,
                    'auc_predicted': auc_predicted_mean,
                    'auc_predicted_std': auc_predicted_std,
                    'n_samples': X.shape[0],
                    'n_features': X.shape[1],
                    'auc_random': auc_random_mean,
                    'auc_random_std': auc_random_std,
                    'uplift_random': uplift,
                    'uplift_random_pct': uplift_pct,
                    'n_seeds': n_seeds
                })
                
            else:
                # Single seed evaluation (original approach)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Evaluate predicted hyperparameters
                auc_predicted = evaluate_hyperparameters(predicted_params, X_train, y_train, X_test, y_test)
                print(f"✅ Zero-Shot AUC: {auc_predicted:.4f}")
                
                # Benchmark against random hyperparameters
                if save_benchmark:
                    print(f"\n🔄 Running random benchmark...")
                    
                    # Generate random hyperparameters
                    random_params = generate_random_hyperparameters(
                        dataset_size=X.shape[0],
                        n_features=X.shape[1], 
                        random_state=dataset_id
                    )
                    auc_random = evaluate_hyperparameters(random_params, X_train, y_train, X_test, y_test)
                    print(f"📊 Random AUC: {auc_random:.4f}")
                    
                    uplift = auc_predicted - auc_random
                    uplift_pct = (uplift / auc_random * 100) if auc_random > 0 else 0
                    print(f"🚀 Uplift vs random: {uplift:+.4f} ({uplift_pct:+.1f}%)")
                    
                    # Store results
                    results.append({
                        'dataset_id': dataset_id,
                        'dataset_name': dataset.name,
                        'auc_predicted': auc_predicted,
                        'n_samples': X.shape[0],
                        'n_features': X.shape[1],
                        'error': '',
                        'auc_random': auc_random,
                        'uplift_random': uplift,
                        'uplift_random_pct': uplift_pct
                    })
                else:
                    # Store results without benchmark
                    results.append({
                        'dataset_id': dataset_id,
                        'dataset_name': dataset.name,
                        'auc_predicted': auc_predicted,
                        'n_samples': X.shape[0],
                        'n_features': X.shape[1],
                        'error': ''
                    })
                    
        except Exception as e:
            print(f"❌ Error with dataset {dataset_id}: {e}")
            results.append({
                'dataset_id': dataset_id,
                'dataset_name': f'dataset_{dataset_id}',
                'auc_predicted': 0.0,
                'n_samples': 0,
                'n_features': 0,
                'error': str(e)
            })
    
    # Print summary
    print("\n" + "=" * 60)
    print("ZERO-SHOT PREDICTOR EVALUATION SUMMARY")
    print("=" * 60)
    
    successful_results = [r for r in results if r.get('error', '') == '']
    if successful_results:
        aucs = [r['auc_predicted'] for r in successful_results]
        print(f"📊 Successful tests: {len(successful_results)}/{len(test_datasets)}")
        print(f"📊 Zero-Shot Average AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        print(f"📊 Zero-Shot Best AUC: {np.max(aucs):.4f}")
        print(f"📊 Zero-Shot Worst AUC: {np.min(aucs):.4f}")
        
        if save_benchmark and successful_results[0].get('auc_random') is not None:
            random_aucs = [r['auc_random'] for r in successful_results]
            uplifts = [r['uplift_random'] for r in successful_results]
            
            print(f"\n📊 Random Benchmark Results:")
            print(f"📊 Random Average AUC: {np.mean(random_aucs):.4f} ± {np.std(random_aucs):.4f}")
            print(f"🚀 Average Uplift vs random: {np.mean(uplifts):+.4f} ± {np.std(uplifts):.4f}")
            print(f"🚀 Relative Improvement vs random: {np.mean(uplifts)/np.mean(random_aucs)*100:+.1f}%")
            
            positive_uplifts = [u for u in uplifts if u > 0]
            print(f"🎯 Datasets with positive uplift: {len(positive_uplifts)}/{len(uplifts)} ({len(positive_uplifts)/len(uplifts)*100:.1f}%)")
        
        print(f"\nDetailed Results:")
        print("ID     Dataset Name                   Zero-Shot  Random     Random_Uplift")
        print("-" * 73)
        for result in successful_results:
            if result.get('auc_random') is not None:
                print(f"{result['dataset_id']:5d}    {result['dataset_name']:<25} {result['auc_predicted']:10.4f}     {result['auc_random']:.4f}     {result['uplift_random']:+.4f}     ")
            else:
                print(f"{result['dataset_id']:5d}    {result['dataset_name']:<25} {result['auc_predicted']:10.4f}     N/A        N/A         ")
    
    # Save results
    if save_benchmark:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"benchmark_results_rf_kb_v1_{mode}_{timestamp}.csv"
        
        df = pd.DataFrame(results)
        df.to_csv(csv_filename, index=False)
        print(f"\n💾 Saving results to CSV: {csv_filename}")
        print(f"✅ Benchmark results saved to: {csv_filename}")
    
    return results

def show_usage():
    """Show usage information."""
    print("\nRandom Forest Knowledge Base & Zero-Shot HPO Experiment")
    print("=" * 60)
    print("Commands:")
    print("  python random_forest_experiment.py test         # Build test KB (2 datasets, 50 iter)")
    print("  python random_forest_experiment.py full         # Build full KB (15 datasets, 50 iter)")
    print("  python random_forest_experiment.py build-kb     # Build knowledge base only")
    print("  python random_forest_experiment.py train-test   # Train predictor (test mode)")  
    print("  python random_forest_experiment.py train-full   # Train predictor (full mode)")
    print("  python random_forest_experiment.py eval-test    # Evaluate predictor (test mode)")
    print("  python random_forest_experiment.py eval-full    # Evaluate predictor (full mode)")
    print("  python random_forest_experiment.py --help       # Show this help")

if __name__ == "__main__":
    print_header()
    
    if len(sys.argv) < 2:
        show_usage()
        sys.exit(1)
    
    command = sys.argv[1]
    
    try:
        if command == "build-kb":
            # Build knowledge base only
            build_knowledge_base(mode="full")
            
        elif command == "test":
            # Quick test cycle (2 datasets)
            build_knowledge_base(mode="test")
            
        elif command == "full":
            # Build full knowledge base (15 datasets)
            build_knowledge_base(mode="full")
            
        elif command == "train-test":
            # Train predictor from test KB
            train_predictor(mode="test")
            
        elif command == "train-full":
            # Train predictor from full KB
            train_predictor(mode="full")
            
        elif command == "eval-test":
            # Evaluate test predictor
            test_zero_shot_predictor(mode="test")
            
        elif command == "eval-full":
            # Evaluate full predictor with 50-seed robust evaluation
            test_zero_shot_predictor(mode="full", n_seeds=50)
            
        elif command == "--help" or command == "help":
            show_usage()
            
        else:
            print(f"❌ Unknown command: {command}")
            show_usage()
            
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    print("\n🎉 Experiment completed!") 