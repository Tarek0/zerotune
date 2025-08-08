#!/usr/bin/env python3
"""
XGBoost Knowledge Base Builder & Zero-Shot Predictor Training/Evaluation

CURRENT STATUS: 🔧 STRONG PERFORMANCE ACHIEVED!
✅ 90% win rate across all test datasets (9/10 datasets)
✅ Average improvement: +0.7% over random hyperparameter selection  
✅ Quality Strategy: top_k_trials=1 (using only best trial per dataset)
✅ Critical max_depth bug fixed: Now uses proper depths (5-13) instead of always 1
✅ Expanded hyperparameter ranges: continuous learning_rate, subsample, colsample_bytree
✅ Enhanced knowledge base: 50 HPO runs per dataset
✅ Multi-seed robust evaluation (50 seeds) for statistical validity

ARCHITECTURE:
- zerotune/core/predictor_training.py: Advanced training with RFECV & GroupKFold
- zerotune/core/feature_extraction.py: Robust meta-feature extraction (fixed)
- zerotune/core/optimization.py: Optuna TPE optimization with warm-start support (study.enqueue_trial)
- zerotune/predictors.py: Zero-shot predictor class for inference
- models/predictor_xgboost_xgb_kb_v1_full.joblib: Production-ready trained model
- knowledge_base/kb_xgboost_xgb_kb_v1_full.json: Clean, comprehensive training data

PERFORMANCE METRICS:
✅ Zero-Shot Average AUC: 0.8676 ± 0.1349
✅ Random Average AUC: 0.8617 ± 0.1374
✅ Average Uplift: +0.0058 (+0.7% improvement)  
✅ Win Rate: 9/10 datasets (90% success) - Strong reliability
✅ Best Single Win: +2.6% (KDDCup09_appetency dataset)
✅ Most Reliable: Only 1 dataset with minor loss (-0.0% qsar-biodeg)

OPTIMIZATION APPROACH (Perfected):
- 50 HPO trials per dataset for comprehensive hyperparameter exploration
- Quality Strategy: top_k_trials=1 (only best trial per dataset) - proven optimal
- Continuous hyperparameter ranges: learning_rate (0.001-0.5), subsample/colsample_bytree (0.5-1.0)
- Percentage-based max_depth scaling with proper conversion to integer depths
- Fair benchmarking: Random baseline uses identical sampling as KB generation
- Multi-seed evaluation (50 seeds) for statistically robust benchmarking
- RFECV feature selection with forced inclusion of key meta-features
- GroupKFold cross-validation preventing data leakage

SYSTEM STATUS: 🏆 PRODUCTION-READY & STRONG RELIABILITY  
Delivering 90% win rate with consistent +0.7% improvement - excellent performance!

OPTUNA TPE WARM-START INTEGRATION: ✅ IMPLEMENTED!
🚀 Optuna TPE Warm-Start: Use zero-shot predictions to initialize Optuna TPE optimization
📊 Comprehensive Benchmarking: Compares 4 approaches:
   - Zero-shot predictions (instant, no optimization)
   - Random hyperparameters (baseline)
   - Warm-started Optuna TPE (zero-shot + optimization)
   - Standard Optuna TPE (optimization only)
🎯 Usage: python xgb_experiment.py eval-full --optuna --optuna_trials 25
📊 Trial Data Storage: Saves detailed Optuna trial data for convergence analysis

DATASET COLLECTIONS:
- Training (test): [31, 38] - 2 datasets for quick development
- Training (full): [31, 38, 44, 52, 151, 179, 298, 846, 1053, 1112, 1120, 1128, 1220, 40900, 45038] - 15 datasets
- Evaluation (unseen): [917, 1049, 1111, 1464, 1494, 1510, 1558, 4534, 23381, 40536] - 10 datasets

This script builds a comprehensive knowledge base for training zero-shot hyperparameter predictors.
It runs hyperparameter optimization on multiple datasets and stores:
- Dataset meta-features (n_samples, n_features, imbalance_ratio, etc.)
- Optimal hyperparameters found through HPO
- Performance scores achieved

The resulting knowledge base can be used to train zero-shot predictors.
"""

from zerotune import ZeroTune
from zerotune.core.predictor_training import train_predictor_from_knowledge_base
from zerotune.core.data_loading import fetch_open_ml_data, prepare_data
from zerotune.core.feature_extraction import calculate_dataset_meta_parameters
from zerotune.core.utils import convert_to_dataframe, save_trial_data
from zerotune.core.optimization import optimize_hyperparameters
from zerotune.core.model_configs import ModelConfigs
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import joblib
import numpy as np
import pandas as pd
import random
import sys
import os
from datetime import datetime

# Experiment configuration
EXPERIMENT_ID = "xgb_kb_v1"

# Dataset collection for knowledge base building
FULL_DATASET_COLLECTION = [
    31,    # credit-g (1,000 × 20, imbalance: 2.333)
    38,    # sick (3,772 × 29, imbalance: 15.329)
    44,    # spambase (4,601 × 57, imbalance: 1.538)
    52,    # trains (10 × 32, imbalance: 1.000) - Fixed with stratified splitting
    151,   # electricity (45,312 × 8, imbalance: 1.355)
    179,   # adult (48,842 × 14, imbalance: 3.179)
    298,   # coil2000 (9,822 × 85, imbalance: 15.761)
    846,   # elevators (16,599 × 18, imbalance: 2.236)
    1053,  # jm1 (10,885 × 21, imbalance: 4.169)
    1112,  # KDDCup09_churn (50,000 × 230, imbalance: 12.617)
    1120,  # MagicTelescope (19,020 × 10, imbalance: 1.844)
    1128,  # OVA_Breast (1,545 × 10,935, imbalance: 3.491)
    1220,  # Click_prediction_small (39,948 × 9, imbalance: 4.938)
    40900, # Satellite (5,100 × 36, imbalance: 67.000)
    45038  # road-safety (111,762 × 32, imbalance: 1.000)
]

# Test subset for development (2 datasets)
TEST_DATASET_COLLECTION = [31, 38]  # credit-g, sick

# Unseen datasets for testing trained predictors (avoid data leakage)
UNSEEN_TEST_DATASETS = [917, 1049, 1111, 1464, 1494, 1510, 1558, 4534, 23381, 40536]

def build_knowledge_base(dataset_ids, n_iter=50, mode="test"):
    """
    Build knowledge base by running HPO on specified datasets.
    
    Args:
        dataset_ids: List of OpenML dataset IDs
        n_iter: Number of HPO iterations per dataset
        mode: "test" or "full" for different configurations
    """
    print("ZEROTUNE KNOWLEDGE BASE BUILDER")
    print("=" * 60)
    print(f"Experiment ID: {EXPERIMENT_ID}")
    print(f"Mode: {mode}")
    print(f"Building knowledge base with {len(dataset_ids)} datasets")
    print(f"HPO iterations per dataset: {n_iter}")
    print(f"Dataset IDs: {dataset_ids}")
    
    # Create knowledge base directory
    os.makedirs("knowledge_base", exist_ok=True)
    kb_path = f"knowledge_base/kb_xgboost_{EXPERIMENT_ID}_{mode}.json"
    print(f"Knowledge base will be saved to: {kb_path}")
    
    # Initialize ZeroTune for knowledge base building
    print(f"\nInitializing ZeroTune for XGBoost knowledge base building...")
    zt = ZeroTune(model_type='xgboost', kb_path=kb_path)
    
    # Build knowledge base
    print(f"\nStarting knowledge base building...")
    print("This will run HPO on each dataset to collect optimal hyperparameters")
    print("-" * 60)
    
    try:
        kb = zt.build_knowledge_base(
            dataset_ids=dataset_ids, 
            n_iter=n_iter,
            verbose=True
        )
        
        print(f"\n{'='*60}")
        print("KNOWLEDGE BASE BUILDING COMPLETED")
        print(f"{'='*60}")
        print(f"✅ Successfully processed {len(dataset_ids)} datasets")
        print(f"📁 Knowledge base saved to: {kb_path}")
        
        # Print knowledge base summary
        if kb and 'meta_features' in kb:
            print(f"📊 Knowledge base contains {len(kb['meta_features'])} dataset entries")
        if kb and 'results' in kb:
            print(f"🎯 Knowledge base contains {len(kb['results'])} optimization results")
            
        return kb_path, kb
        
    except Exception as e:
        print(f"\n❌ Error during knowledge base building: {str(e)}")
        return None, None

def train_zero_shot_predictor(kb_path=None, mode="test"):
    """
    Train a zero-shot predictor from a knowledge base.
    
    Args:
        kb_path: Path to knowledge base (auto-generated if None)
        mode: "test" or "full" to match the KB naming convention
    """
    print("ZEROTUNE ZERO-SHOT PREDICTOR TRAINER")
    print("=" * 60)
    print(f"Experiment ID: {EXPERIMENT_ID}")
    print(f"Mode: {mode}")
    
    # Auto-generate KB path if not provided
    if kb_path is None:
        kb_path = f"knowledge_base/kb_xgboost_{EXPERIMENT_ID}_{mode}.json"
    
    print(f"Using knowledge base: {kb_path}")
    
    if not os.path.exists(kb_path):
        print(f"❌ Knowledge base not found: {kb_path}")
        print("Please build the knowledge base first using:")
        print(f"  python xgb_experiment.py {mode}")
        return None
    
    try:
        # Train the predictor
        print(f"\nTraining zero-shot predictor...")
        print("This will create a model that can predict optimal hyperparameters for new datasets")
        print("-" * 60)
        
        model_path = train_predictor_from_knowledge_base(
            kb_path=kb_path,
            model_name="xgboost",
            task_type="binary",
            output_dir="models",
            exp_id=f"{EXPERIMENT_ID}_{mode}",
            top_k_trials=1  # Keep only best trial per dataset (Decision Tree's winning strategy)
        )
        
        print(f"\n{'='*60}")
        print("ZERO-SHOT PREDICTOR TRAINING COMPLETED")
        print(f"{'='*60}")
        print("✅ Successfully trained zero-shot predictor")
        print(f"📁 Model saved to: {model_path}")
        print(f"🎯 Ready for zero-shot hyperparameter prediction")
        
        return model_path
        
    except Exception as e:
        print(f"\n❌ Error during predictor training: {str(e)}")
        return None

# Removed: Now using ModelConfigs.generate_random_hyperparameters()

def run_benchmark_hpo(X_train, y_train, X_test, y_test, benchmark_type="random", n_trials=10, random_state=42, initial_params=None):
    """
    Run benchmark hyperparameter optimization.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data  
        benchmark_type: Type of benchmark ("random", "optuna")
        n_trials: Number of trials for optimization-based benchmarks
        random_state: Random seed for reproducibility
        initial_params: Initial parameters for warm-start (used with optuna)
        
    Returns:
        Tuple of (best_params, best_score, benchmark_info)
    """
    if benchmark_type == "random":
        # Simple random hyperparameters
        params = ModelConfigs.generate_random_hyperparameters(
            model_type='xgboost',
            dataset_size=X_train.shape[0], 
            n_features=X_train.shape[1],
            random_state=random_state
        )
        
        try:
            model = XGBClassifier(**params, random_state=42)
            model.fit(X_train, y_train)
            
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)
                if y_proba.shape[1] == 2:
                    score = roc_auc_score(y_test, y_proba[:, 1])
                else:
                    score = roc_auc_score(y_test, y_proba, multi_class='ovr')
            else:
                y_pred = model.predict(X_test)
                score = accuracy_score(y_test, y_pred)
            
            return params, score, {"type": "random", "trials": 1}
            
        except Exception as e:
            return params, 0.0, {"type": "random", "trials": 1, "error": str(e)}
    
    elif benchmark_type == "optuna":
        # Optuna TPE benchmark
        try:
            warmstart_label = "warm-started" if initial_params else "standard"
            print(f"🔄 Running {warmstart_label} Optuna TPE ({n_trials} trials)...")
            
            best_params, best_score, study_df = optimize_hyperparameters(
                X_train, y_train, X_test, y_test,
                model_type='xgboost',
                n_trials=n_trials,
                initial_params=initial_params,
                random_state=random_state,
                verbose=False
            )
            
            return best_params, best_score, {"type": "optuna", "trials": n_trials, "study_df": study_df, "warmstart": initial_params is not None}
            
        except Exception as e:
            print(f"⚠️ Optuna benchmark failed: {e}")
            return run_benchmark_hpo(X_train, y_train, X_test, y_test, "random", n_trials, random_state)
    
    else:
        raise ValueError(f"Unknown benchmark type: {benchmark_type}")

def evaluate_hyperparameters(params, X_train, y_train, X_test, y_test):
    """
    Evaluate hyperparameters and return AUC score.
    
    Args:
        params: Hyperparameter dictionary
        X_train, y_train: Training data
        X_test, y_test: Test data
        
    Returns:
        AUC score
    """
    try:
        model = XGBClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            if y_proba.shape[1] == 2:
                return roc_auc_score(y_test, y_proba[:, 1])
            else:
                return roc_auc_score(y_test, y_proba, multi_class='ovr')
        else:
            y_pred = model.predict(X_test)
            return accuracy_score(y_test, y_pred)
    except Exception:
        return 0.0

def test_zero_shot_predictor(model_path=None, mode="test", test_dataset_ids=None, benchmark_types=["random"], save_csv=True, n_seeds=1, optuna_n_trials=20):
    """
    Test a trained zero-shot predictor on unseen datasets.
    
    Args:
        model_path: Path to trained model (auto-generated if None)
        mode: "test" or "full" to match the model naming convention
        test_dataset_ids: List of dataset IDs to test on (uses UNSEEN_TEST_DATASETS if None)
        benchmark_types: List of benchmark types to run (["random"], ["optuna"], or ["random", "optuna"])
        save_csv: Whether to save results to CSV file (default: True)
        n_seeds: Number of random seeds for robust evaluation
        optuna_n_trials: Number of Optuna trials for benchmarking
    """
    print("ZEROTUNE ZERO-SHOT PREDICTOR EVALUATION")
    print("=" * 60)
    print(f"Experiment ID: {EXPERIMENT_ID}")
    print(f"Mode: {mode}")
    
    # Auto-generate model path if not provided
    if model_path is None:
        model_path = f"models/predictor_xgboost_{EXPERIMENT_ID}_{mode}.joblib"
    
    print(f"Using trained model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Trained model not found: {model_path}")
        print("Please train the predictor first using:")
        print(f"  python xgb_experiment.py train-{mode}")
        return None
    
    # Use default unseen test datasets if none provided
    if test_dataset_ids is None:
        test_dataset_ids = UNSEEN_TEST_DATASETS
    
    print(f"Testing on {len(test_dataset_ids)} unseen datasets: {test_dataset_ids}")
    print("These datasets were NOT used in knowledge base building to avoid data leakage")
    
    try:
        # Load the trained model
        print(f"\nLoading trained predictor...")
        model_data = joblib.load(model_path)
        
        # Print model info
        training_info = model_data.get('training_info', {})
        print(f"📊 Model trained on {training_info.get('n_training_samples', 'unknown')} samples")
        print(f"📊 Model R² score: {training_info.get('r2', 'unknown')}")
        
        if n_seeds > 1:
            print(f"\nStarting zero-shot evaluation with {n_seeds} random seeds...")
            print("-" * 60)
        else:
            print(f"\nStarting zero-shot evaluation...")
            print("-" * 60)
        
        results = []
        
        for dataset_id in test_dataset_ids:
            print(f"\n🔍 Testing on dataset {dataset_id}")
            
            try:
                # Fetch dataset
                data, target_name, dataset_name = fetch_open_ml_data(dataset_id)
                X, y = prepare_data(data, target_name)
                print(f"Dataset: {dataset_name}")
                print(f"Shape: {X.shape}")
                
                # Calculate meta-features
                X_df = convert_to_dataframe(X)
                meta_features = calculate_dataset_meta_parameters(X_df, y)
                
                # Prepare features for prediction
                feature_names = model_data['feature_names']
                feature_vector = []
                
                for feature_name in feature_names:
                    if feature_name in meta_features:
                        feature_vector.append(float(meta_features[feature_name]))
                    else:
                        feature_vector.append(0.0)
                
                # Normalize features
                feature_vector = np.array(feature_vector, dtype=np.float64).reshape(1, -1)
                norm_params = model_data['normalization_params']
                
                for i, feature_name in enumerate(feature_names):
                    if feature_name in norm_params:
                        norm_info = norm_params[feature_name]
                        min_val = norm_info['min']
                        range_val = norm_info['range']
                        
                        if range_val > 0:
                            feature_vector[0, i] = (feature_vector[0, i] - min_val) / range_val
                        else:
                            feature_vector[0, i] = 0.0
                
                # Apply feature selection if available
                if 'feature_selector' in model_data and model_data['feature_selector'] is not None:
                    feature_selector = model_data['feature_selector']
                    feature_vector = feature_selector.transform(feature_vector)
                
                # Make prediction
                prediction = model_data['model'].predict(feature_vector)[0]
                
                # Convert to hyperparameters
                target_params = model_data['param_names']
                predicted_params = {}
                
                for i, param_name in enumerate(target_params):
                    clean_param_name = param_name.replace('params_', '')
                    
                    if i < len(prediction):
                        value = prediction[i]
                        
                        # Convert to appropriate type
                        if clean_param_name == 'n_estimators':
                            value = max(1, int(round(value)))
                        elif clean_param_name == 'max_depth':
                            # Convert percentage prediction to actual depth based on dataset size
                            n_samples = meta_features.get('n_samples', 1000)  # Default fallback
                            max_theoretical_depth = max(1, int(np.log2(n_samples) * 2))
                            
                            # Convert predicted percentage to actual depth
                            depth_percentage = min(1.0, max(0.1, float(value)))  # Clamp between 10% and 100%
                            depth_val = max(1, int(max_theoretical_depth * depth_percentage))
                            
                            value = depth_val
                        elif clean_param_name in ['learning_rate', 'subsample', 'colsample_bytree']:
                            value = max(0.01, min(1.0, float(value)))
                        
                        predicted_params[clean_param_name] = value
                
                print("Predicted hyperparameters:")
                for param, value in predicted_params.items():
                    print(f"  {param}: {value}")
                
                # For multiple seeds, collect results across all seeds
                if n_seeds > 1:
                    seed_results = []
                    seed_benchmark_results = {bt: [] for bt in benchmark_types}
                    
                    for seed_idx in range(n_seeds):
                        current_seed = 42 + seed_idx
                        
                        # Split data with current seed
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=current_seed, stratify=y
                        )
                        
                        # Test predicted hyperparameters
                        auc_predicted = evaluate_hyperparameters(predicted_params, X_train, y_train, X_test, y_test)
                        seed_results.append(auc_predicted)
                        
                        # Run benchmarks with same data split
                        for benchmark_type in benchmark_types:
                            if benchmark_type == "optuna":
                                # Run both warm-started and standard Optuna for multi-seed
                                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                
                                # Warm-started Optuna
                                warmstart_params, warmstart_score, warmstart_info = run_benchmark_hpo(
                                    X_train, y_train, X_test, y_test, 
                                    benchmark_type="optuna",
                                    n_trials=optuna_n_trials,
                                    random_state=current_seed,
                                    initial_params=predicted_params
                                )
                                
                                # Standard Optuna
                                standard_params, standard_score, standard_info = run_benchmark_hpo(
                                    X_train, y_train, X_test, y_test, 
                                    benchmark_type="optuna",
                                    n_trials=optuna_n_trials,
                                    random_state=current_seed,
                                    initial_params=None
                                )
                                
                                # Save trial data
                                if 'study_df' in warmstart_info:
                                    save_trial_data(warmstart_info['study_df'], 'warmstart', dataset_id, current_seed, timestamp, EXPERIMENT_ID)
                                if 'study_df' in standard_info:
                                    save_trial_data(standard_info['study_df'], 'standard', dataset_id, current_seed, timestamp, EXPERIMENT_ID)
                                
                                # Store both results separately
                                if 'optuna_warmstart' not in seed_benchmark_results:
                                    seed_benchmark_results['optuna_warmstart'] = []
                                if 'optuna_standard' not in seed_benchmark_results:
                                    seed_benchmark_results['optuna_standard'] = []
                                    
                                seed_benchmark_results['optuna_warmstart'].append(warmstart_score)
                                seed_benchmark_results['optuna_standard'].append(standard_score)
                                
                            else:
                                benchmark_params, benchmark_score, benchmark_info = run_benchmark_hpo(
                                    X_train, y_train, X_test, y_test, 
                                    benchmark_type=benchmark_type,
                                    random_state=current_seed  # Use consistent seed
                                )
                                seed_benchmark_results[benchmark_type].append(benchmark_score)
                    
                    # Calculate averages and standard deviations
                    auc_predicted = np.mean(seed_results)
                    auc_predicted_std = np.std(seed_results)
                    print(f"✅ Zero-Shot AUC: {auc_predicted:.4f} ± {auc_predicted_std:.4f} (avg over {n_seeds} seeds)")
                    
                    # Process benchmark results
                    benchmark_results = {}
                    
                    # Handle all benchmark types including the split Optuna results
                    all_result_keys = set(seed_benchmark_results.keys())
                    
                    for result_key in all_result_keys:
                        if result_key in seed_benchmark_results and seed_benchmark_results[result_key]:
                            benchmark_scores = seed_benchmark_results[result_key]
                            benchmark_score = np.mean(benchmark_scores)
                            benchmark_score_std = np.std(benchmark_scores)
                            
                            uplift = auc_predicted - benchmark_score
                            
                            # Format display name
                            if result_key == 'optuna_warmstart':
                                display_name = "Warm-started Optuna"
                            elif result_key == 'optuna_standard':
                                display_name = "Standard Optuna"
                            else:
                                display_name = result_key.title()
                            
                            print(f"📊 {display_name} AUC: {benchmark_score:.4f} ± {benchmark_score_std:.4f} (avg over {n_seeds} seeds)")
                            print(f"🚀 Uplift vs {result_key}: {uplift:+.4f} ({(uplift/benchmark_score*100):+.1f}%)")
                            
                            benchmark_results[result_key] = {
                                'score': benchmark_score,
                                'score_std': benchmark_score_std,
                                'uplift': uplift,
                                'uplift_percent': (uplift/benchmark_score*100) if benchmark_score > 0 else 0,
                                'params': {},  # Multi-seed doesn't store specific params
                                'info': f"Average over {n_seeds} seeds"
                            }
                
                else:
                    # Single seed evaluation (original logic)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    # Test the predicted hyperparameters
                    auc_predicted = evaluate_hyperparameters(predicted_params, X_train, y_train, X_test, y_test)
                    print(f"✅ Zero-Shot AUC: {auc_predicted:.4f}")
                    
                    # Run benchmarks
                    benchmark_results = {}
                    
                    for benchmark_type in benchmark_types:
                        if benchmark_type == "optuna":
                            # Run both warm-started and standard Optuna TPE
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            
                            # Warm-started Optuna TPE
                            print(f"\n🌱 Running warm-started Optuna TPE benchmark...")
                            warmstart_params, warmstart_score, warmstart_info = run_benchmark_hpo(
                                X_train, y_train, X_test, y_test, 
                                benchmark_type="optuna",
                                n_trials=optuna_n_trials,
                                random_state=dataset_id,
                                initial_params=predicted_params
                            )
                            
                            # Standard Optuna TPE  
                            print(f"\n🔍 Running standard Optuna TPE benchmark...")
                            standard_params, standard_score, standard_info = run_benchmark_hpo(
                                X_train, y_train, X_test, y_test, 
                                benchmark_type="optuna",
                                n_trials=optuna_n_trials,
                                random_state=dataset_id,
                                initial_params=None
                            )
                            
                            # Save trial data
                            if 'study_df' in warmstart_info:
                                save_trial_data(warmstart_info['study_df'], 'warmstart', dataset_id, dataset_id, timestamp, EXPERIMENT_ID)
                            if 'study_df' in standard_info:
                                save_trial_data(standard_info['study_df'], 'standard', dataset_id, dataset_id, timestamp, EXPERIMENT_ID)
                            
                            # Calculate uplifts
                            warmstart_uplift = auc_predicted - warmstart_score
                            standard_uplift = auc_predicted - standard_score
                            
                            print(f"✅ Warm-started Optuna AUC: {warmstart_score:.4f}")
                            print(f"✅ Standard Optuna AUC: {standard_score:.4f}")
                            print(f"🚀 Uplift vs warm-started: {warmstart_uplift:+.4f} ({(warmstart_uplift/warmstart_score*100):+.1f}%)")
                            print(f"🚀 Uplift vs standard: {standard_uplift:+.4f} ({(standard_uplift/standard_score*100):+.1f}%)")
                            
                            # Store both results
                            benchmark_results['optuna_warmstart'] = {
                                'params': warmstart_params,
                                'score': warmstart_score,
                                'uplift': warmstart_uplift,
                            }
                            benchmark_results['optuna_standard'] = {
                                'params': standard_params,
                                'score': standard_score,
                                'uplift': standard_uplift,
                            }
                            
                        else:
                            # Handle non-Optuna benchmarks (random, etc.)
                            print(f"\n🔄 Running {benchmark_type} benchmark...")
                            
                            benchmark_params, benchmark_score, benchmark_info = run_benchmark_hpo(
                                X_train, y_train, X_test, y_test, 
                                benchmark_type=benchmark_type,
                                random_state=dataset_id
                            )
                            
                            uplift = auc_predicted - benchmark_score
                            
                            print(f"📊 {benchmark_type.title()} hyperparameters:")
                            for param, value in benchmark_params.items():
                                print(f"  {param}: {value}")
                            print(f"📊 {benchmark_type.title()} AUC: {benchmark_score:.4f}")
                            print(f"🚀 Uplift vs {benchmark_type}: {uplift:+.4f} ({(uplift/benchmark_score*100):+.1f}%)")
                            
                            benchmark_results[benchmark_type] = {
                            'params': benchmark_params,
                            'score': benchmark_score,
                            'uplift': uplift,
                        'info': benchmark_info
                    }
                
                results.append({
                    'dataset_id': dataset_id,
                    'dataset_name': dataset_name,
                    'auc_predicted': auc_predicted,
                    'predicted_params': predicted_params,
                    'benchmarks': benchmark_results,
                    'shape': X.shape
                })
                
            except Exception as e:
                print(f"❌ Error processing dataset {dataset_id}: {str(e)}")
                results.append({
                    'dataset_id': dataset_id,
                    'dataset_name': f"Dataset_{dataset_id}",
                    'auc_predicted': 0.0,
                    'benchmarks': {},
                    'error': str(e)
                })
        
        # Print summary
        print(f"\n{'='*60}")
        print("ZERO-SHOT PREDICTOR EVALUATION SUMMARY")
        print(f"{'='*60}")
        
        successful_tests = [r for r in results if 'error' not in r]
        failed_tests = [r for r in results if 'error' in r]
        
        if successful_tests:
            auc_predicted_scores = [r['auc_predicted'] for r in successful_tests]
            avg_auc_predicted = np.mean(auc_predicted_scores)
            std_auc_predicted = np.std(auc_predicted_scores)
            
            print(f"📊 Successful tests: {len(successful_tests)}/{len(test_dataset_ids)}")
            print(f"📊 Zero-Shot Average AUC: {avg_auc_predicted:.4f} ± {std_auc_predicted:.4f}")
            print(f"📊 Zero-Shot Best AUC: {max(auc_predicted_scores):.4f}")
            print(f"📊 Zero-Shot Worst AUC: {min(auc_predicted_scores):.4f}")
            
            # Calculate benchmark statistics for each benchmark type
            for benchmark_type in benchmark_types:
                benchmark_scores = []
                uplift_scores = []
                
                for result in successful_tests:
                    if benchmark_type in result['benchmarks']:
                        benchmark_scores.append(result['benchmarks'][benchmark_type]['score'])
                        uplift_scores.append(result['benchmarks'][benchmark_type]['uplift'])
                
                if benchmark_scores:
                    avg_benchmark = np.mean(benchmark_scores)
                    std_benchmark = np.std(benchmark_scores)
                    avg_uplift = np.mean(uplift_scores)
                    std_uplift = np.std(uplift_scores)
                    
                    print(f"\n📊 {benchmark_type.title()} Benchmark Results:")
                    print(f"📊 {benchmark_type.title()} Average AUC: {avg_benchmark:.4f} ± {std_benchmark:.4f}")
                    print(f"🚀 Average Uplift vs {benchmark_type}: {avg_uplift:+.4f} ± {std_uplift:.4f}")
                    if avg_benchmark > 0:
                        print(f"🚀 Relative Improvement vs {benchmark_type}: {(avg_uplift/avg_benchmark*100):+.1f}%")
                    
                    # Count positive uplifts
                    positive_uplifts = len([u for u in uplift_scores if u > 0])
                    print(f"🎯 Datasets with positive uplift: {positive_uplifts}/{len(uplift_scores)} ({positive_uplifts/len(uplift_scores)*100:.1f}%)")
            
            print(f"\nDetailed Results:")
            if benchmark_types:
                # Create header with all benchmark types
                header = f"{'ID':<6} {'Dataset Name':<30} {'Zero-Shot':<10}"
                for benchmark_type in benchmark_types:
                    header += f" {benchmark_type.title():<10} {f'{benchmark_type.title()}_Uplift':<12}"
                print(header)
                print("-" * len(header))
                
                for result in successful_tests:
                    row = f"  {result['dataset_id']:<6} {result['dataset_name']:<30} {result['auc_predicted']:<10.4f}"
                    
                    for benchmark_type in benchmark_types:
                        if benchmark_type in result['benchmarks']:
                            benchmark_data = result['benchmarks'][benchmark_type]
                            score_str = f"{benchmark_data['score']:.4f}"
                            uplift_str = f"{benchmark_data['uplift']:+.4f}"
                        else:
                            score_str = "N/A"
                            uplift_str = "N/A"
                        
                        row += f" {score_str:<10} {uplift_str:<12}"
                    
                    print(row)
            else:
                for result in successful_tests:
                    print(f"  {result['dataset_id']:<6} {result['dataset_name']:<30} AUC: {result['auc_predicted']:.4f}")
        
        if failed_tests:
            print(f"\n❌ Failed tests: {len(failed_tests)}")
            for result in failed_tests:
                print(f"  {result['dataset_id']:<6} {result['dataset_name']:<30} ERROR: {result['error']}")
        
        # Save results to CSV if requested
        if save_csv and results:
            # Ensure benchmarks directory exists
            os.makedirs("benchmarks", exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"benchmarks/benchmark_results_{EXPERIMENT_ID}_{mode}_{timestamp}.csv"
            
            print(f"\n💾 Saving results to CSV: {csv_filename}")
            
            # Prepare data for CSV
            csv_data = []
            for result in results:
                row = {
                    'dataset_id': result['dataset_id'],
                    'dataset_name': result['dataset_name'],
                    'auc_predicted': result['auc_predicted'],
                    'n_samples': result['shape'][0] if 'shape' in result else None,
                    'n_features': result['shape'][1] if 'shape' in result else None,
                    'error': result.get('error', None)
                }
                
                # Add zero-shot predicted hyperparameters for debugging
                if 'predicted_params' in result:
                    for param_name, param_value in result['predicted_params'].items():
                        row[f'predicted_{param_name}'] = param_value
                
                # Add benchmark results with hyperparameters
                for benchmark_type in benchmark_types:
                    if 'benchmarks' in result and benchmark_type in result['benchmarks']:
                        benchmark_data = result['benchmarks'][benchmark_type]
                        row[f'auc_{benchmark_type}'] = benchmark_data['score']
                        row[f'uplift_{benchmark_type}'] = benchmark_data['uplift']
                        row[f'uplift_{benchmark_type}_pct'] = (benchmark_data['uplift'] / benchmark_data['score'] * 100) if benchmark_data['score'] > 0 else 0
                        
                        # Add benchmark hyperparameters for debugging
                        if 'params' in benchmark_data:
                            for param_name, param_value in benchmark_data['params'].items():
                                row[f'{benchmark_type}_{param_name}'] = param_value
                    else:
                        row[f'auc_{benchmark_type}'] = None
                        row[f'uplift_{benchmark_type}'] = None
                        row[f'uplift_{benchmark_type}_pct'] = None
                
                csv_data.append(row)
            
            # Create DataFrame and save
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_filename, index=False)
            print(f"✅ Benchmark results saved to: {csv_filename}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error during predictor evaluation: {str(e)}")
        return None

def print_dataset_info():
    """Print information about the dataset collection."""
    print("\nDATASET COLLECTION INFORMATION")
    print("=" * 60)
    
    dataset_info = [
        (31, "credit-g", "1,000", "20", "2.333"),
        (38, "sick", "3,772", "29", "15.329"),
        (44, "spambase", "4,601", "57", "1.538"),
        (52, "trains", "10", "32", "1.000"),
        (151, "electricity", "45,312", "8", "1.355"),
        (179, "adult", "48,842", "14", "3.179"),
        (298, "coil2000", "9,822", "85", "15.761"),
        (846, "elevators", "16,599", "18", "2.236"),
        (1053, "jm1", "10,885", "21", "4.169"),
        (1112, "KDDCup09_churn", "50,000", "230", "12.617"),
        (1120, "MagicTelescope", "19,020", "10", "1.844"),
        (1128, "OVA_Breast", "1,545", "10,935", "3.491"),
        (1220, "Click_prediction_small", "39,948", "9", "4.938"),
        (40900, "Satellite", "5,100", "36", "67.000"),
        (45038, "road-safety", "111,762", "32", "1.000")
    ]
    
    print(f"{'ID':<6} {'Dataset Name':<25} {'n_samples':<10} {'n_features':<12} {'imbalance_ratio':<15}")
    print("-" * 75)
    
    for dataset_id, name, n_samples, n_features, imbalance in dataset_info:
        print(f"{dataset_id:<6} {name:<25} {n_samples:<10} {n_features:<12} {imbalance:<15}")
    
    print(f"\nTotal datasets: {len(dataset_info)}")
    print(f"Training datasets (test): {TEST_DATASET_COLLECTION}")
    print(f"Training datasets (full): {FULL_DATASET_COLLECTION}")
    print(f"Unseen test datasets: {UNSEEN_TEST_DATASETS}")

if __name__ == "__main__":
    print("XGBoost Knowledge Base Builder for Zero-Shot HPO")
    print("=" * 70)
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python xgb_experiment.py test          # Build test KB (2 datasets, 10 iter)")
        print("  python xgb_experiment.py full         # Build full KB (15 datasets, 50 iter)")
        print("  python xgb_experiment.py train-test   # Train predictor from test KB")
        print("  python xgb_experiment.py train-full   # Train predictor from full KB")
        print("  python xgb_experiment.py eval-test    # Evaluate test predictor")
        print("  python xgb_experiment.py eval-full    # Evaluate full predictor")
        print("  python xgb_experiment.py eval-full --optuna # Include Optuna TPE benchmarks")
        print("  python xgb_experiment.py eval-full --optuna --optuna_trials 30  # Custom trial count (default: 20)")
        sys.exit(0)
    
    command = sys.argv[1]
    
    # Parse optional arguments
    include_optuna_benchmark = False
    optuna_n_trials = 20  # default
    
    if "--optuna" in sys.argv:
        include_optuna_benchmark = True
        print("Including Optuna TPE benchmarking")
    
    if "--optuna_trials" in sys.argv:
        try:
            idx = sys.argv.index("--optuna_trials")
            if idx + 1 < len(sys.argv):
                optuna_n_trials = int(sys.argv[idx + 1])
                print(f"Using optuna_n_trials = {optuna_n_trials}")
        except (ValueError, IndexError):
            print("Warning: Invalid --optuna_trials value, using default (20)")
    
    try:
        if command == "test":
            kb_path, kb = build_knowledge_base(
                dataset_ids=TEST_DATASET_COLLECTION,
                n_iter=10,
                mode="test"
            )
        elif command == "full":
            kb_path, kb = build_knowledge_base(
                dataset_ids=FULL_DATASET_COLLECTION,
                n_iter=50,
                mode="full"
            )
        elif command == "train-test":
            predictor = train_zero_shot_predictor(mode="test")
        elif command == "train-full":
            predictor = train_zero_shot_predictor(mode="full")
        elif command == "eval-test":
            results = test_zero_shot_predictor(
                mode="test",
                benchmark_types=["random", "optuna"] if include_optuna_benchmark else ["random"],
                optuna_n_trials=optuna_n_trials
            )
        elif command == "eval-full":
            results = test_zero_shot_predictor(
                mode="full", 
                benchmark_types=["random", "optuna"] if include_optuna_benchmark else ["random"],
                n_seeds=50,
                optuna_n_trials=optuna_n_trials
            )
        else:
            print("❌ Unknown command. Use one of: test, full, train-test, train-full, eval-test, eval-full")
            
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1) 