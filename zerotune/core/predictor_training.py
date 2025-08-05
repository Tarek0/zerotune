"""
Predictor Training Module for ZeroTune.

This module provides functionality to train zero-shot hyperparameter predictors
from knowledge bases built by the ZeroTune system.
"""

import os
import json
import joblib
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold


def train_predictor_from_knowledge_base(
    kb_path: str,
    model_name: str,
    task_type: str = "binary",
    output_dir: str = "models",
    exp_id: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True,
    top_k_per_seed: int = 3
) -> str:
    """
    Train a zero-shot predictor from a knowledge base.
    
    Args:
        kb_path: Path to the knowledge base JSON file
        model_name: Name of the target model ("xgboost", "random_forest", etc.)
        task_type: Type of task ("binary", "multiclass", "regression")
        output_dir: Directory to save the trained predictor
        exp_id: Experiment ID for naming (extracted from kb_path if None)
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        verbose: Whether to print progress information
        top_k_per_seed: Number of top trials to keep per dataset/seed combination
        
    Returns:
        Path to the saved predictor model
    """
    if verbose:
        print(f"Training zero-shot predictor from knowledge base: {kb_path}")
    
    # Load knowledge base
    if not os.path.exists(kb_path):
        raise FileNotFoundError(f"Knowledge base not found: {kb_path}")
    
    with open(kb_path, 'r') as f:
        kb_data = json.load(f)
    
    # Extract experiment ID from kb_path if not provided
    if exp_id is None:
        kb_filename = os.path.basename(kb_path)
        # Extract from pattern: kb_{model_type}_{exp_id}_{mode}.json
        parts = kb_filename.replace('.json', '').split('_')
        if len(parts) >= 3:
            exp_id = '_'.join(parts[2:-1])  # Everything between model_type and mode
        else:
            exp_id = "unknown"
    
    if verbose:
        print(f"Experiment ID: {exp_id}")
        print(f"Target model: {model_name}_{task_type}")
    
    # Prepare training data
    X_features, y_params, feature_names, param_names = _prepare_training_data(kb_data, top_k_per_seed, verbose)
    
    if len(X_features) == 0:
        raise ValueError("No training data found in knowledge base")
    
    if verbose:
        print(f"Training data: {len(X_features)} samples, {len(feature_names)} features, {len(param_names)} target params")
    
    # Handle insufficient data for train/test split
    if len(X_features) < 2:
        if verbose:
            print("Warning: Insufficient data for train/test split. Using all data for training.")
        X_train, X_test = X_features, X_features
        y_train, y_test = y_params, y_params
        test_size = 0.0  # Update test_size for logging
    else:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y_params, test_size=test_size, random_state=random_state
        )
    
    # Calculate normalization parameters
    norm_params = _calculate_normalization_params(X_train, feature_names)
    
    # Normalize training data
    X_train_norm = _normalize_features(X_train, norm_params, feature_names)
    X_test_norm = _normalize_features(X_test, norm_params, feature_names)
    
    # Train the predictor model with hyperparameter optimization
    if verbose:
        print("Training Random Forest predictor with hyperparameter optimization...")
    
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint as sp_randint
    
    # Define hyperparameter search space
    param_dist = {
        'n_estimators': sp_randint(100, 300),
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': sp_randint(2, 11),
        'min_samples_leaf': sp_randint(1, 5),
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }
    
    # Initialize base regressor
    base_regressor = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    
    # Set up RandomizedSearchCV
    hpo_search = RandomizedSearchCV(
        base_regressor, 
        param_distributions=param_dist, 
        cv=4,
        scoring='neg_mean_squared_error', 
        n_jobs=-1, 
        n_iter=100, 
        random_state=random_state,
        verbose=1 if verbose else 0
    )
    
    # Fit with hyperparameter optimization
    hpo_search.fit(X_train_norm, y_train)
    
    # Get the best predictor
    best_predictor = hpo_search.best_estimator_
    
    if verbose:
        print(f"Best hyperparameters: {hpo_search.best_params_}")
        print(f"Best CV score (neg_MSE): {hpo_search.best_score_:.4f}")
    
    # Apply RFECV for feature selection using the tuned model
    if verbose:
        print("Applying RFECV feature selection...")
    
    # Use KFold cross-validation for RFECV
    cv_folds = KFold(n_splits=min(4, len(X_train_norm)), shuffle=True, random_state=random_state)
    
    # Create RFECV with the best estimator
    rfecv = RFECV(
        estimator=best_predictor,
        step=1,
        cv=cv_folds,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    # Fit RFECV to select optimal features
    rfecv.fit(X_train_norm, y_train)
    
    # Transform training data to selected features
    X_train_selected = rfecv.transform(X_train_norm)
    X_test_selected = rfecv.transform(X_test_norm)
    
    # Get selected feature names
    selected_feature_mask = rfecv.support_
    selected_features = [feature_names[i] for i, selected in enumerate(selected_feature_mask) if selected]
    
    if verbose:
        print(f"RFECV selected {rfecv.n_features_} out of {len(feature_names)} features")
        print(f"Selected features: {selected_features}")
        print(f"Feature ranking: {rfecv.ranking_}")
    
    # Retrain the final model on selected features only
    final_predictor = RandomForestRegressor(**hpo_search.best_params_, random_state=random_state, n_jobs=-1)
    final_predictor.fit(X_train_selected, y_train)
    
    # Update the predictor model to the final one
    predictor_model = final_predictor
    
    # Evaluate the model on selected features
    y_pred = predictor_model.predict(X_test_selected)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate NMAE per parameter
    nmae_scores = _calculate_parameter_nmae(y_test, y_pred, param_names)
    avg_nmae = np.mean(list(nmae_scores.values()))
    
    # Calculate Top-K accuracy (better than random)
    topk_results = _calculate_top_k_accuracy(y_test, y_pred, param_names, k=10)
    
    if verbose:
        print(f"Predictor performance:")
        print(f"  Overall: MSE = {mse:.6f}, R² = {r2:.4f}, Avg NMAE = {avg_nmae:.4f}")
        print(f"  Top-K Accuracy: {topk_results['overall_top_k_accuracy']:.1%} (better than {topk_results['k_baselines']} random baselines)")
        print(f"  Per-parameter NMAE:")
        for param_name, nmae in nmae_scores.items():
            clean_name = param_name.replace('params_', '')
            topk_acc = topk_results['param_specific_accuracy'].get(param_name, 0.0)
            print(f"    {clean_name}: NMAE={nmae:.4f}, Top-K={topk_acc:.1%}")
        print(f"  🎯 Interpretation: {topk_results['overall_top_k_accuracy']:.1%} of predictions outperform random hyperparameters")
    
    # Create model data structure
    model_data = {
        'model': predictor_model,
        'dataset_features': feature_names,
        'selected_features': selected_features,
        'feature_selector': rfecv,
        'target_params': param_names,
        'normalization_params': norm_params,
        'training_info': {
            'kb_path': kb_path,
            'exp_id': exp_id,
            'n_training_samples': len(X_train),
            'n_test_samples': len(X_test),
            'n_original_features': len(feature_names),
            'n_selected_features': len(selected_features),
            'mse': mse,
            'r2': r2,
            'model_name': model_name,
            'task_type': task_type
        }
    }
    
    # Save the trained model
    os.makedirs(output_dir, exist_ok=True)
    model_filename = f"{model_name}_{task_type}_classifier_{exp_id}.joblib"
    if task_type == "regression":
        model_filename = f"{model_name}_regressor_{exp_id}.joblib"
    
    model_path = os.path.join(output_dir, model_filename)
    joblib.dump(model_data, model_path)
    
    if verbose:
        print(f"✅ Trained predictor saved to: {model_path}")
    
    return model_path


def _prepare_training_data(kb_data: Dict[str, Any], top_k_per_seed: int = 3, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Prepare training data from knowledge base, filtering to top-k trials per dataset/seed.
    
    Args:
        kb_data: Knowledge base data dictionary
        top_k_per_seed: Number of top trials to keep per dataset/seed combination
        verbose: Whether to print progress information
        
    Returns:
        Tuple of (X_features, y_params, feature_names, param_names)
    """
    import pandas as pd
    
    meta_features = kb_data.get('meta_features', [])
    results = kb_data.get('results', [])
    
    if not meta_features or not results:
        return np.array([]), np.array([]), [], []
    
    # Handle both list and dict formats
    if isinstance(meta_features, list):
        meta_features_list = meta_features
    else:
        meta_features_list = list(meta_features.values())
    
    if isinstance(results, list):
        results_list = results
    else:
        results_list = list(results.values())
    
    if not meta_features_list or not results_list:
        return np.array([]), np.array([]), [], []
    
    # Get all feature names (use first dataset as reference)
    first_dataset = meta_features_list[0]
    feature_names = list(first_dataset.keys())
    
    X_features = []
    y_params = []
    param_names = None
    total_trials_before = 0
    total_trials_after = 0
    
    # Process each result (dataset)
    for i, result in enumerate(results_list):
        # Use corresponding meta-features if available, otherwise use the first one
        if i < len(meta_features_list):
            features = meta_features_list[i]
        else:
            features = meta_features_list[0]  # Fallback to first meta-features
        
        # Check if this result has trials dataframe
        if 'trials_dataframe' in result and 'trials_columns' in result:
            # Reconstruct DataFrame from stored data
            trials_data = result['trials_dataframe']
            trials_columns = result['trials_columns']
            df_trials = pd.DataFrame(trials_data, columns=trials_columns)
            
            total_trials_before += len(df_trials)
            
            # Filter to top-k trials per seed
            if 'seed' in df_trials.columns and 'value' in df_trials.columns:
                # Group by seed and get top-k trials per seed
                top_trials = df_trials.groupby('seed').apply(
                    lambda x: x.nlargest(top_k_per_seed, 'value')
                ).reset_index(drop=True)
            else:
                # If no seed column, just get top-k overall
                top_trials = df_trials.nlargest(top_k_per_seed, 'value') if 'value' in df_trials.columns else df_trials
            
            total_trials_after += len(top_trials)
            
            # Extract parameter names from the first trial if not already set
            if param_names is None:
                param_cols = [col for col in top_trials.columns if col.startswith('params_')]
                param_names = param_cols
            
            # Add each top trial as a training sample
            for _, trial in top_trials.iterrows():
                # Feature vector (same meta-features for all trials from this dataset)
                feature_vector = [features.get(fname, 0.0) for fname in feature_names]
                X_features.append(feature_vector)
                
                # Parameter vector from this trial
                param_vector = []
                for param_name in param_names:
                    param_value = trial.get(param_name, 0.0)
                    param_vector.append(float(param_value))
                y_params.append(param_vector)
        
        else:
            # Fallback: use best_hyperparameters if no trials dataframe
            total_trials_before += 1
            total_trials_after += 1
            
            # Feature vector
            feature_vector = [features.get(fname, 0.0) for fname in feature_names]
            X_features.append(feature_vector)
            
            # Get parameter names from best result if not already set
            if param_names is None:
                params_dict = result.get('best_hyperparameters', result.get('best_params', {}))
                param_names = [f"params_{param}" for param in params_dict.keys()]
            
            # Target parameter vector
            param_vector = []
            params_dict = result.get('best_hyperparameters', result.get('best_params', {}))
            
            for param_name in param_names:
                clean_param = param_name.replace('params_', '')
                param_value = params_dict.get(clean_param, 0.0)
                param_vector.append(float(param_value))
            y_params.append(param_vector)
    
    if verbose and total_trials_before > 0:
        print(f"Filtered trials: {total_trials_before} → {total_trials_after} (kept top-{top_k_per_seed} per seed)")
    
    return np.array(X_features), np.array(y_params), feature_names, param_names if param_names else []


def _calculate_normalization_params(X: np.ndarray, feature_names: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Calculate normalization parameters for features.
    
    Args:
        X: Feature matrix
        feature_names: List of feature names
        
    Returns:
        Dictionary of normalization parameters for each feature
    """
    norm_params = {}
    
    for i, feature_name in enumerate(feature_names):
        feature_values = X[:, i]
        norm_params[feature_name] = {
            'min': float(np.min(feature_values)),
            'max': float(np.max(feature_values)),
            'range': float(np.max(feature_values) - np.min(feature_values))
        }
    
    return norm_params


def _calculate_parameter_nmae(y_true: np.ndarray, y_pred: np.ndarray, param_names: List[str]) -> Dict[str, float]:
    """
    Calculate Normalized Mean Absolute Error (NMAE) for each parameter.
    
    Args:
        y_true: True parameter values (n_samples, n_params)
        y_pred: Predicted parameter values (n_samples, n_params)
        param_names: Names of parameters
        
    Returns:
        Dictionary mapping parameter names to NMAE values
    """
    # Define parameter ranges for normalization
    param_ranges = {
        'params_n_estimators': (50, 1000),
        'params_max_depth': (1, 16), 
        'params_learning_rate': (0.01, 0.3),
        'params_subsample': (0.5, 1.0),
        'params_colsample_bytree': (0.5, 1.0)
    }
    
    nmae_scores = {}
    
    for i, param_name in enumerate(param_names):
        if param_name in param_ranges:
            min_val, max_val = param_ranges[param_name]
            
            # Normalize to [0, 1]
            y_true_norm = (y_true[:, i] - min_val) / (max_val - min_val)
            y_pred_norm = (y_pred[:, i] - min_val) / (max_val - min_val)
            
            # Calculate MAE on normalized values
            nmae = mean_absolute_error(y_true_norm, y_pred_norm)
            nmae_scores[param_name] = nmae
        else:
            # Fallback: use raw MAE for unknown parameters
            mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
            nmae_scores[param_name] = mae
    
    return nmae_scores


def _calculate_top_k_accuracy(y_true: np.ndarray, y_pred: np.ndarray, param_names: List[str], k: int = 5) -> Dict[str, float]:
    """
    Calculate Top-K Parameter Set Accuracy by comparing predicted hyperparameters 
    against random baselines.
    
    Args:
        y_true: True parameter values (n_samples, n_params)
        y_pred: Predicted parameter values (n_samples, n_params)
        param_names: Names of parameters
        k: Number of random baselines to generate for comparison
        
    Returns:
        Dictionary with accuracy metrics
    """
    # Define parameter ranges for random generation
    param_ranges = {
        'params_n_estimators': [50, 100, 200, 500, 1000],  # Categorical
        'params_max_depth': list(range(1, 17)),  # Integer range
        'params_learning_rate': (0.01, 0.3),  # Continuous
        'params_subsample': (0.5, 1.0),  # Continuous
        'params_colsample_bytree': (0.5, 1.0)  # Continuous
    }
    
    n_samples = y_true.shape[0]
    better_than_random_count = 0
    total_comparisons = 0
    
    param_specific_accuracy = {}
    
    for i, param_name in enumerate(param_names):
        if param_name in param_ranges:
            param_better_count = 0
            
            for sample_idx in range(n_samples):
                true_val = y_true[sample_idx, i]
                pred_val = y_pred[sample_idx, i]
                
                # Calculate error for our prediction
                pred_error = abs(true_val - pred_val)
                
                # Generate k random values and calculate their errors
                random_errors = []
                for _ in range(k):
                    if isinstance(param_ranges[param_name], list):
                        # Categorical parameter
                        random_val = np.random.choice(param_ranges[param_name])
                    else:
                        # Continuous parameter
                        min_val, max_val = param_ranges[param_name]
                        random_val = np.random.uniform(min_val, max_val)
                    
                    random_error = abs(true_val - random_val)
                    random_errors.append(random_error)
                
                # Check if our prediction is better than at least one random value
                if pred_error < max(random_errors):
                    param_better_count += 1
                    better_than_random_count += 1
                
                total_comparisons += 1
            
            # Parameter-specific accuracy
            param_specific_accuracy[param_name] = param_better_count / n_samples
    
    overall_accuracy = better_than_random_count / total_comparisons if total_comparisons > 0 else 0.0
    
    return {
        'overall_top_k_accuracy': overall_accuracy,
        'param_specific_accuracy': param_specific_accuracy,
        'k_baselines': k
    }


def _normalize_features(X: np.ndarray, norm_params: Dict[str, Dict[str, float]], feature_names: List[str]) -> np.ndarray:
    """
    Normalize features using the provided normalization parameters.
    
    Args:
        X: Feature matrix
        norm_params: Normalization parameters
        feature_names: List of feature names
        
    Returns:
        Normalized feature matrix
    """
    X_norm = X.copy()
    
    for i, feature_name in enumerate(feature_names):
        if feature_name in norm_params:
            norm_info = norm_params[feature_name]
            min_val = norm_info['min']
            range_val = norm_info['range']
            
            if range_val > 0:
                X_norm[:, i] = (X_norm[:, i] - min_val) / range_val
            else:
                X_norm[:, i] = 0.0
    
    return X_norm 