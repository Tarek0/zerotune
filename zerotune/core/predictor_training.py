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
from sklearn.metrics import mean_squared_error, r2_score


def train_predictor_from_knowledge_base(
    kb_path: str,
    model_name: str,
    task_type: str = "binary",
    output_dir: str = "models",
    exp_id: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True
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
    X_features, y_params, feature_names, param_names = _prepare_training_data(kb_data)
    
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
    
    # Train the predictor model
    if verbose:
        print("Training Random Forest predictor...")
    
    predictor_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=random_state,
        n_jobs=-1
    )
    predictor_model.fit(X_train_norm, y_train)
    
    # Evaluate the model
    y_pred = predictor_model.predict(X_test_norm)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    if verbose:
        print(f"Predictor performance: MSE = {mse:.6f}, R² = {r2:.4f}")
    
    # Create model data structure
    model_data = {
        'model': predictor_model,
        'dataset_features': feature_names,
        'target_params': param_names,
        'normalization_params': norm_params,
        'training_info': {
            'kb_path': kb_path,
            'exp_id': exp_id,
            'n_training_samples': len(X_train),
            'n_test_samples': len(X_test),
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


def _prepare_training_data(kb_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Prepare training data from knowledge base.
    
    Args:
        kb_data: Knowledge base data dictionary
        
    Returns:
        Tuple of (X_features, y_params, feature_names, param_names)
    """
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
    
    # Get all parameter names (use first result as reference)
    first_result = results_list[0]
    if 'best_hyperparameters' in first_result:
        param_names = [f"params_{param}" for param in first_result['best_hyperparameters'].keys()]
    elif 'best_params' in first_result:
        param_names = [f"params_{param}" for param in first_result['best_params'].keys()]
    else:
        return np.array([]), np.array([]), [], []
    
    X_features = []
    y_params = []
    
    # For each result, use the corresponding meta-features or the first available
    for i, result in enumerate(results_list):
        # Use corresponding meta-features if available, otherwise use the first one
        if i < len(meta_features_list):
            features = meta_features_list[i]
        else:
            features = meta_features_list[0]  # Fallback to first meta-features
        
        # Feature vector
        feature_vector = [features.get(fname, 0.0) for fname in feature_names]
        X_features.append(feature_vector)
        
        # Target parameter vector
        param_vector = []
        # Check which key is used for parameters
        params_dict = result.get('best_hyperparameters', result.get('best_params', {}))
        
        for param_name in param_names:
            clean_param = param_name.replace('params_', '')
            param_value = params_dict.get(clean_param, 0.0)
            param_vector.append(float(param_value))
        y_params.append(param_vector)
    
    return np.array(X_features), np.array(y_params), feature_names, param_names


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