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
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import make_scorer


class CombinedFeatureSelector:
    """
    Custom feature selector that combines forced inclusion of important features
    with RFECV-selected statistical features.
    """
    def __init__(self, support_mask):
        self.support_ = support_mask
        self.n_features_ = np.sum(support_mask)
        
    def transform(self, X):
        return X[:, self.support_]
        
    def get_support(self, indices=False):
        return self.support_ if not indices else np.where(self.support_)[0]


def train_predictor_from_knowledge_base(
    kb_path: str,
    model_name: str = "xgboost",
    task_type: str = "binary",
    output_dir: str = "models",
    exp_id: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True,
    top_k_trials: int = 3
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
        top_k_trials: Number of top trials to keep per dataset
        
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
    
    # Extract training data from knowledge base
    X_features, y_params, feature_names, param_names, dataset_groups = _prepare_training_data(kb_data, top_k_trials, verbose)
    
    if len(X_features) == 0:
        raise ValueError("No training data found in knowledge base")
    
    if verbose:
        print(f"Training data shape: {X_features.shape}")
        print(f"Target data shape: {y_params.shape}")
        print(f"Features: {feature_names}")
        print(f"Parameters: {param_names}")
    
    # Check if we have enough data for train/test split
    if len(X_features) < 2:
        print(f"⚠️  Warning: Only {len(X_features)} samples available. Using all data for training (no test split).")
        X_train, X_test = X_features, X_features
        y_train, y_test = y_params, y_params
        groups_train, groups_test = dataset_groups, dataset_groups
        test_size = 0.0  # For logging purposes
    else:
        # Split data using GroupShuffleSplit to ensure datasets don't get split across train/test
        from sklearn.model_selection import GroupShuffleSplit
        
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X_features, y_params, groups=dataset_groups))
        
        X_train, X_test = X_features[train_idx], X_features[test_idx]
        y_train, y_test = y_params[train_idx], y_params[test_idx]
        groups_train, groups_test = dataset_groups[train_idx], dataset_groups[test_idx]
    
    # Calculate normalization parameters from training data
    normalization_params = _calculate_normalization_params(X_train, feature_names)
    
    # Normalize features
    X_train_norm = _normalize_features(X_train, normalization_params)
    X_test_norm = _normalize_features(X_test, normalization_params)
    
    if verbose:
        print(f"Training set: {X_train_norm.shape[0]} samples")
        print(f"Test set: {X_test_norm.shape[0]} samples")
        print(f"Number of unique datasets in training: {len(np.unique(groups_train))}")
    
    # Use MultiOutputRegressor to handle multiple target parameters
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import RandomizedSearchCV, GroupKFold
    from scipy.stats import randint as sp_randint
    
    # Define hyperparameter search space for the predictor itself (much simpler to avoid overfitting)
    param_dist = {
        'n_estimators': sp_randint(10, 50),           # Much fewer trees
        'max_depth': [3, 5, 7, 10],                  # Much shallower trees  
        'min_samples_split': sp_randint(10, 30),     # Higher regularization
        'min_samples_leaf': sp_randint(5, 15),       # Higher regularization
        'max_features': ['sqrt', 'log2'],            # Feature subsampling only
        'bootstrap': [True]                          # Always use bootstrap
    }
    
    # Create base regressor
    base_regressor = RandomForestRegressor(random_state=random_state)
    regressor = MultiOutputRegressor(base_regressor)
    
    # Set up cross-validation strategy using GroupKFold to prevent data leakage
    n_groups = len(np.unique(groups_train))
    cv_strategy = GroupKFold(n_splits=min(4, n_groups))
    
    if verbose:
        print(f"Using GroupKFold with {min(4, n_groups)} splits")
        print("Training predictor with hyperparameter optimization...")
    
    # Hyperparameter optimization for the predictor itself
    search = RandomizedSearchCV(
        regressor,
        param_distributions={'estimator__' + key: value for key, value in param_dist.items()},
        n_iter=50,
        cv=cv_strategy,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        random_state=random_state,
        verbose=0
    )
    
    # Fit with groups to ensure proper cross-validation
    search.fit(X_train_norm, y_train, groups=groups_train)
    
    # Get the best predictor
    best_predictor = search.best_estimator_
    
    if verbose:
        print(f"Best predictor R² score: {search.best_score_:.4f}")
        print(f"Best hyperparameters: {search.best_params_}")
    
    # Feature selection using RFECV with the best predictor
    from sklearn.feature_selection import RFECV
    from sklearn.metrics import make_scorer
    
    # Define minimum number of features to select (at least 25% or 5, whichever is higher)
    min_features = max(5, len(feature_names) // 4)
    
    if verbose:
        print(f"Performing feature selection with RFECV (min features: {min_features})...")
    
    # Create a base RandomForestRegressor for RFECV (not MultiOutputRegressor)
    # We'll use the first target parameter for feature selection
    rfecv_estimator = RandomForestRegressor(
        **{k.replace('estimator__', ''): v for k, v in search.best_params_.items()}, 
        random_state=random_state
    )
    
    # Use NMAE as the scoring metric for RFECV (we'll use just the first parameter for selection)
    def single_output_nmae_scorer(y_true, y_pred):
        """NMAE scorer for single output (first parameter only)"""
        # Use the first parameter (colsample_bytree) for feature selection
        param_ranges = {0: (0.5, 1.0)}  # colsample_bytree range
        min_val, max_val = param_ranges[0]
        
        # Normalize to [0, 1]
        y_true_norm = (y_true - min_val) / (max_val - min_val)
        y_pred_norm = (y_pred - min_val) / (max_val - min_val)
        
        # Calculate MAE on normalized values
        from sklearn.metrics import mean_absolute_error
        nmae = mean_absolute_error(y_true_norm, y_pred_norm)
        return -nmae  # Negative because sklearn expects higher=better
    
    nmae_scorer = make_scorer(single_output_nmae_scorer, greater_is_better=True)
    
    rfecv = RFECV(
        estimator=rfecv_estimator,
        step=1,
        cv=cv_strategy,
        scoring=nmae_scorer,
        min_features_to_select=min_features,
        n_jobs=-1
    )
    
    # Fit RFECV with groups using only the first parameter for feature selection
    rfecv.fit(X_train_norm, y_train[:, 0], groups=groups_train)
    
    # Force inclusion of important meta-features
    important_features = ['n_samples', 'n_features', 'imbalance_ratio', 'n_highly_target_corr']
    important_indices = [i for i, name in enumerate(feature_names) if name in important_features]
    
    if verbose:
        print(f"RFECV selected {rfecv.n_features_} features out of {len(feature_names)}")
        print(f"Forcing inclusion of {len(important_indices)} important features: {important_features}")
    
    # Combine important features with RFECV-selected features
    selected_feature_mask = rfecv.support_.copy()
    for idx in important_indices:
        selected_feature_mask[idx] = True
    
    X_train_selected = X_train_norm[:, selected_feature_mask]
    X_test_selected = X_test_norm[:, selected_feature_mask]
    selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_feature_mask[i]]
    
    if verbose:
        print(f"Final feature set: {len(selected_features)} features")
        print(f"Selected features: {selected_features}")
    
    # Train final model on selected features
    final_predictor = MultiOutputRegressor(
        RandomForestRegressor(**{k.replace('estimator__', ''): v for k, v in search.best_params_.items()}, 
                             random_state=random_state)
    )
    final_predictor.fit(X_train_selected, y_train)
    
    # Evaluate on test set
    if test_size > 0:
        y_pred = final_predictor.predict(X_test_selected)
        
        # Calculate metrics
        from sklearn.metrics import r2_score
        r2 = r2_score(y_test, y_pred, multioutput='uniform_average')
        
        # Calculate NMAE for each parameter
        nmae_scores = _calculate_parameter_nmae(y_test, y_pred, param_names)
        avg_nmae = np.mean(list(nmae_scores.values()))
        
        # Calculate Top-K accuracy
        top_k_scores = _calculate_top_k_accuracy(y_test, y_pred, param_names, k_values=[1, 3, 5])
        
        if verbose:
            print(f"\n=== PREDICTOR EVALUATION ===")
            print(f"R² Score: {r2:.4f}")
            print(f"Average NMAE: {avg_nmae:.4f}")
            print("NMAE per parameter:")
            for param, nmae in nmae_scores.items():
                print(f"  {param}: {nmae:.4f}")
            print("Top-K Accuracy (better than random):")
            for k, acc in top_k_scores.items():
                print(f"  Top-{k}: {acc:.1%}")
    
    # Create combined feature selector for saving
    feature_selector = CombinedFeatureSelector(selected_feature_mask)
    
    # Prepare model data for saving
    model_data = {
        'model': final_predictor,
        'feature_names': feature_names,
        'param_names': param_names,
        'normalization_params': normalization_params,
        'feature_selector': feature_selector,
        'selected_features': selected_features,
        'model_name': model_name,
        'task_type': task_type,
        'training_info': {
            'kb_path': kb_path,
            'n_training_samples': len(X_train),
            'n_test_samples': len(X_test),
            'n_datasets': len(np.unique(dataset_groups)),
            'r2_score': r2 if test_size > 0 else None,
            'average_nmae': avg_nmae if test_size > 0 else None,
            'top_k_accuracy': top_k_scores if test_size > 0 else None,
            'best_predictor_params': search.best_params_,
            'selected_feature_count': len(selected_features)
        }
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate model filename
    if exp_id is None:
        # Extract experiment ID from knowledge base filename
        kb_filename = os.path.basename(kb_path)
        if kb_filename.startswith('kb_'):
            exp_id = kb_filename.replace('kb_', '').replace('.json', '')
        else:
            exp_id = 'default'
    
    model_filename = f"predictor_{model_name}_{exp_id}.joblib"
    model_path = os.path.join(output_dir, model_filename)
    
    # Save the model
    import joblib
    joblib.dump(model_data, model_path)
    
    if verbose:
        print(f"\n=== MODEL SAVED ===")
        print(f"Predictor saved to: {model_path}")
        print(f"Model contains:")
        print(f"  - Trained predictor: {type(final_predictor).__name__}")
        print(f"  - Feature names: {len(feature_names)} total, {len(selected_features)} selected")
        print(f"  - Parameter names: {len(param_names)}")
        print(f"  - Normalization parameters")
        print(f"  - Feature selector")
    
    return model_path


def _prepare_training_data(kb_data: Dict[str, Any], top_k_trials: int = 3, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], np.ndarray]:
    """
    Prepare training data from knowledge base, filtering to top-k trials per dataset.
    
    Args:
        kb_data: Knowledge base data dictionary
        top_k_trials: Number of top trials to keep per dataset
        verbose: Whether to print progress information
        
    Returns:
        Tuple of (X_features, y_params, feature_names, param_names, dataset_groups)
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
    dataset_groups = []  # Track which dataset each sample comes from
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
            
            # Get top-k trials overall (no seed-based grouping)
            if 'value' in df_trials.columns:
                top_trials = df_trials.nlargest(top_k_trials, 'value')
            else:
                top_trials = df_trials.head(top_k_trials)  # Fallback if no value column
            
            total_trials_after += len(top_trials)
            
            # Extract hyperparameters from top trials
            param_cols = [col for col in top_trials.columns if col.startswith('params_')]
            
            if param_cols:
                # Initialize param_names from first dataset
                if param_names is None:
                    param_names = [col.replace('params_', '') for col in param_cols]
                
                # Add each trial as a training sample
                for _, trial in top_trials.iterrows():
                    # Features (same for all trials from this dataset)
                    feature_vector = [features.get(fname, 0.0) for fname in feature_names]
                    X_features.append(feature_vector)
                    
                    # Parameters (from this specific trial)
                    param_vector = []
                    for param_name in param_names:
                        param_col = f'params_{param_name}'
                        if param_col in trial:
                            param_vector.append(trial[param_col])
                        else:
                            param_vector.append(0.0)  # Default value if missing
                    y_params.append(param_vector)
                    
                    # Dataset group (for preventing data leakage in cross-validation)
                    dataset_groups.append(i)
        else:
            # Fallback to best_hyperparameters if no trials dataframe
            if 'best_hyperparameters' in result or 'best_params' in result:
                best_params = result.get('best_hyperparameters', result.get('best_params', {}))
                
                if param_names is None:
                    param_names = list(best_params.keys())
                
                # Features
                feature_vector = [features.get(fname, 0.0) for fname in feature_names]
                X_features.append(feature_vector)
                
                # Parameters
                param_vector = [best_params.get(pname, 0.0) for pname in param_names]
                y_params.append(param_vector)
                
                # Dataset group
                dataset_groups.append(i)
                
                total_trials_before += 1
                total_trials_after += 1
    
    if verbose:
        print(f"Extracted {len(X_features)} training samples from {len(results_list)} datasets")
        print(f"Filtered trials: {total_trials_before} → {total_trials_after} (kept top-{top_k_trials} per dataset)")
    
    if not X_features:
        return np.array([]), np.array([]), [], [], np.array([])
    
    return (np.array(X_features), np.array(y_params), feature_names, 
            param_names or [], np.array(dataset_groups))


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
        min_val = float(np.min(feature_values))
        max_val = float(np.max(feature_values))
        range_val = float(max_val - min_val)
        
        # Handle zero-range features (all values are the same)
        if range_val == 0.0:
            # For zero-range features, disable normalization by setting range to 1.0
            # This will result in (value - min) / 1.0 = (value - min), which is safer
            range_val = 1.0
            print(f"⚠️  Zero range detected for {feature_name}, disabling normalization (range=1.0)")
        
        # Ensure range is never too small to avoid numerical issues
        if range_val < 1e-10:
            range_val = 1.0
            print(f"⚠️  Very small range detected for {feature_name}, using range=1.0")
        
        norm_params[feature_name] = {
            'min': min_val,
            'max': max_val,
            'range': range_val
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


def _nmae_scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Custom NMAE scorer for RFECV that computes average NMAE across all parameters.
    Returns negative NMAE since sklearn scorers expect higher=better.
    """
    # Define parameter ranges for normalization (same as in _calculate_parameter_nmae)
    param_ranges = {
        0: (50, 1000),      # n_estimators (index 0)
        1: (1, 16),         # max_depth (index 1) 
        2: (0.01, 0.3),     # learning_rate (index 2)
        3: (0.5, 1.0),      # subsample (index 3)
        4: (0.5, 1.0)       # colsample_bytree (index 4)
    }
    
    if y_pred.ndim == 1:
        # Single output case - shouldn't happen with multi-output regressor
        return -mean_absolute_error(y_true, y_pred)
    
    nmae_values = []
    n_params = min(y_true.shape[1], len(param_ranges))
    
    for i in range(n_params):
        if i in param_ranges:
            min_val, max_val = param_ranges[i]
            
            # Normalize to [0, 1]
            y_true_norm = (y_true[:, i] - min_val) / (max_val - min_val)
            y_pred_norm = (y_pred[:, i] - min_val) / (max_val - min_val)
            
            # Calculate MAE on normalized values
            nmae = mean_absolute_error(y_true_norm, y_pred_norm)
            nmae_values.append(nmae)
    
    # Return negative average NMAE (higher is better for sklearn)
    avg_nmae = np.mean(nmae_values) if nmae_values else 1.0
    return -avg_nmae


def _calculate_top_k_accuracy(y_true: np.ndarray, y_pred: np.ndarray, param_names: List[str], k_values: List[int] = [5]) -> Dict[int, float]:
    """
    Calculate Top-K Parameter Set Accuracy by comparing predicted hyperparameters 
    against random baselines.
    
    Args:
        y_true: True parameter values (n_samples, n_params)
        y_pred: Predicted parameter values (n_samples, n_params)
        param_names: Names of parameters
        k_values: List of numbers of random baselines to generate for comparison
        
    Returns:
        Dictionary mapping k to accuracy
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
    overall_better_than_random_count = 0
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
                for k in k_values:
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
                    overall_better_than_random_count += 1
                
                total_comparisons += 1
            
            # Parameter-specific accuracy
            param_specific_accuracy[param_name] = param_better_count / n_samples
    
    overall_accuracy = overall_better_than_random_count / total_comparisons if total_comparisons > 0 else 0.0
    
    return {k: overall_accuracy for k in k_values}


def _normalize_features(X: np.ndarray, norm_params: Dict[str, Dict[str, float]]) -> np.ndarray:
    """
    Normalize features using the provided normalization parameters.
    
    Args:
        X: Feature matrix
        norm_params: Normalization parameters
        
    Returns:
        Normalized feature matrix
    """
    X_norm = X.copy()
    
    for i, feature_name in enumerate(norm_params.keys()):
        if feature_name in norm_params:
            norm_info = norm_params[feature_name]
            min_val = norm_info['min']
            range_val = norm_info['range']
            
            if range_val > 0:
                X_norm[:, i] = (X_norm[:, i] - min_val) / range_val
            else:
                X_norm[:, i] = 0.0
    
    return X_norm 