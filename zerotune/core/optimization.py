"""
Optimization functions for ZeroTune.

This module handles optimization logic, including hyperparameter tuning
and the core optimization algorithm for ZeroTune.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, r2_score, roc_auc_score


def calculate_performance_score(model, X, y, metric="accuracy"):
    """
    Calculate performance score of a model on a dataset.
    
    Args:
        model: Trained model with predict method
        X: Features for evaluation
        y: Target values for evaluation
        metric (str): Performance metric to use. Options include:
            - "accuracy": Classification accuracy
            - "f1": F1 score (weighted)
            - "mse": Mean squared error (negated for maximization)
            - "rmse": Root mean squared error (negated for maximization)
            - "r2": R-squared score
            - "roc_auc": Area under ROC curve
        
    Returns:
        float: Performance score according to the specified metric
        
    Note:
        For regression metrics (mse, rmse), the score is negated to convert
        it to a maximization problem (higher is better).
    """
    y_pred = model.predict(X)
    
    if metric == "accuracy":
        return accuracy_score(y, y_pred)
    elif metric == "f1":
        return f1_score(y, y_pred, average="weighted")
    elif metric == "mse":
        return -mean_squared_error(y, y_pred)  # Negative for maximization
    elif metric == "rmse":
        return -np.sqrt(mean_squared_error(y, y_pred))  # Negative for maximization
    elif metric == "r2":
        return r2_score(y, y_pred)
    elif metric == "roc_auc":
        # For ROC AUC we need probability predictions
        # Check if model has predict_proba method (most sklearn classifiers do)
        if hasattr(model, 'predict_proba'):
            y_score = model.predict_proba(X)
            # If binary classification, we need the scores for the positive class (column 1)
            if y_score.shape[1] == 2:
                return roc_auc_score(y, y_score[:, 1])
            else:
                # For multi-class, use OVR approach
                return roc_auc_score(y, y_score, multi_class='ovr', average='macro')
        else:
            # Fallback to decision function if available
            if hasattr(model, 'decision_function'):
                y_score = model.decision_function(X)
                return roc_auc_score(y, y_score)
            else:
                # If neither is available, fall back to accuracy
                print("Model doesn't support probability predictions, falling back to accuracy")
                return accuracy_score(y, y_pred)
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def evaluate_configuration(model_class, hyperparameters, X_train, X_val, y_train, y_val, metric="accuracy"):
    """
    Train a model with given hyperparameters and evaluate on validation set.
    
    Args:
        model_class: Class of the ML model to use
        hyperparameters: Dictionary of hyperparameters to use
        X_train: Training features
        X_val: Validation features
        y_train: Training targets
        y_val: Validation targets
        metric: Evaluation metric to use
        
    Returns:
        float: Performance score on validation set
    """
    try:
        model = model_class(**hyperparameters)
        model.fit(X_train, y_train)
        score = calculate_performance_score(model, X_val, y_val, metric)
        return score
    except Exception as e:
        print(f"Error evaluating configuration: {e}")
        return float("-inf")  # Return worst possible score on error


def train_final_model(model_class, best_hyperparameters, X_train, y_train):
    """
    Train final model with best hyperparameters on full training set.
    
    Args:
        model_class: Class of the ML model to use
        best_hyperparameters: Dictionary of best hyperparameters
        X_train: Training features
        y_train: Training targets
        
    Returns:
        Trained model
    """
    model = model_class(**best_hyperparameters)
    model.fit(X_train, y_train)
    return model


def optimize_hyperparameters(model_class, param_grid, X_train, y_train, 
                             metric="accuracy", n_iter=10, test_size=0.2, 
                             random_state=42, verbose=True):
    """
    Simple random search optimization for hyperparameters.
    
    Args:
        model_class: Class of the ML model to use
        param_grid: Dictionary of hyperparameter ranges
        X_train: Training features
        y_train: Training targets
        metric: Evaluation metric
        n_iter: Number of iterations/configurations to try
        test_size: Validation split ratio
        random_state: Random state for reproducibility
        verbose: Whether to print progress
        
    Returns:
        tuple: (best_params, best_score, all_results)
    """
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=test_size, random_state=random_state
    )
    
    best_score = float("-inf")
    best_params = None
    all_results = []
    
    for i in range(n_iter):
        # Sample random hyperparameters
        hyperparameters = {}
        for param, param_range in param_grid.items():
            if isinstance(param_range, list):
                hyperparameters[param] = np.random.choice(param_range)
            elif isinstance(param_range, tuple) and len(param_range) == 3:
                low, high, scale = param_range
                if scale == "linear":
                    hyperparameters[param] = np.random.uniform(low, high)
                elif scale == "log":
                    hyperparameters[param] = np.exp(np.random.uniform(np.log(low), np.log(high)))
                elif scale == "int":
                    hyperparameters[param] = np.random.randint(low, high)
            else:
                raise ValueError(f"Unsupported param_range format for {param}: {param_range}")
        
        # Evaluate configuration
        score = evaluate_configuration(
            model_class, hyperparameters, X_train_split, X_val, y_train_split, y_val, metric
        )
        
        all_results.append({"hyperparameters": hyperparameters, "score": score})
        
        # Update best if improvement
        if score > best_score:
            best_score = score
            best_params = hyperparameters
            if verbose:
                print(f"Iteration {i+1}/{n_iter}: New best score: {best_score}")
                print(f"Parameters: {best_params}")
        elif verbose:
            print(f"Iteration {i+1}/{n_iter}: Score: {score}")
    
    return best_params, best_score, all_results


def find_similar_datasets(target_meta_features, knowledge_base, n_neighbors=3):
    """
    Find similar datasets in the knowledge base based on meta-features.
    
    This function is a key component of ZeroTune's meta-learning approach. It uses
    the nearest neighbors algorithm to identify datasets in the knowledge base that
    are most similar to the target dataset based on their meta-features.
    
    Args:
        target_meta_features (dict): Meta-features of the target dataset, calculated
            using the feature extraction functions.
        knowledge_base (dict): Knowledge base containing dataset meta-features and
            hyperparameter optimization results.
        n_neighbors (int): Number of similar datasets to retrieve. Default is 3.
        
    Returns:
        list: Indices of the most similar datasets in the knowledge base. Returns
            an empty list if no matching datasets are found or if the knowledge base
            is empty.
            
    Note:
        The function uses standardized features and Euclidean distance to calculate
        similarity. Only features present in both the target dataset and knowledge
        base datasets are considered for comparison.
    """
    if not knowledge_base or 'meta_features' not in knowledge_base:
        return []
    
    # Convert meta-features to DataFrame
    kb_meta_features = pd.DataFrame(knowledge_base['meta_features'])
    
    # Create a new row for the target dataset
    target_df = pd.DataFrame([target_meta_features])
    
    # Ensure columns match
    common_cols = list(set(kb_meta_features.columns).intersection(set(target_df.columns)))
    if not common_cols:
        return []
    
    kb_meta_features = kb_meta_features[common_cols]
    target_df = target_df[common_cols]
    
    # Scale the features
    scaler = StandardScaler()
    kb_meta_features_scaled = scaler.fit_transform(kb_meta_features)
    target_scaled = scaler.transform(target_df)
    
    # Use NearestNeighbors to find similar datasets
    nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(kb_meta_features)))
    nn.fit(kb_meta_features_scaled)
    
    distances, indices = nn.kneighbors(target_scaled)
    
    return indices[0]


def retrieve_best_configurations(similar_indices, knowledge_base, model_type='default'):
    """
    Retrieve best configurations from similar datasets.
    
    Args:
        similar_indices (list): Indices of similar datasets
        knowledge_base (dict): Knowledge base with dataset results
        model_type (str): Type of model to retrieve configurations for
        
    Returns:
        list: List of hyperparameter configurations
    """
    if not knowledge_base or 'results' not in knowledge_base:
        return []
    
    best_configs = []
    
    # Filter by model type if specified
    kb_results = knowledge_base['results']
    if model_type != 'default':
        kb_filtered = []
        for result in kb_results:
            if 'model_type' in result and result['model_type'] == model_type:
                kb_filtered.append(result)
        kb_results = kb_filtered
    
    # For each similar dataset
    for idx in similar_indices:
        if idx < len(kb_results):
            # Get the result for this dataset
            result = kb_results[idx]
            
            # Add best configuration to the list
            if 'best_hyperparameters' in result:
                best_configs.append(result['best_hyperparameters'])
    
    return best_configs 