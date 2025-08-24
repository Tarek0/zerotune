"""
Optimization functions for ZeroTune.

This module handles optimization logic, including hyperparameter tuning
and the core optimization algorithm for ZeroTune.
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Union, Any, Type, Callable
import numpy as np
import pandas as pd
import optuna
from optuna.trial import Trial
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, r2_score, roc_auc_score

# Type aliases
ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]
ModelClass = Type[BaseEstimator]
HyperParams = Dict[str, Any]
ParamGrid = Dict[str, Union[List[Any], Tuple[float, float, str]]]
MetricType = Callable[[ArrayLike, ArrayLike], float]


def calculate_performance_score(
    model: BaseEstimator,
    X: ArrayLike,
    y: ArrayLike,
    metric: str = "accuracy"
) -> float:
    """
    Calculate performance score of a model on a dataset.
    
    Args:
        model: Trained model with predict method
        X: Features for evaluation
        y: Target values for evaluation
        metric: Performance metric to use. Options include:
            - "accuracy": Classification accuracy
            - "f1": F1 score (weighted)
            - "mse": Mean squared error (negated for maximization)
            - "rmse": Root mean squared error (negated for maximization)
            - "r2": R-squared score
            - "roc_auc": Area under ROC curve
        
    Returns:
        Performance score according to the specified metric
        
    Note:
        For regression metrics (mse, rmse), the score is negated to convert
        it to a maximization problem (higher is better).
    """
    y_pred = model.predict(X)
    
    if metric == "accuracy":
        return float(accuracy_score(y, y_pred))
    elif metric == "f1":
        return float(f1_score(y, y_pred, average="weighted"))
    elif metric == "mse":
        return float(-mean_squared_error(y, y_pred))  # Negative for maximization
    elif metric == "rmse":
        return float(-np.sqrt(mean_squared_error(y, y_pred)))  # Negative for maximization
    elif metric == "r2":
        return float(r2_score(y, y_pred))
    elif metric == "roc_auc":
        # For ROC AUC we need probability predictions
        # Check if model has predict_proba method (most sklearn classifiers do)
        if hasattr(model, 'predict_proba'):
            y_score = model.predict_proba(X)
            # If binary classification, we need the scores for the positive class (column 1)
            if y_score.shape[1] == 2:
                return float(roc_auc_score(y, y_score[:, 1]))
            else:
                # For multi-class, use OVR approach
                return float(roc_auc_score(y, y_score, multi_class='ovr', average='macro'))
        else:
            # Fallback to decision function if available
            if hasattr(model, 'decision_function'):
                y_score = model.decision_function(X)
                return float(roc_auc_score(y, y_score))
            else:
                # If neither is available, fall back to accuracy
                print("Model doesn't support probability predictions, falling back to accuracy")
                return float(accuracy_score(y, y_pred))
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def evaluate_configuration(
    model_class: ModelClass,
    hyperparameters: HyperParams,
    X_train: ArrayLike,
    X_val: ArrayLike,
    y_train: ArrayLike,
    y_val: ArrayLike,
    metric: str = "accuracy"
) -> float:
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
        Performance score on validation set
    """
    try:
        model = model_class(**hyperparameters)
        model.fit(X_train, y_train)
        score = calculate_performance_score(model, X_val, y_val, metric)
        return score
    except Exception as e:
        print(f"Error evaluating configuration: {e}")
        return float("-inf")  # Return worst possible score on error


def train_final_model(
    model_class: ModelClass,
    best_hyperparameters: HyperParams,
    X_train: ArrayLike,
    y_train: ArrayLike
) -> BaseEstimator:
    """
    Train final model with best hyperparameters on full training set.
    
    Args:
        model_class: Class of the ML model to use
        best_hyperparameters: Dictionary of best hyperparameters
        X_train: Training features
        y_train: Training targets
        
    Returns:
        Trained model instance
    """
    model = model_class(**best_hyperparameters)
    model.fit(X_train, y_train)
    return model


def _convert_relative_to_absolute_params(
    params: HyperParams,
    param_grid: ParamGrid,
    dataset_meta_params: Dict[str, Any]
) -> HyperParams:
    """
    Convert relative parameters to absolute values based on dataset meta-parameters.
    
    Args:
        params: Dictionary of relative parameter values
        param_grid: Parameter grid configuration
        dataset_meta_params: Dictionary of dataset meta-parameters
        
    Returns:
        Dictionary of absolute parameter values
    """
    absolute_params = params.copy()
    
    for param, value in params.items():
        if param in param_grid:
            if isinstance(param_grid[param], dict):
                config = param_grid[param]
                if 'dependency' in config and 'multiplier' in config:
                    dependency = config['dependency']
                    multiplier = config['multiplier']
                    if dependency in dataset_meta_params:
                        base_value = dataset_meta_params[dependency]
                        if config.get('param_type') == 'int':
                            absolute_params[param] = int(value * base_value * multiplier)
                        else:
                            absolute_params[param] = value * base_value * multiplier
            elif isinstance(param_grid[param], list):
                # For categorical parameters, find the closest value in the allowed choices
                choices = param_grid[param]
                if isinstance(value, (int, float)):
                    # Find the closest value in the choices
                    closest_value = min(choices, key=lambda x: abs(float(x) - float(value)))
                    absolute_params[param] = closest_value
                else:
                    # For non-numeric values, keep as is if in choices, otherwise use first choice
                    absolute_params[param] = value if value in choices else choices[0]
            elif isinstance(param_grid[param], tuple) and len(param_grid[param]) == 3:
                # For range parameters, clip to the allowed range
                low, high, _ = param_grid[param]
                if isinstance(value, (int, float)):
                    absolute_params[param] = max(low, min(high, value))
    
    return absolute_params


def _optuna_objective(
    trial: Trial,
    model_class: ModelClass,
    param_grid: ParamGrid,
    X_train: ArrayLike,
    X_val: ArrayLike,
    y_train: ArrayLike,
    y_val: ArrayLike,
    metric: str,
    dataset_meta_params: Optional[Dict[str, Any]] = None,
    is_warm_start: bool = False,
    model_random_state: Optional[int] = None
) -> float:
    """
    Objective function for Optuna optimization.
    
    Args:
        trial: Optuna trial object
        model_class: Class of the ML model to use
        param_grid: Dictionary of hyperparameter ranges
        X_train: Training features
        X_val: Validation features
        y_train: Training targets
        y_val: Validation targets
        metric: Evaluation metric to use
        dataset_meta_params: Optional dictionary of dataset meta-parameters for relative parameter conversion
        is_warm_start: Whether this is a warm-start trial
        model_random_state: Optional random state for model initialization
        
    Returns:
        Performance score for the trial
    """
    # For warm-start trials, use the provided parameters directly
    if is_warm_start:
        # Get warmstart parameters from system attributes (where they're actually stored)
        if hasattr(trial, 'system_attrs') and 'fixed_params' in trial.system_attrs:
            hyperparameters = trial.system_attrs['fixed_params']
        else:
            # Fallback to trial.params (might be empty for enqueued trials)
            hyperparameters = trial.params
    else:
        # Sample hyperparameters based on parameter grid
        hyperparameters: HyperParams = {}
        for param, param_range in param_grid.items():
            if isinstance(param_range, list):
                hyperparameters[param] = trial.suggest_categorical(param, param_range)
            elif isinstance(param_range, tuple) and len(param_range) == 3:
                low, high, scale = param_range
                if scale == "linear":
                    hyperparameters[param] = trial.suggest_float(param, low, high)
                elif scale == "log":
                    hyperparameters[param] = trial.suggest_float(param, low, high, log=True)
                elif scale == "int":
                    hyperparameters[param] = trial.suggest_int(param, int(low), int(high))
            else:
                raise ValueError(f"Unsupported param_range format for {param}: {param_range}")
    
    # Convert relative parameters to absolute if dataset meta-parameters are provided
    if dataset_meta_params is not None:
        hyperparameters = _convert_relative_to_absolute_params(
            hyperparameters, param_grid, dataset_meta_params
        )
    
    # Add model random state for consistency with benchmark evaluation
    if model_random_state is not None:
        hyperparameters['random_state'] = model_random_state
    
    # Evaluate configuration
    return evaluate_configuration(
        model_class, hyperparameters, X_train, X_val, y_train, y_val, metric
    )


def optimize_hyperparameters(
    model_class: ModelClass,
    param_grid: ParamGrid,
    X_train: ArrayLike,
    y_train: ArrayLike,
    metric: str = "accuracy",
    n_iter: int = 10,
    test_size: float = 0.2,
    random_state: Optional[int] = 42,
    verbose: bool = True,
    warm_start_configs: Optional[List[HyperParams]] = None,
    dataset_meta_params: Optional[Dict[str, Any]] = None,
    model_random_state: Optional[int] = None
) -> Tuple[HyperParams, float, List[Dict[str, Any]], 'pd.DataFrame']:
    """
    Optimize hyperparameters using Optuna with warm-starting capabilities.
    
    Args:
        model_class: Class of the ML model to use
        param_grid: Dictionary of hyperparameter ranges
        X_train: Training features
        y_train: Training targets
        metric: Evaluation metric to use
        n_iter: Number of optimization trials
        test_size: Validation split ratio
        random_state: Random state for reproducibility
        verbose: Whether to print progress
        warm_start_configs: Optional list of hyperparameter configurations to warm-start from
        dataset_meta_params: Optional dictionary of dataset meta-parameters for relative parameter conversion
        model_random_state: Optional random state for model initialization (for consistency with benchmark)
        
    Returns:
        A tuple containing:
            - best_params: Dictionary of best hyperparameters found
            - best_score: Best performance score achieved
            - all_results: List of all configurations and their scores
            - df_trials: DataFrame with all Optuna trials
    """
    # Split data for training and validation using stratified splitting
    # This ensures both classes are present in train and test sets
    stratified_split = StratifiedShuffleSplit(
        n_splits=1, 
        test_size=test_size, 
        random_state=random_state
    )
    
    train_idx, val_idx = next(stratified_split.split(X_train, y_train))
    
    # Handle both pandas DataFrames/Series and numpy arrays
    if hasattr(X_train, 'iloc'):
        # Pandas DataFrame
        X_train_split, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    else:
        # Numpy array
        X_train_split, X_val = X_train[train_idx], X_train[val_idx]
    
    if hasattr(y_train, 'iloc'):
        # Pandas Series
        y_train_split, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    else:
        # Numpy array
        y_train_split, y_val = y_train[train_idx], y_train[val_idx]
    
    if verbose:
        print(f"Running optimization with {n_iter} trials")
    
    # Create Optuna study
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    
    # Add warm-start configurations if provided
    if warm_start_configs:
        if verbose:
            print(f"Warm-starting optimization with {len(warm_start_configs)} configurations")
        
        # Convert warm-start configurations to match parameter grid
        converted_configs = []
        for config in warm_start_configs:
            if dataset_meta_params is not None:
                converted_config = _convert_relative_to_absolute_params(
                    config, param_grid, dataset_meta_params
                )
            else:
                converted_config = _convert_relative_to_absolute_params(
                    config, param_grid, {}
                )
            converted_configs.append(converted_config)
        
        # Enqueue converted configurations
        for config in converted_configs:
            study.enqueue_trial(config)
    
    # Create objective function with warm-start flag tracking
    warm_start_count = len(warm_start_configs) if warm_start_configs else 0
    
    def objective(trial: Trial) -> float:
        is_warm_start = trial.number < warm_start_count
        return _optuna_objective(
            trial,
            model_class,
            param_grid,
            X_train_split,
            X_val,
            y_train_split,
            y_val,
            metric,
            dataset_meta_params,
            is_warm_start,
            model_random_state
        )
    
    # Run optimization
    study.optimize(objective, n_trials=n_iter, show_progress_bar=verbose)
    
    # Collect results
    all_results = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            result = {
                "hyperparameters": trial.params,
                "score": trial.value,
                "state": trial.state.name,
                "is_warm_start": trial.number < warm_start_count
            }
            all_results.append(result)
    
    # Get trials dataframe
    df_trials = study.trials_dataframe()
    
    # Get best results
    best_params = study.best_params
    best_score = study.best_value
    
    # Convert relative parameters to absolute if needed
    if dataset_meta_params is not None:
        best_params = _convert_relative_to_absolute_params(
            best_params, param_grid, dataset_meta_params
        )
    
    if verbose:
        print(f"Best score: {best_score:.4f}")
        print("Best hyperparameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
    
    return best_params, best_score, all_results, df_trials
 