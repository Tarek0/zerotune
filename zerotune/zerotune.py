"""ZeroTune: A module for one-shot hyperparameter optimization.

This module provides functionality for zero-shot hyperparameter optimization using
meta-learning approaches. It includes utilities for dataset meta-parameter calculation,
hyperparameter optimization with Optuna, and model evaluation.
"""

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import random
import optuna
from typing import Dict, List, Tuple, Union, Optional, Any
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from scipy.stats import randint as sp_randint
import warnings
import logging

# Set up Optuna's logging level
optuna.logging.set_verbosity(optuna.logging.ERROR)
warnings.filterwarnings("ignore")

# ===================== Dataset Meta-Features Calculation =====================

def calculate_dataset_meta_parameters(X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """Calculate meta-parameters (features) of a dataset.
    
    Args:
        X: The feature data.
        y: The target data.
    
    Returns:
        A dictionary of meta-parameters.
    """
    meta_parameters = {}
    meta_parameters.update(_calculate_dataset_size(X))
    meta_parameters.update(_calculate_class_imbalance_ratio(y))
    meta_parameters.update(_calculate_correlation_metrics(X, y, correlation_cutoff=0.10))
    meta_parameters.update(_calculate_feature_moments_and_variances(X))
    meta_parameters.update(_calculate_row_moments_and_variances(X))
    
    return meta_parameters

def _calculate_dataset_size(X: pd.DataFrame) -> Dict[str, int]:
    """Calculate the size parameters of a dataset.
    
    Args:
        X: The feature data.
    
    Returns:
        Dictionary containing n_samples and n_features.
    """
    return {"n_samples": X.shape[0], "n_features": X.shape[1]}

def _calculate_class_imbalance_ratio(y: pd.Series) -> Dict[str, float]:
    """Calculate the class imbalance ratio of a dataset.

    Args:
        y: Target values (class labels).

    Returns:
        Dictionary containing the ratio of the majority class size to the minority class size.
    """
    # Count the occurrences of each class
    class_counts = np.bincount(y)

    # Find the counts of majority and minority classes
    majority_class_count = np.max(class_counts)
    minority_class_count = np.min(class_counts)

    # Calculate the imbalance ratio
    imbalance_ratio = majority_class_count / minority_class_count

    return {"imbalance_ratio": imbalance_ratio}

def _calculate_correlation_metrics(X: pd.DataFrame, y: pd.Series, correlation_cutoff: float = 0.1) -> Dict[str, float]:
    """Calculate correlation metrics between features and target variable.

    Args:
        X: The input features.
        y: The target variable.
        correlation_cutoff: Minimum absolute correlation threshold.

    Returns:
        Dictionary containing correlation statistics.
    """
    df = pd.DataFrame(X.copy())
    df['target'] = pd.Series(y.copy())
    correlation_matrix = df.corr(method='pearson')
    correlations_with_target = abs(correlation_matrix['target'])

    informative_features = correlations_with_target[correlations_with_target > correlation_cutoff].sort_values(ascending=False)
    n_informative = len(informative_features) - 1

    return {'n_highly_target_corr': n_informative,
            'avg_target_corr': correlations_with_target.mean(),
            'var_target_corr': correlations_with_target.var()}

def _calculate_feature_moments_and_variances(X: pd.DataFrame) -> Dict[str, float]:
    """Calculate statistical moments for each feature.

    Args:
        X: The input features.

    Returns:
        Dictionary containing moment statistics across features.
    """
    # Converting to DataFrame if not already
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # Calculate moments for each column
    moment_1 = X.apply(lambda x: x.mean(), axis=0)
    moment_2 = X.apply(lambda x: x.var(), axis=0)
    moment_3 = X.apply(lambda x: skew(x.dropna()), axis=0)
    moment_4 = X.apply(lambda x: kurtosis(x.dropna()), axis=0)

    # Calculate the averages and variances
    moments = {'avg_feature_m1': moment_1.mean(),
              'var_feature_m1': moment_1.var(),
              'avg_feature_m2': moment_2.mean(),
              'var_feature_m2': moment_2.var(),
              'avg_feature_m3': moment_3.mean(),
              'var_feature_m3': moment_3.var(),
              'avg_feature_m4': moment_4.mean(),
              'var_feature_m4': moment_4.var()}

    return moments

def _calculate_row_moments_and_variances(X: pd.DataFrame) -> Dict[str, float]:
    """Calculate statistical moments for each sample (row).

    Args:
        X: The input features.

    Returns:
        Dictionary containing moment statistics across rows.
    """
    # Converting to DataFrame if not already
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # Calculate moments for each row
    moment_1 = X.apply(lambda x: x.mean(), axis=1)
    moment_2 = X.apply(lambda x: x.var(), axis=1)
    moment_3 = X.apply(lambda x: skew(x.dropna()), axis=1)
    moment_4 = X.apply(lambda x: kurtosis(x.dropna()), axis=1)

    # Calculate the averages and variances
    moments = {'avg_row_m1': moment_1.mean(),
              'var_row_m1': moment_1.var(),
              'avg_row_m2': moment_2.mean(),
              'var_row_m2': moment_2.var(),
              'avg_row_m3': moment_3.mean(),
              'var_row_m3': moment_3.var(),
              'avg_row_m4': moment_4.mean(),
              'var_row_m4': moment_4.var()}

    return moments

# ===================== Hyperparameter Transformation =====================

def relative2absolute_dict(param_config: Dict, dataset_properties: Dict, param_dict: Dict) -> Dict:
    """Convert relative parameter values to absolute values based on dataset properties.
    
    Args:
        param_config: Parameter configuration with dependencies.
        dataset_properties: Dataset properties (e.g., n_samples).
        param_dict: Dictionary of relative parameter values.
        
    Returns:
        Dictionary of absolute parameter values.
    """
    # Create a copy of the param_dict to avoid modifying the original
    absolute_param_dict = param_dict.copy()
    
    params_with_dependency = [param for param, details in param_config.items() if 'dependency' in details]
    for p in params_with_dependency:
        dependency_col = param_config[p]['dependency']
        dependency_value = dataset_properties[dependency_col]
        absolute_param_dict[p] = max(int(dependency_value * absolute_param_dict[p]), 1)
        
    return absolute_param_dict

def generate_random_params(param_config: Dict, random_seed: int) -> Dict:
    """Generate random hyperparameters according to parameter configuration.
    
    Args:
        param_config: Configuration defining parameter ranges.
        random_seed: Random seed for reproducibility.
        
    Returns:
        Dictionary of randomly generated parameter values.
    """
    random_params = {}
    
    random.seed(random_seed)
    for param, config in param_config.items():
        min_value = min(config['percentage_splits'])
        max_value = max(config['percentage_splits'])
     
        random_params[param] = random.uniform(min_value, max_value)
    
    return random_params

# ===================== Model Evaluation =====================

def evaluate_model(X: pd.DataFrame, y: pd.Series, model_name: str, hyperparams: Dict, 
                   random_seed: int = 42, n_folds: int = 3, n_seeds: int = 20) -> Tuple[float, List[float]]:
    """Evaluate model performance with given hyperparameters.
    
    Args:
        X: Feature data.
        y: Target data.
        model_name: Model name (e.g. "DecisionTreeClassifier").
        hyperparams: Hyperparameters for the model.
        random_seed: Base random seed.
        n_folds: Number of cross-validation folds.
        n_seeds: Number of seeds to evaluate.
        
    Returns:
        Tuple of (mean performance score, list of individual seed scores).
    """
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    
    seed_scores = []

    # Determine if we have a multi-class problem (more than 2 unique values)
    n_classes = len(np.unique(y))
    if n_classes > 2:
        # For multi-class, use 'ovr' (one-vs-rest) ROC AUC scoring
        scoring = 'roc_auc_ovr'
    else:
        # For binary classification, use standard ROC AUC
        scoring = 'roc_auc'

    for i in range(n_seeds):
        seed = random_seed + i
        if model_name == "DecisionTreeClassifier2Param" or model_name == "DecisionTreeClassifier4Param":
            model = DecisionTreeClassifier(random_state=seed, **hyperparams)
        elif model_name == "RandomForestClassifier":
            model = RandomForestClassifier(random_state=seed, **hyperparams)
        elif model_name == "XGBClassifier":
            from xgboost import XGBClassifier
            model = XGBClassifier(random_state=seed, **hyperparams)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        try:
            scores = cross_val_score(model, X, y, scoring=scoring, cv=cv, n_jobs=-1)
            seed_scores.append(np.mean(scores))
        except Exception as e:
            print(f"Warning: ROC AUC scoring failed, falling back to accuracy: {str(e)}")
            # Fallback to accuracy if ROC AUC fails
            scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
            seed_scores.append(np.mean(scores))

    final_score = np.mean(seed_scores)
    return final_score, seed_scores

# ===================== HPO with Optuna =====================

def optuna_objective(trial, X, y, param_config, meta_params, dataset_meta_params, model_name="DecisionTreeClassifier4Param", random_seed=42):
    """Objective function for Optuna optimization.
    
    Args:
        trial: Optuna trial.
        X: Feature data.
        y: Target data.
        param_config: Parameter configuration.
        meta_params: Meta-parameters to use.
        dataset_meta_params: Dataset meta-parameters.
        model_name: Model name.
        random_seed: Random seed.
        
    Returns:
        Score of the model.
    """
    # Generate hyperparameters based on the trial
    hyperparams = {}
    for param, config in param_config.items():
        if 'percentage_splits' in config:
            min_value = min(config['percentage_splits'])
            max_value = max(config['percentage_splits'])
            hyperparams[param] = trial.suggest_uniform(param, min_value, max_value)

    predicted_hyperparams = relative2absolute_dict(param_config, dataset_meta_params, hyperparams)
    
    score, _ = evaluate_model(X, y, model_name, predicted_hyperparams, random_seed, n_seeds=1)
    return score

def optuna_hpo(X: pd.DataFrame, y: pd.Series, meta_params: List[str], param_config: Dict, 
               zerotune_params: Optional[Dict] = None, n_trials: int = 100, 
               n_seeds: int = 20, seed: Optional[int] = None, 
               model_name: str = "DecisionTreeClassifier4Param") -> Dict:
    """Run hyperparameter optimization with Optuna.
    
    Args:
        X: Feature data.
        y: Target data.
        meta_params: List of meta-parameters to use.
        param_config: Parameter configuration.
        zerotune_params: Initial parameters from ZeroTune (for warm start).
        n_trials: Number of Optuna trials.
        n_seeds: Number of random seeds to evaluate.
        seed: Specific seed to use (if None, use range(n_seeds)).
        model_name: Model name.
        
    Returns:
        Dictionary with optimization results.
    """
    all_dataset_meta_params = calculate_dataset_meta_parameters(X, y)
    dataset_meta_params = {key: all_dataset_meta_params[key] for key in meta_params}
    dataset_meta_params_inc_dependencies = {key: all_dataset_meta_params[key] for key in ['n_samples']}
    dataset_meta_params_inc_dependencies.update(dataset_meta_params)
    
    results = []
    best_perfs = []
    
    # Adjust the seeds to process
    seeds_to_process = [seed] if seed is not None else range(n_seeds)
    
    for seed_value in seeds_to_process:
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed_value))
        
        if zerotune_params:
            # Enqueue the ZeroTune hyperparameters trial
            study.enqueue_trial(zerotune_params)
        
        study.optimize(
            lambda trial: optuna_objective(
                trial, X, y, param_config, meta_params, 
                dataset_meta_params_inc_dependencies, model_name=model_name, random_seed=seed_value
            ), 
            n_trials=n_trials
        )
        
        best_hyperparams = study.best_params
        best_perf = study.best_value
        
        # Get trials DataFrame and add seed column
        df_trials = study.trials_dataframe()
        df_trials['seed'] = seed_value
        
        result = {
            "best_hyperparams": best_hyperparams,
            "best_perf": best_perf,
            "df_trials": df_trials
        }
        results.append(result)
        best_perfs.append(best_perf)
    
    # Calculate the average of best_perf across all seeds
    average_best_perf = sum(best_perfs) / len(best_perfs)
    
    # Combine all results into a single DataFrame for analysis
    combined_trials_df = pd.concat([result['df_trials'] for result in results], ignore_index=True)
    
    return {
        "all_results": results,
        "average_best_perf": average_best_perf,
        "n_seed_scores": best_perfs,
        "combined_trials_df": combined_trials_df
    }

# ===================== Random Hyperparameter Evaluation =====================

def random_hyperparameter_evaluation(X: pd.DataFrame, y: pd.Series, meta_params: List[str], 
                                    param_config: Dict, model_name: str,
                                    random_seed: int = 42, n_seeds: int = 20) -> Tuple[Dict, float, List[float]]:
    """Evaluate model with randomly generated hyperparameters.
    
    Args:
        X: Feature data.
        y: Target data.
        meta_params: List of meta-parameters to use.
        param_config: Parameter configuration.
        model_name: Model name.
        random_seed: Random seed.
        n_seeds: Number of seeds to evaluate.
        
    Returns:
        Tuple of (hyperparameters, performance score, individual seed scores).
    """
    all_dataset_meta_params = calculate_dataset_meta_parameters(X, y)
    
    dataset_meta_params = {key: all_dataset_meta_params[key] for key in meta_params}
    dataset_meta_params_inc_dependencies = {key: all_dataset_meta_params[key] for key in ['n_samples']}
    dataset_meta_params_inc_dependencies.update(dataset_meta_params)
    
    all_rand_perfs = []
    all_rand_scores = []
    
    for seed in range(n_seeds):
        # Generate random parameters for this seed
        rand_hyperparams = generate_random_params(param_config, random_seed + seed)
        rand_hyperparams_abs = relative2absolute_dict(param_config, dataset_meta_params_inc_dependencies, rand_hyperparams)
        
        # Evaluate the model for this seed
        rand_perf, rand_scores = evaluate_model(X, y, model_name, rand_hyperparams_abs, random_seed=random_seed + seed, n_seeds=1)
        
        # Store the performance and scores
        all_rand_perfs.append(rand_perf)
        all_rand_scores.extend(rand_scores)
    
    # Calculate the average performance across all seeds
    avg_rand_perf = np.mean(all_rand_perfs)
    
    return rand_hyperparams, avg_rand_perf, all_rand_scores

# ===================== ZeroTune Training =====================

def train_zerotune_model(df: pd.DataFrame, dataset_features: List[str], targets: List[str], 
                        condition_column: Optional[str] = None, n_iter: int = 100) -> Tuple[RandomForestRegressor, float, Dict]:
    """Train a ZeroTune model using a Random Forest Multi-Output Regressor with normalized targets.

    Args:
        df: DataFrame containing the dataset.
        dataset_features: Column names to be used as model features.
        targets: Target column names in df.
        condition_column: Column name to be used for defining groups in cross-validation.
        n_iter: Number of random search iterations.

    Returns:
        Tuple of (fitted RandomForestRegressor model, normalized MSE score, normalization parameters).
    """
    # Clean the data: remove any rows with NaN values in features or targets
    clean_df = df.copy()
    feature_cols = dataset_features.copy()
    target_cols = targets.copy()
    
    if condition_column is not None:
        feature_cols.append(condition_column)
    
    # Check for NaN values
    nan_rows = clean_df[feature_cols + target_cols].isna().any(axis=1)
    if nan_rows.any():
        print(f"Warning: Removing {nan_rows.sum()} rows with NaN values")
        clean_df = clean_df[~nan_rows]
    
    X = clean_df[dataset_features]
    y_original = clean_df[targets]
    
    # Normalize target values to [0,1] range to provide a fairer MSE calculation
    # across hyperparameters with different scales
    normalization_params = {}
    y_normalized = pd.DataFrame()
    
    for col in targets:
        # Calculate min and max for each target parameter
        min_val = y_original[col].min()
        max_val = y_original[col].max()
        range_val = max_val - min_val
        
        # Store normalization parameters
        normalization_params[col] = {
            'min': min_val,
            'max': max_val,
            'range': range_val
        }
        
        # Normalize the target parameter to [0,1] range
        # Handle the case where min == max (no range)
        if range_val > 0:
            y_normalized[col] = (y_original[col] - min_val) / range_val
        else:
            y_normalized[col] = 0.5  # If all values are the same, use 0.5 as normalized value
    
    param_dist = {
        'n_estimators': sp_randint(100, 300),
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': sp_randint(2, 11),
        'min_samples_leaf': sp_randint(1, 5),
        'max_features': ['sqrt', 'log2', None],  # Removed 'auto' which is no longer valid
        'bootstrap': [True, False]
    }

    # Initialize the regressor
    regressor = RandomForestRegressor(random_state=42)
    
    # Choose the cross-validation strategy
    if condition_column is None:
        cv_strategy = 4
    else:
        groups = clean_df[condition_column]
        cv_strategy = GroupKFold(n_splits=4)
    
    # Set up RandomizedSearchCV with the chosen cross-validation strategy
    hpo_search = RandomizedSearchCV(regressor, param_distributions=param_dist, cv=cv_strategy,
                                   scoring='neg_mean_squared_error', n_jobs=4, n_iter=n_iter, random_state=42)
    
    # Fit the model on normalized targets
    if condition_column is None:
        hpo_search.fit(X, y_normalized)
    else:
        hpo_search.fit(X, y_normalized, groups=groups)

    # Best parameters and normalized score
    best_params = hpo_search.best_params_
    normalized_score = hpo_search.best_score_

    # Calculate raw (unnormalized) MSE for reference
    best_regressor = RandomForestRegressor(**best_params, random_state=42)
    best_regressor.fit(X, y_normalized)
    
    # Make normalized predictions
    y_pred_norm = best_regressor.predict(X)
    
    # Denormalize predictions for raw MSE calculation
    y_pred_raw = np.zeros_like(y_pred_norm)
    for i, col in enumerate(targets):
        params = normalization_params[col]
        if params['range'] > 0:
            y_pred_raw[:, i] = y_pred_norm[:, i] * params['range'] + params['min']
        else:
            y_pred_raw[:, i] = params['min']
    
    # Calculate raw MSE
    raw_mse = np.mean((y_original.values - y_pred_raw) ** 2)
    
    print(f'Zero-shot predictor normalized MSE: {-normalized_score:.4f}')
    print(f'Zero-shot predictor raw MSE: {raw_mse:.4f}')

    # Train the final model on the entire dataset with best parameters
    best_regressor = RandomForestRegressor(**best_params, random_state=42)
    best_regressor.fit(X, y_normalized)
    
    return best_regressor, normalized_score, normalization_params

def predict_hyperparameters(model: RandomForestRegressor, X: pd.DataFrame, target_columns: List[str], 
                           normalization_params: Dict) -> Dict[str, float]:
    """Predict hyperparameters using a trained ZeroTune model and denormalize the predictions.
    
    Args:
        model: Trained RandomForestRegressor model.
        X: Feature data to predict for.
        target_columns: Names of the target hyperparameters.
        normalization_params: Parameters used for normalization during training.
        
    Returns:
        Dictionary mapping target column names to their denormalized predicted values.
    """
    # Make predictions using the trained model (normalized values)
    predictions = model.predict(X)
    
    # Create a dictionary for denormalized predictions
    predictions_dict = {}
    
    # Denormalize the predictions
    for i, column in enumerate(target_columns):
        if column in normalization_params:
            params = normalization_params[column]
            # Denormalize the prediction
            if params['range'] > 0:
                pred_value = predictions[0][i] * params['range'] + params['min']
            else:
                pred_value = params['min']
            predictions_dict[column] = pred_value
        else:
            # Fallback if normalization parameters not available for this column
            predictions_dict[column] = predictions[0][i]
    
    return predictions_dict

def remove_param_prefix(param_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Remove 'params_' prefix from parameter names.
    
    Args:
        param_dict: Dictionary with parameter names.
        
    Returns:
        Dictionary with 'params_' prefix removed from keys.
    """
    return {key.replace('params_', ''): value for key, value in param_dict.items()} 