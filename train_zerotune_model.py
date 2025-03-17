#!/usr/bin/env python
"""
Script to train Random Forest, XGBoost, and Decision Tree ZeroTune models using synthetic data.

This script:
1. Directly generates synthetic training data for both binary and multi-class problems
2. Trains separate meta-models for binary and multi-class classification
3. Saves the trained models in the proper format for ZeroTune
4. Uses Optuna for efficient hyperparameter optimization

Usage:
    python train_zerotune_model.py                            # Train all models and types
    python train_zerotune_model.py --model xgb                # Train only XGBoost models (all types)
    python train_zerotune_model.py --model dt                 # Train only Decision Tree models (all types)
    python train_zerotune_model.py --model rf                 # Train only Random Forest models (all types)
    python train_zerotune_model.py --type binary              # Train all models for binary classification only
    python train_zerotune_model.py --model xgb --type binary  # Train only XGBoost models for binary classification
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
import joblib
from tqdm import tqdm
from xgboost import XGBRegressor

# Create output directory
os.makedirs("zerotune/models", exist_ok=True)

def generate_synthetic_dataset_features(n_samples=100, classification_type="binary"):
    """Generate synthetic dataset features for training.
    
    Args:
        n_samples: Number of synthetic datasets to generate
        classification_type: Either "binary", "multi-class", or "regression"
    """
    # Initialize the dataset to store meta-features
    data = []
    
    # Range of dataset sizes and features to generate
    n_samples_range = [100, 200, 500, 1000, 2000, 5000]
    n_features_range = [5, 10, 20, 50, 100]
    
    # For multiclass, we'll have 3 to 10 classes
    if classification_type == "binary":
        n_classes_range = [2]
    elif classification_type == "multi-class":
        n_classes_range = [3, 4, 5, 7, 10]
    else:  # regression
        n_classes_range = [0]  # Not used for regression
    
    # Generate a diverse set of datasets
    for _ in range(n_samples):
        n_samples_val = np.random.choice(n_samples_range)
        n_features_val = np.random.choice(n_features_range)
        n_informative = max(5, int(n_features_val * 0.5))  # Ensure at least 5 informative features
        
        if classification_type == "regression":
            # Create a random regression dataset
            X, y = make_regression(
                n_samples=n_samples_val,
                n_features=n_features_val,
                n_informative=n_informative,
                noise=np.random.uniform(0.1, 1.0),
                random_state=np.random.randint(0, 1000)
            )
            n_classes_val = 0  # Not applicable for regression
            imbalance_ratio = 0.0  # Not applicable for regression
        else:
            # For classification (binary or multi-class)
            n_classes_val = np.random.choice(n_classes_range)
            
            # Create a random classification dataset
            X, y = make_classification(
                n_samples=n_samples_val,
                n_features=n_features_val,
                n_informative=n_informative,
                n_redundant=int(n_features_val * 0.1),
                n_classes=n_classes_val,
                n_clusters_per_class=1,  # Use 1 cluster per class to avoid the constraint error
                class_sep=np.random.uniform(0.8, 2.0),
                random_state=np.random.randint(0, 1000)
            )
            imbalance_ratio = 1.0  # We'll use balanced classes for simplicity
        
        # Calculate dataset meta-features
        dataset_features = {
            'n_samples': n_samples_val,
            'n_features': n_features_val,
            'n_classes': n_classes_val,
            'n_highly_target_corr': n_informative,
            'imbalance_ratio': imbalance_ratio
        }
        
        data.append(dataset_features)
    
    return pd.DataFrame(data)

def find_best_hyperparams_for_dataset(features_row, model_type="rf", n_trials=25, timeout=60):
    """Find the best hyperparameters for a given dataset using Optuna.
    
    Args:
        features_row: Row of dataset features
        model_type: Type of model to optimize ('rf', 'xgb', or 'dt')
        n_trials: Number of Optuna trials
        timeout: Timeout in seconds for Optuna optimization
    
    Returns:
        Tuple of (best hyperparameters, best score)
    """
    # Extract dataset characteristics
    n_samples = int(features_row['n_samples'])
    n_features = int(features_row['n_features'])
    n_classes = int(features_row['n_classes'])
    
    # Determine if this is a regression problem
    is_regression = (n_classes == 0)
    
    # Create a synthetic dataset with these characteristics
    n_informative = max(5, int(n_features * 0.5))  # Ensure at least 5 informative features
    
    if is_regression:
        # Generate regression dataset
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            noise=0.5,
            random_state=42
        )
        # For regression, split the data for validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    else:
        # Generate classification dataset
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=int(n_features * 0.1),
            n_classes=n_classes,
            n_clusters_per_class=1,  # Use 1 cluster per class to avoid the constraint error
            class_sep=1.5,
            random_state=42
        )
        # For classification, we can use cross-validation or split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Define the objective function for Optuna based on model type
    if model_type == "dt":
        # Decision Tree objective function
        def objective(trial):
            # Define the hyperparameters to optimize
            max_depth = trial.suggest_int("max_depth", 1, 50, log=True) if trial.suggest_categorical("use_max_depth", [True, False]) else None
            min_samples_split = trial.suggest_float("min_samples_split", 0.01, 0.5) if is_regression else trial.suggest_int("min_samples_split", 2, 20)
            min_samples_leaf = trial.suggest_float("min_samples_leaf", 0.01, 0.5) if is_regression else trial.suggest_int("min_samples_leaf", 1, 20)
            max_features = trial.suggest_float("max_features", 0.3, 1.0) if trial.suggest_categorical("use_max_features", [True, False]) else None
            
            # Create and evaluate the model
            if is_regression:
                model = DecisionTreeRegressor(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=42
                )
                model.fit(X_train, y_train)
                # For regression, use negative MSE (higher is better)
                from sklearn.metrics import mean_squared_error
                y_pred = model.predict(X_val)
                score = -mean_squared_error(y_val, y_pred)
            else:
                model = DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=42
                )
                model.fit(X_train, y_train)
                # For classification, use accuracy or ROC AUC
                if n_classes == 2:
                    from sklearn.metrics import roc_auc_score
                    y_pred_proba = model.predict_proba(X_val)[:, 1]
                    score = roc_auc_score(y_val, y_pred_proba)
                else:
                    from sklearn.metrics import accuracy_score
                    y_pred = model.predict(X_val)
                    score = accuracy_score(y_val, y_pred)
            
            return score
    
    elif model_type == "rf":
        # Random Forest objective function
        def objective(trial):
            # Define the hyperparameters to optimize
            n_estimators = trial.suggest_int("n_estimators", 50, 300)
            max_depth = trial.suggest_int("max_depth", 3, 50, log=True) if trial.suggest_categorical("use_max_depth", [True, False]) else None
            min_samples_split = trial.suggest_float("min_samples_split", 0.01, 0.5) if is_regression else trial.suggest_int("min_samples_split", 2, 20)
            min_samples_leaf = trial.suggest_float("min_samples_leaf", 0.01, 0.5) if is_regression else trial.suggest_int("min_samples_leaf", 1, 20)
            max_features_options = [None, "sqrt", "log2"]
            max_features_str = trial.suggest_categorical("max_features_str", max_features_options)
            max_features = None if max_features_str == "None" else max_features_str
            if max_features is None and trial.suggest_categorical("use_max_features_float", [True, False]):
                max_features = trial.suggest_float("max_features_float", 0.3, 1.0)
            
            # Create and evaluate the model
            if is_regression:
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train, y_train)
                # For regression, use negative MSE (higher is better)
                from sklearn.metrics import mean_squared_error
                y_pred = model.predict(X_val)
                score = -mean_squared_error(y_val, y_pred)
            else:
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train, y_train)
                # For classification, use accuracy or ROC AUC
                if n_classes == 2:
                    from sklearn.metrics import roc_auc_score
                    y_pred_proba = model.predict_proba(X_val)[:, 1]
                    score = roc_auc_score(y_val, y_pred_proba)
                else:
                    from sklearn.metrics import accuracy_score
                    y_pred = model.predict(X_val)
                    score = accuracy_score(y_val, y_pred)
            
            return score
    
    elif model_type == "xgb":
        # XGBoost objective function
        def objective(trial):
            try:
                # Import XGBoost
                from xgboost import XGBClassifier, XGBRegressor
                
                # Define the hyperparameters to optimize
                n_estimators = trial.suggest_int("n_estimators", 50, 300)
                max_depth = trial.suggest_int("max_depth", 3, 15)
                learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
                subsample = trial.suggest_float("subsample", 0.5, 1.0)
                colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
                gamma = trial.suggest_float("gamma", 0, 1.0)
                
                # Create and evaluate the model
                if is_regression:
                    model = XGBRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        gamma=gamma,
                        random_state=42
                    )
                    model.fit(X_train, y_train)
                    # For regression, use negative MSE (higher is better)
                    from sklearn.metrics import mean_squared_error
                    y_pred = model.predict(X_val)
                    score = -mean_squared_error(y_val, y_pred)
                else:
                    model = XGBClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        gamma=gamma,
                        random_state=42,
                        use_label_encoder=False,
                        objective='binary:logistic' if n_classes == 2 else 'multi:softprob',
                        eval_metric='logloss'
                    )
                    model.fit(X_train, y_train)
                    # For classification, use accuracy or ROC AUC
                    if n_classes == 2:
                        from sklearn.metrics import roc_auc_score
                        y_pred_proba = model.predict_proba(X_val)[:, 1]
                        score = roc_auc_score(y_val, y_pred_proba)
                    else:
                        from sklearn.metrics import accuracy_score
                        y_pred = model.predict(X_val)
                        score = accuracy_score(y_val, y_pred)
                
                return score
            except Exception as e:
                print(f"Error in XGBoost optimization: {str(e)}")
                return float('-inf')  # Return worst possible score on error
    
    else:
        # Unknown model type
        print(f"Unknown model type: {model_type}")
        return {}, 0.0
    
    # Run Optuna optimization
    try:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        # Get the best parameters
        best_params = study.best_params
        best_score = study.best_value
        
        # Process the parameters for return
        if model_type == "dt":
            # Handle 'None' values for max_depth and max_features
            if "use_max_depth" in best_params and not best_params["use_max_depth"]:
                best_params["max_depth"] = None
            del best_params["use_max_depth"]
            
            if "use_max_features" in best_params and not best_params["use_max_features"]:
                best_params["max_features"] = None
            del best_params["use_max_features"]
            
        elif model_type == "rf":
            # Handle 'None' values for max_depth and max_features
            if "use_max_depth" in best_params and not best_params["use_max_depth"]:
                best_params["max_depth"] = None
            del best_params["use_max_depth"]
            
            # Process max_features
            if "max_features_str" in best_params:
                if best_params["max_features_str"] == "None":
                    if "use_max_features_float" in best_params and best_params["use_max_features_float"]:
                        best_params["max_features"] = best_params["max_features_float"]
                    else:
                        best_params["max_features"] = None
                else:
                    best_params["max_features"] = best_params["max_features_str"]
                
                del best_params["max_features_str"]
            
            if "use_max_features_float" in best_params:
                del best_params["use_max_features_float"]
            
            if "max_features_float" in best_params:
                del best_params["max_features_float"]
        
        # Return the best hyperparameters and the score
        return best_params, best_score
        
    except Exception as e:
        print(f"Error during optimization: {str(e)}")
        return {}, 0.0

def create_normalization_params(df):
    """Create normalization parameters for the meta-features."""
    normalization_params = {}
    for column in df.columns:
        normalization_params[column] = {
            'min': df[column].min(),
            'max': df[column].max(),
            'range': df[column].max() - df[column].min()
        }
    return normalization_params

def train_and_save_model(X_df, y_df, model_type, classification_type, model_file):
    """Train a meta-model and save it in the right format.
    
    Args:
        X_df: DataFrame with features
        y_df: DataFrame with target variables
        model_type: Type of model ('rf', 'xgb', or 'dt')
        classification_type: Either "binary", "multi-class", or "regression"
        model_file: Filename to save the model
    """
    # Create normalization parameters
    normalization_params = create_normalization_params(X_df)
    
    # Handle NaN values in the target data
    y_df = y_df.fillna(0)  # Replace NaN values with 0
    
    # Convert string parameters to appropriate types
    # For example, convert 'sqrt' or 'log2' to the actual function
    if 'params_max_features_str' in y_df.columns:
        # Create a new column with the converted values
        y_df['params_max_features'] = y_df['params_max_features_str'].apply(
            lambda x: 0 if x is None or x == "None" else (1 if x == "sqrt" else 2)
        )
        # Drop the string column
        y_df = y_df.drop(columns=['params_max_features_str'])
    
    # Select the appropriate model based on model_type
    if model_type == "rf":
        model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)
    elif model_type == "dt":
        model = DecisionTreeRegressor(random_state=42)
    elif model_type == "xgb":
        model = XGBRegressor(random_state=42, objective='reg:squarederror')
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Train the model
    model.fit(X_df, y_df)
    
    # Get the score
    score = model.score(X_df, y_df)
    print(f"Model type: {model_type} - {classification_type}")
    print(f"Model R² score: {score:.4f}")
    print(f"Feature importance: {model.feature_importances_}")
    
    # Create the model data dictionary
    model_data = {
        'model': model,
        'dataset_features': list(X_df.columns),
        'target_params': list(y_df.columns),
        'normalization_params': normalization_params,
        'score': score
    }
    
    # Save the model
    model_path = os.path.join("zerotune/models", model_file)
    joblib.dump(model_data, model_path)
    print(f"Model saved to {model_path}")

def main():
    """Main function to generate data and train models."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train ZeroTune models using synthetic data.')
    parser.add_argument('--model', choices=['dt', 'rf', 'xgb', 'all'], default='all',
                        help='Which model type to train (default: all)')
    parser.add_argument('--type', choices=['binary', 'multi-class', 'regression', 'all'], default='all',
                        help='Which problem type to train for (default: all)')
    parser.add_argument('--n-datasets', type=int, default=20,
                        help='Number of synthetic datasets to generate for each type (default: 20)')
    parser.add_argument('--n-trials', type=int, default=25,
                        help='Number of Optuna trials for hyperparameter optimization (default: 25)')
    parser.add_argument('--timeout', type=int, default=60,
                        help='Timeout in seconds for Optuna optimization (default: 60)')
    args = parser.parse_args()
    
    # Determine which models to train
    if args.model == 'all':
        model_types = ["dt", "rf", "xgb"]
    else:
        model_types = [args.model]
    
    # Determine which problem types to train for
    if args.type == 'all':
        problem_types = ["binary", "multi-class"]
        # Include regression models based on user selection
        if args.model in ["dt", "rf", "xgb"]:
            problem_types.append("regression")
    else:
        problem_types = [args.type]
    
    print(f"Training models: {model_types}")
    print(f"For problem types: {problem_types}")
    print(f"Using {args.n_datasets} synthetic datasets per type")
    print(f"Running {args.n_trials} Optuna trials with {args.timeout}s timeout")
    
    # Train separate models for each specified problem type
    for classification_type in problem_types:
        print(f"\nGenerating synthetic data for {classification_type}")
        
        # Generate dataset features for this classification type
        dataset_features_df = generate_synthetic_dataset_features(
            n_samples=args.n_datasets,  # Use the command line argument
            classification_type=classification_type
        )
        
        for model_type in model_types:
            # Skip if model_type is "xgb" and XGBoost is not installed
            if model_type == "xgb":
                try:
                    import xgboost
                except ImportError:
                    print("XGBoost not installed. Skipping XGBoost model.")
                    continue
            
            print(f"\nTraining {model_type} model for {classification_type}")
            
            # Collect hyperparameter data for each dataset
            target_data = []
            
            # Use tqdm for progress monitoring
            progress_desc = f"{model_type}-{classification_type}"
            for idx, (_, row) in enumerate(tqdm(list(dataset_features_df.iterrows()), desc=progress_desc)):
                # Find best hyperparameters for this dataset
                best_params, best_score = find_best_hyperparams_for_dataset(
                    row, 
                    model_type=model_type,
                    n_trials=args.n_trials,
                    timeout=args.timeout
                )
                
                # Skip if we couldn't find good hyperparameters
                if not best_params:
                    continue
                
                # Add this data point for training
                data_point = row.to_dict()
                for param_name, param_value in best_params.items():
                    data_point[f"params_{param_name}"] = param_value
                
                # Add score information
                data_point["optimization_score"] = best_score
                target_data.append(data_point)
            
            # Convert to DataFrame
            if not target_data:
                print(f"No valid data collected for {model_type} - {classification_type}. Skipping.")
                continue
            
            training_df = pd.DataFrame(target_data)
            
            # Separate features and targets
            feature_columns = ["n_samples", "n_features", "n_highly_target_corr", "imbalance_ratio"]
            
            # Define target parameters based on model type
            if model_type == "dt":
                if classification_type == "regression":
                    target_columns = ["params_max_depth", "params_min_samples_split", 
                                     "params_min_samples_leaf", "params_max_features"]
                    model_file = f"decision_tree_regressor.joblib"
                else:
                    target_columns = ["params_max_depth", "params_min_samples_split", 
                                     "params_min_samples_leaf", "params_max_features"]
                    model_file = f"decision_tree_{classification_type.replace('-', '')}_classifier.joblib"
            elif model_type == "rf":
                if classification_type == "regression":
                    target_columns = ["params_n_estimators", "params_max_depth", "params_min_samples_split", 
                                     "params_min_samples_leaf", "params_max_features_str"]
                    model_file = f"random_forest_regressor.joblib"
                else:
                    target_columns = ["params_n_estimators", "params_max_depth", "params_min_samples_split", 
                                     "params_min_samples_leaf", "params_max_features_str"]
                    model_file = f"random_forest_{classification_type.replace('-', '')}_classifier.joblib"
            elif model_type == "xgb":
                if classification_type == "regression":
                    target_columns = ["params_n_estimators", "params_max_depth", "params_learning_rate",
                                     "params_subsample", "params_colsample_bytree", "params_gamma"]
                    model_file = f"xgboost_regressor.joblib"
                else:
                    target_columns = ["params_n_estimators", "params_max_depth", "params_learning_rate",
                                     "params_subsample", "params_colsample_bytree", "params_gamma"]
                    model_file = f"xgboost_{classification_type.replace('-', '')}_classifier.joblib"
            
            # Filter columns for features and targets
            X_df = training_df[feature_columns]
            
            # Ensure all target columns exist
            for col in target_columns:
                if col not in training_df.columns:
                    print(f"Column {col} not found in training data. Adding default values.")
                    # Add default values
                    if col == "params_n_estimators":
                        training_df[col] = 100
                    elif col == "params_max_depth":
                        training_df[col] = 10
                    elif col == "params_min_samples_split":
                        training_df[col] = 2
                    elif col == "params_min_samples_leaf":
                        training_df[col] = 1
                    elif col == "params_max_features":
                        training_df[col] = 0.7
                    elif col == "params_max_features_str":
                        training_df[col] = "sqrt"  # Default to sqrt for max_features_str
                    elif col == "params_learning_rate":
                        training_df[col] = 0.1
                    elif col == "params_subsample":
                        training_df[col] = 0.9
                    elif col == "params_colsample_bytree":
                        training_df[col] = 0.9
                    elif col == "params_gamma":
                        training_df[col] = 0.1
            
            y_df = training_df[target_columns]
            
            # Train and save the model
            train_and_save_model(X_df, y_df, model_type, classification_type, model_file)

if __name__ == "__main__":
    main() 