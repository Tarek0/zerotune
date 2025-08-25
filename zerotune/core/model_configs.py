"""
Model configuration classes for ZeroTune.

This module contains the configurations for different machine learning models
that can be optimized using ZeroTune's meta-learning approach.
"""

from typing import Dict, List, Union, Any, Optional
import random
import math
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Type aliases
ModelConfig = Dict[str, Any]
ParamConfig = Dict[str, Dict[str, Union[List[float], str, float]]]


class ModelConfigs:
    """
    Configuration class for model hyperparameters.
    
    This class provides static methods to retrieve standardized configurations
    for different machine learning models, including default instances and
    hyperparameter search spaces.
    """
    
    @staticmethod
    def get_decision_tree_config() -> ModelConfig:
        """
        Returns configuration for DecisionTreeClassifier.
        
        The configuration includes the model instance, metric for evaluation,
        and parameter configuration with percentage-based splits for different
        hyperparameters.
        
        Returns:
            Configuration dictionary with keys:
                - name: Name of the model
                - model: Instance of DecisionTreeClassifier
                - metric: Evaluation metric to use
                - param_config: Dictionary of hyperparameters and their search ranges
        """
        return {
            "name": "DecisionTreeClassifier",
            "model": DecisionTreeClassifier(random_state=42),
            'metric': 'roc_auc',
            'param_config': {
                'max_depth': {'percentage_splits': [0.25, 0.50, 0.70, 0.8, 0.9, 0.999], 'param_type': "float", 'dependency': 'n_samples'},
                'min_samples_split': {'min_value': 0.01, 'max_value': 0.5, 'param_type': "float"},  # 1% to 50% of samples
                'min_samples_leaf': {'min_value': 0.005, 'max_value': 0.2, 'param_type': "float"},  # 0.5% to 20% of samples
                'max_features': {'min_value': 0.1, 'max_value': 1.0, 'param_type': "float"}  # 10% to 100% of features
            }
        }
    
    @staticmethod
    def get_random_forest_config() -> ModelConfig:
        """
        Returns configuration for RandomForestClassifier.
        
        The configuration includes the model instance, metric for evaluation,
        and parameter configuration with percentage-based splits for different
        hyperparameters.
        
        Returns:
            Configuration dictionary with keys:
                - name: Name of the model
                - model: Instance of RandomForestClassifier
                - metric: Evaluation metric to use
                - param_config: Dictionary of hyperparameters and their search ranges
        """
        return {
            "name": "RandomForestClassifier",
            "model": RandomForestClassifier(random_state=42),
            'metric': 'roc_auc',
            'param_config': {
                'n_estimators': {'min_value': 10, 'max_value': 250, 'param_type': "int"},
                'max_depth': {'percentage_splits': [0.25, 0.50, 0.70, 0.8, 0.9, 0.999], 'param_type': "float", 'dependency': 'n_samples'},
                'min_samples_split': {'min_value': 0.01, 'max_value': 0.20, 'param_type': "float"},
                'min_samples_leaf': {'min_value': 0.005, 'max_value': 0.10, 'param_type': "float"},
                'max_features': {'min_value': 0.10, 'max_value': 0.99, 'param_type': "float"}
            }
        }
    
    @staticmethod
    def get_xgboost_config() -> ModelConfig:
        """
        Returns configuration for XGBClassifier.
        
        The configuration includes the model instance, metric for evaluation,
        and parameter configuration with percentage-based splits for different
        hyperparameters.
        
        Returns:
            Configuration dictionary with keys:
                - name: Name of the model
                - model: Instance of XGBClassifier
                - metric: Evaluation metric to use
                - param_config: Dictionary of hyperparameters and their search ranges
        """
        return {
            "name": "XGBClassifier",
            "model": XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                enable_categorical=True
            ),
            'metric': 'roc_auc',
            'param_config': {
                'n_estimators': {'min_value': 10, 'max_value': 250, 'param_type': "int"},
                'max_depth': {'percentage_splits': [0.25, 0.50, 0.70, 0.8, 0.9, 0.999], 'param_type': "float", 'dependency': 'n_samples'},
                'learning_rate': {'min_value': 0.001, 'max_value': 0.5, 'param_type': "float"},
                'subsample': {'min_value': 0.5, 'max_value': 1.0, 'param_type': "float"},
                'colsample_bytree': {'min_value': 0.5, 'max_value': 1.0, 'param_type': "float"}
            }
        }

    @staticmethod
    def get_param_grid_for_optimization(model_type: str, dataset_shape: tuple) -> Dict[str, Any]:
        """
        Convert model param_config to parameter grid suitable for Optuna optimization.
        
        Args:
            model_type: Type of model ('decision_tree', 'random_forest', 'xgboost')
            dataset_shape: (n_samples, n_features) tuple for dataset-dependent parameters
            
        Returns:
            Parameter grid dictionary suitable for optimize_hyperparameters
        """
        # Get the model configuration
        if model_type == 'decision_tree':
            config = ModelConfigs.get_decision_tree_config()
        elif model_type == 'random_forest':
            config = ModelConfigs.get_random_forest_config()
        elif model_type == 'xgboost':
            config = ModelConfigs.get_xgboost_config()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        param_config = config['param_config']
        param_grid = {}
        n_samples, n_features = dataset_shape
        
        for param_name, param_info in param_config.items():
            # Handle direct list parameters (e.g., n_estimators: [50, 100, 200, 500, 1000])
            if isinstance(param_info, list):
                param_grid[param_name] = param_info
                continue
                
            param_type = param_info.get('param_type', 'float')
            
            if 'percentage_splits' in param_info:
                # Use percentage splits to define ranges
                splits = param_info['percentage_splits']
                min_val = min(splits)
                max_val = max(splits)
                
                if param_info.get('dependency') == 'n_samples':
                    # Parameters that depend on dataset size - use as percentages
                    if param_name in ['min_samples_split', 'min_samples_leaf']:
                        param_grid[param_name] = (min_val, max_val, 'linear')
                    elif param_name == 'max_depth':
                        # For max_depth, convert to reasonable integer ranges
                        max_theoretical_depth = max(1, int(np.log2(n_samples) * 2))
                        param_grid[param_name] = (1, min(50, max_theoretical_depth), 'int')
                elif param_info.get('dependency') == 'n_features':
                    # Parameters that depend on number of features - use as percentages
                    param_grid[param_name] = (min_val, max_val, 'linear')
                else:
                    # No dependency, use values directly
                    if param_type == 'int':
                        param_grid[param_name] = (int(min_val), int(max_val), 'int')
                    else:
                        param_grid[param_name] = (min_val, max_val, 'linear')
            elif 'min_value' in param_info and 'max_value' in param_info:
                # Direct min/max specification
                min_val = param_info['min_value']
                max_val = param_info['max_value']
                scale = param_info.get('scale', 'linear')
                if param_type == 'int':
                    param_grid[param_name] = (int(min_val), int(max_val), 'int')
                else:
                    param_grid[param_name] = (min_val, max_val, scale)
        
        return param_grid
    
    @staticmethod
    def generate_random_hyperparameters(
        model_type: str,
        dataset_size: int = 1000,
        n_features: int = 10,
        random_state: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate random hyperparameters for any ML algorithm using ModelConfigs.
        
        This function generates random hyperparameters for Decision Trees, Random Forest,
        XGBoost, etc. using their respective ModelConfigs to ensure consistency with
        knowledge base generation.
        
        Args:
            model_type: Type of model ('decision_tree', 'random_forest', 'xgboost')
            dataset_size: Size of dataset for percentage-based parameter calculation
            n_features: Number of features for feature-based parameter calculation
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary of random hyperparameters matching the model's configuration
            
        Raises:
            ValueError: If model_type is not supported
        """
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
        
        # Get configuration for the specified model type
        if model_type == 'decision_tree':
            config = ModelConfigs.get_decision_tree_config()
        elif model_type == 'random_forest':
            config = ModelConfigs.get_random_forest_config()
        elif model_type == 'xgboost':
            config = ModelConfigs.get_xgboost_config()
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Supported types: 'decision_tree', 'random_forest', 'xgboost'")
        
        param_config = config['param_config']
        random_params = {}
        
        # Handle max_depth (common to all tree-based models)
        if 'max_depth' in param_config:
            max_theoretical_depth = max(1, int(math.log2(dataset_size) * 2))
            max_depth_percentages = param_config['max_depth']['percentage_splits']
            max_depth_options = [max(1, int(p * max_theoretical_depth)) for p in max_depth_percentages]
            
            # Random Forest allows unlimited depth (None)
            if model_type == 'random_forest':
                max_depth_options.append(None)
            
            random_params['max_depth'] = random.choice(max_depth_options)
        
        # Handle parameters specific to each model type
        if model_type == 'decision_tree':
            # Decision Tree: min_samples_split, min_samples_leaf, max_features (as fractions)
            for param_name in ['min_samples_split', 'min_samples_leaf', 'max_features']:
                if param_name in param_config:
                    param_conf = param_config[param_name]
                    random_params[param_name] = random.uniform(param_conf['min_value'], param_conf['max_value'])
        
        elif model_type == 'random_forest':
            # Random Forest: n_estimators, min_samples_split, min_samples_leaf, max_features (as absolute values)
            if 'n_estimators' in param_config:
                n_est_config = param_config['n_estimators']
                random_params['n_estimators'] = random.randint(n_est_config['min_value'], n_est_config['max_value'])
            
            # Convert percentage-based parameters to absolute values
            for param_name, multiplier in [('min_samples_split', dataset_size), ('min_samples_leaf', dataset_size), ('max_features', n_features)]:
                if param_name in param_config:
                    param_conf = param_config[param_name]
                    pct_value = random.uniform(param_conf['min_value'], param_conf['max_value'])
                    
                    if param_name == 'min_samples_split':
                        random_params[param_name] = max(2, int(pct_value * multiplier))
                    elif param_name == 'min_samples_leaf':
                        random_params[param_name] = max(1, int(pct_value * multiplier))
                    elif param_name == 'max_features':
                        random_params[param_name] = max(1, int(pct_value * multiplier))
        
        elif model_type == 'xgboost':
            # XGBoost: n_estimators, learning_rate, subsample, colsample_bytree
            for param_name in ['n_estimators']:
                if param_name in param_config:
                    param_conf = param_config[param_name]
                    random_params[param_name] = random.randint(param_conf['min_value'], param_conf['max_value'])
            
            for param_name in ['learning_rate', 'subsample', 'colsample_bytree']:
                if param_name in param_config:
                    param_conf = param_config[param_name]
                    random_params[param_name] = round(random.uniform(param_conf['min_value'], param_conf['max_value']), 6)
        
        return random_params 