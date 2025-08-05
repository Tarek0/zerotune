"""
ZeroTune Knowledge Base Builder for training zero-shot predictors.

This module provides functionality to build knowledge bases from multiple datasets
that can then be used to train zero-shot hyperparameter predictors.
"""

import os
import importlib
from typing import Dict, List, Optional, Tuple, Union, Any, Type
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

# Import components from ZeroTune submodules
from zerotune.core.data_loading import (
    fetch_open_ml_data,
    prepare_data,
    get_dataset_ids,
    get_recommended_datasets
)
from zerotune.core.feature_extraction import calculate_dataset_meta_parameters
from zerotune.core.knowledge_base import (
    load_knowledge_base,
    save_knowledge_base,
    update_knowledge_base
)
from zerotune.core.optimization import (
    optimize_hyperparameters,
    train_final_model,
    calculate_performance_score
)
from zerotune.core.model_configs import ModelConfigs
from zerotune.core.config import CONFIG, get_knowledge_base_path
from zerotune.core.utils import convert_to_dataframe


class ZeroTune:
    """
    Knowledge base builder for training zero-shot hyperparameter predictors.
    
    This class builds comprehensive knowledge bases by running hyperparameter optimization
    on multiple datasets and storing:
    - Dataset meta-features (characteristics like n_samples, n_features, imbalance_ratio, etc.)
    - Optimal hyperparameters found for each dataset
    - Performance scores achieved
    
    The resulting knowledge base provides training data for zero-shot predictors.
    
    For zero-shot predictions, use ZeroTunePredictor instead of this class.
    """
    
    def __init__(self, model_type: str = None, kb_path: Optional[str] = None) -> None:
        """
        Initialize ZeroTune knowledge base builder.
        
        Args:
            model_type: Type of model to optimize. 
                Options: "decision_tree", "random_forest", "xgboost"
            kb_path: Path to knowledge base file. If not provided,
                the default knowledge base path will be used.
        """
        # Use config defaults if not specified
        self.model_type: str = model_type or CONFIG["defaults"]["model_type"]
        self.model_class: Optional[Type[BaseEstimator]] = None
        self.model_config: Dict[str, Any] = {}
        self.kb_path: str = ""
        self.knowledge_base: Dict[str, Any] = {}
        
        # Load the appropriate model class
        self._load_model_class()
        
        # Get model configuration based on model_type
        self.model_config = self._get_model_config()
        
        # Set knowledge base path
        self.kb_path = kb_path or get_knowledge_base_path()
        
        # Initialize knowledge base
        self.knowledge_base = load_knowledge_base(self.kb_path, self.model_type)
    
    def _load_model_class(self) -> None:
        """
        Load the appropriate model class based on model_type.
        
        This method imports and initializes the appropriate scikit-learn or XGBoost
        model class based on the model_type specified during initialization.
        
        Raises:
            ValueError: If an unsupported model type is specified
            ImportError: If the required package for a model is not installed
        """
        try:
            if self.model_type == "decision_tree":
                from sklearn.tree import DecisionTreeClassifier
                self.model_class = DecisionTreeClassifier
            elif self.model_type == "random_forest":
                from sklearn.ensemble import RandomForestClassifier
                self.model_class = RandomForestClassifier
            elif self.model_type == "xgboost":
                import xgboost as xgb
                self.model_class = xgb.XGBClassifier
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
        except ImportError as e:
            raise ImportError(f"Required package for {self.model_type} not installed: {e}")
    
    def _get_model_config(self) -> Dict[str, Any]:
        """
        Get model configuration based on model_type.
        
        Returns:
            Model configuration dictionary containing hyperparameter settings and constraints
        
        Raises:
            ValueError: If no configuration is available for the specified model type
        """
        if self.model_type == "decision_tree":
            return ModelConfigs.get_decision_tree_config()
        elif self.model_type == "random_forest":
            return ModelConfigs.get_random_forest_config()
        elif self.model_type == "xgboost":
            return ModelConfigs.get_xgboost_config()
        else:
            raise ValueError(f"No configuration available for model type: {self.model_type}")
    
    def _convert_param_config_to_grid(
        self,
        param_config: Dict[str, Dict[str, Any]],
        X_shape: Tuple[int, int]
    ) -> Dict[str, Tuple[float, float, str]]:
        """
        Convert parameter configuration to parameter grid for Optuna optimization.
        
        Args:
            param_config: Parameter configuration from model config
            X_shape: Shape of the input data (n_samples, n_features)
            
        Returns:
            Parameter grid suitable for Optuna optimization (param_name -> (low, high, scale))
        """
        param_grid = {}
        n_samples, n_features = X_shape
        
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
                    # Parameters that depend on dataset size
                    if param_name == 'max_depth':
                        max_depth = max(1, int(np.log2(n_samples)))
                        param_grid[param_name] = (1, max_depth, "int")
                    else:
                        # Scale by n_samples
                        base_val = n_samples
                        multiplier = param_info.get('multiplier', 1)
                        param_grid[param_name] = (
                            int(min_val * base_val * multiplier), 
                            int(max_val * base_val * multiplier), 
                            "int"
                        )
                elif param_info.get('dependency') == 'n_features':
                    # Parameters that depend on number of features
                    base_val = n_features
                    multiplier = param_info.get('multiplier', 1)
                    if param_type == 'int':
                        calculated_min = max(1, int(min_val * base_val * multiplier))
                        calculated_max = int(max_val * base_val * multiplier)
                        
                        # Apply min/max constraints if specified
                        if 'min_value' in param_info:
                            calculated_min = max(calculated_min, param_info['min_value'])
                        if 'max_value' in param_info:
                            calculated_max = min(calculated_max, param_info['max_value'])
                        
                        param_grid[param_name] = (calculated_min, calculated_max, "int")
                    else:
                        calculated_min = min_val * base_val * multiplier
                        calculated_max = max_val * base_val * multiplier
                        
                        # Apply min/max constraints if specified
                        if 'min_value' in param_info:
                            calculated_min = max(calculated_min, param_info['min_value'])
                        if 'max_value' in param_info:
                            calculated_max = min(calculated_max, param_info['max_value'])
                        
                        param_grid[param_name] = (calculated_min, calculated_max, "linear")
                else:
                    # Direct percentage values (e.g., learning_rate, subsample, colsample_bytree)
                    if param_type == 'float':
                        param_grid[param_name] = (min_val, max_val, "linear")
                    else:
                        param_grid[param_name] = (int(min_val), int(max_val), "int")
            else:
                # Fallback for old-style configurations
                param_range = param_info.get('range', [0.01, 1.0])
                if param_type == 'int':
                    param_grid[param_name] = (param_range[0], param_range[1], "int")
                else:
                    param_grid[param_name] = (param_range[0], param_range[1], "linear")
        
        return param_grid
    
    def build_knowledge_base(
        self,
        dataset_ids: Optional[List[int]] = None,
        n_datasets: int = 5,
        n_iter: Optional[int] = None,
        random_state: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Build a knowledge base by running optimization on multiple datasets.
        
        This method fetches multiple datasets from OpenML and runs optimization on each
        to build a comprehensive knowledge base that can be used to train zero-shot predictors.
        
        Args:
            dataset_ids: List of OpenML dataset IDs to use. If not provided,
                a curated list of datasets will be used.
            n_datasets: Number of datasets to use if dataset_ids is not provided
            n_iter: Number of iterations for each optimization
            random_state: Random seed for reproducibility
            verbose: Whether to print progress information during optimization
            
        Returns:
            The updated knowledge base
        """
        # Use defaults from config if parameters not provided
        n_iter = n_iter if n_iter is not None else CONFIG["defaults"]["n_iter"]
        random_state = random_state if random_state is not None else CONFIG["defaults"]["random_state"]
        
        # Get list of dataset IDs if not provided
        if dataset_ids is None:
            dataset_ids = get_recommended_datasets(n_datasets=n_datasets)
            
        if verbose:
            print(f"Building knowledge base using {len(dataset_ids)} datasets...")
        
        # Process each dataset
        for dataset_id in dataset_ids:
            if verbose:
                print(f"\nProcessing dataset ID: {dataset_id}")
            
            try:
                # Fetch dataset from OpenML
                data, target_name, dataset_name = fetch_open_ml_data(dataset_id)
                
                if verbose:
                    print(f"Dataset: {dataset_name}, Target: {target_name}")
                
                # Prepare data for modeling
                X, y = prepare_data(data, target_name)
                
                if verbose:
                    print(f"Data shape: {X.shape}, Target shape: {y.shape}")
                
                # Run optimization to find best hyperparameters for this dataset
                best_params, best_score = self._optimize_single_dataset(
                    X, y, n_iter, random_state, verbose
                )
                
                if verbose:
                    print(f"Completed optimization for dataset {dataset_name}")
                    print(f"Best parameters: {best_params}")
                    print(f"Best score: {best_score}")
                
            except Exception as e:
                if verbose:
                    print(f"Error processing dataset {dataset_id}: {str(e)}")
        
        if verbose:
            print("\nKnowledge base building completed.")
        
        return self.knowledge_base
    
    def _optimize_single_dataset(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        y: Union[np.ndarray, pd.Series],
        n_iter: int,
        random_state: int,
        verbose: bool
    ) -> Tuple[Dict[str, Any], float]:
        """
        Run hyperparameter optimization on a single dataset and update knowledge base.
        
        Args:
            X: Feature matrix
            y: Target values
            n_iter: Number of optimization iterations
            random_state: Random seed
            verbose: Whether to print progress
            
        Returns:
            Tuple of (best_params, best_score)
        """
        # Convert to dataframe if needed for meta-feature extraction
        X_df = convert_to_dataframe(X)
        
        # Calculate meta-features for the dataset
        meta_features = calculate_dataset_meta_parameters(X_df, y)
        
        if verbose:
            print(f"Calculated meta-features: {meta_features}")
        
        # Split data for training and validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )
        
        # Convert parameter config to grid
        param_grid = self._convert_param_config_to_grid(
            self.model_config['param_config'],
            X_train.shape
        )
        
        # Run hyperparameter optimization (no warm start - pure data collection)
        if verbose:
            print("Running hyperparameter optimization for knowledge base data collection")
        
        best_params, best_score, _, df_trials = optimize_hyperparameters(
            self.model_class,
            param_grid,
            X_train,
            y_train,
            metric=self.model_config["metric"],
            n_iter=n_iter,
            test_size=0.2,
            random_state=random_state,
            verbose=verbose,
            warm_start_configs=None,  # No warm start for knowledge base building
            dataset_meta_params=meta_features
        )
        
        # Train final model with best parameters
        trained_model = train_final_model(
            self.model_class,
            best_params,
            X_train,
            y_train
        )
        
        # Calculate final score on test set
        final_score = calculate_performance_score(
            trained_model,
            X_test,
            y_test,
            metric=self.model_config["metric"]
        )
        
        if verbose:
            print(f"Final test score: {final_score:.4f}")
        
        # Update knowledge base with new results
        if self.knowledge_base:
            self.knowledge_base = update_knowledge_base(
                self.knowledge_base,
                dataset_name="unknown",  # Could be improved by passing dataset name
                meta_features=meta_features,
                best_hyperparameters=best_params,
                best_score=final_score,
                model_type=self.model_type,
                df_trials=df_trials
            )
            # Save updated knowledge base
            save_knowledge_base(self.knowledge_base, self.kb_path)
        
        return best_params, final_score
    
    def save_knowledge_base(self):
        """
        Save the current knowledge base to disk.
        
        Returns:
            bool: True if successful, False otherwise
        """
        return save_knowledge_base(self.knowledge_base, self.kb_path, self.model_type)
    
    def load_knowledge_base(self):
        """
        Load knowledge base from disk.
        
        Returns:
            dict: Loaded knowledge base
        """
        self.knowledge_base = load_knowledge_base(self.kb_path, self.model_type)
        return self.knowledge_base 