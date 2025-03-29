"""
Main ZeroTune interface for hyperparameter optimization.

This module provides the main interface for ZeroTune, integrating data loading,
feature extraction, model configuration, knowledge base management, and optimization.
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
    find_similar_datasets,
    retrieve_best_configurations,
    optimize_hyperparameters,
    train_final_model,
    calculate_performance_score
)
from zerotune.core.model_configs import ModelConfigs
from zerotune.core.config import CONFIG, get_knowledge_base_path
from zerotune.core.utils import convert_to_dataframe


class ZeroTune:
    """
    Main class for ZeroTune hyperparameter optimization.
    
    This class provides the primary interface for meta-learning based
    hyperparameter optimization. It leverages knowledge from previous
    optimization runs on similar datasets to accelerate hyperparameter tuning.
    """
    
    def __init__(self, model_type: str = None, kb_path: Optional[str] = None) -> None:
        """
        Initialize ZeroTune with a specified model type.
        
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
            raise ImportError(f"Failed to import model class for {self.model_type}: {str(e)}")
    
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
    ) -> Dict[str, List[Union[int, float]]]:
        """
        Convert parameter configuration format to the format expected by optimize_hyperparameters.
        
        Args:
            param_config: Parameter configuration from model config
            X_shape: Shape of the feature matrix to calculate dependent values
            
        Returns:
            Parameter grid in format expected by optimize_hyperparameters
        """
        n_samples, n_features = X_shape
        param_grid: Dict[str, List[Union[int, float]]] = {}
        
        for param, config in param_config.items():
            if 'percentage_splits' in config:
                splits = config['percentage_splits']
                param_type = config.get('param_type', 'float')
                dependency = config.get('dependency', None)
                multiplier = config.get('multiplier', 1)
                
                if dependency == 'n_samples':
                    base_value = n_samples
                elif dependency == 'n_features':
                    base_value = n_features
                else:
                    base_value = 1
                
                if param_type == 'int':
                    values = [int(split * base_value * multiplier) for split in splits]
                    values = [max(1, val) for val in values]  # Ensure no zeros
                    param_grid[param] = values
                else:
                    param_grid[param] = splits
            else:
                # Default to original value if format not recognized
                param_grid[param] = config
        
        return param_grid
    
    def optimize(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        n_iter: Optional[int] = None,
        test_size: Optional[float] = None,
        random_state: Optional[int] = None,
        verbose: bool = True
    ) -> Tuple[Dict[str, Any], float, BaseEstimator]:
        """
        Optimize hyperparameters for a given dataset using meta-learning.
        
        This is the main method for using ZeroTune. It performs the following steps:
        1. Calculates meta-features for the provided dataset
        2. Finds similar datasets in the knowledge base
        3. Retrieves best hyperparameter configurations from similar datasets
        4. Performs hyperparameter optimization using the retrieved configurations as guidance
        5. Trains a final model with the best hyperparameters
        6. Updates the knowledge base with the new optimization results
        
        Args:
            X: Feature matrix for the dataset to optimize
            y: Target values for the dataset
            n_iter: Number of iterations for hyperparameter optimization (default from config)
            test_size: Proportion of the dataset to use for testing when splitting (default from config)
            random_state: Random seed for reproducibility (default from config)
            verbose: Whether to print progress information during optimization
            
        Returns:
            A tuple containing:
                - best_params: The best hyperparameters found
                - best_score: The performance score achieved with the best hyperparameters
                - trained_model: A trained model instance using the best hyperparameters
            
        Note:
            This method updates the knowledge base with the results of the optimization.
        """
        # Use defaults from config if parameters not provided
        n_iter = n_iter if n_iter is not None else CONFIG["defaults"]["n_iter"]
        test_size = test_size if test_size is not None else CONFIG["defaults"]["test_size"]
        random_state = random_state if random_state is not None else CONFIG["defaults"]["random_state"]
        
        # Convert to dataframe if needed for meta-feature extraction
        X_df = convert_to_dataframe(X)
        
        # Calculate meta-features for the dataset
        meta_features = calculate_dataset_meta_parameters(X_df, y)
        
        if verbose:
            print(f"Calculated meta-features: {meta_features}")
        
        # Split data for training and validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Find similar datasets
        similar_datasets = find_similar_datasets(
            meta_features, 
            self.knowledge_base,
            n_neighbors=CONFIG["defaults"]["n_neighbors"]
        )
        
        if verbose:
            print(f"Found similar datasets: {similar_datasets}")
        
        # Retrieve best configurations from similar datasets
        best_configs = retrieve_best_configurations(
            similar_datasets,
            self.knowledge_base,
            model_type=self.model_type
        )
        
        if verbose:
            print(f"Retrieved {len(best_configs)} configurations from similar datasets")
            
        # Convert parameter config to grid
        param_grid = self._convert_param_config_to_grid(
            self.model_config['param_config'],
            X_train.shape
        )
        
        if verbose:
            print(f"Parameter grid: {param_grid}")
        
        # Perform hyperparameter optimization
        best_params, best_score, all_results = optimize_hyperparameters(
            self.model_class,
            param_grid,
            X_train, y_train,
            metric=self.model_config['metric'],
            n_iter=n_iter,
            test_size=test_size,
            random_state=random_state,
            verbose=verbose
        )
        
        if verbose:
            print(f"Best parameters: {best_params}")
            print(f"Best score: {best_score}")
        
        # Train final model
        final_model = train_final_model(
            self.model_class,
            best_params,
            X_train, y_train
        )
        
        # Evaluate on test set
        test_score = calculate_performance_score(
            final_model,
            X_test, y_test,
            metric=self.model_config['metric']
        )
        
        if verbose:
            print(f"Test score: {test_score}")
        
        # Update knowledge base
        dataset_name = f"user_dataset_{len(self.knowledge_base.get('datasets', []))}"
        update_knowledge_base(
            self.knowledge_base,
            dataset_name=dataset_name,
            meta_features=meta_features,
            best_hyperparameters=best_params,
            best_score=best_score,
            all_results=all_results,
            model_type=self.model_type
        )
        
        # Save updated knowledge base
        save_knowledge_base(self.knowledge_base, self.kb_path, self.model_type)
        
        return best_params, best_score, final_model
    
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
        to build a comprehensive knowledge base that can be used for future optimizations.
        
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
            dataset_ids = get_recommended_datasets('classification', n_datasets)
            
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
                
                # Run optimization
                best_params, best_score, model = self.optimize(
                    X, y,
                    n_iter=n_iter,
                    random_state=random_state,
                    verbose=verbose
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