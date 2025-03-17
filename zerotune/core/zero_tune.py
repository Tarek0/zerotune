"""
Main ZeroTune interface for hyperparameter optimization.

This module provides the main interface for ZeroTune, integrating data loading,
feature extraction, model configuration, knowledge base management, and optimization.
"""

import os
import importlib
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
    update_knowledge_base,
    get_knowledge_base_path
)
from zerotune.core.optimization import (
    find_similar_datasets,
    retrieve_best_configurations,
    optimize_hyperparameters,
    train_final_model,
    calculate_performance_score
)
from zerotune.core.model_configs import ModelConfigs


class ZeroTune:
    """
    Main class for ZeroTune hyperparameter optimization.
    
    This class provides the primary interface for meta-learning based
    hyperparameter optimization. It leverages knowledge from previous
    optimization runs on similar datasets to accelerate hyperparameter tuning.
    """
    
    def __init__(self, model_type="decision_tree", kb_path=None):
        """
        Initialize ZeroTune with a specified model type.
        
        Args:
            model_type (str): Type of model to optimize. 
                Options: "decision_tree", "random_forest", "xgboost"
            kb_path (str, optional): Path to knowledge base file. If not provided,
                the default knowledge base path will be used.
        """
        self.model_type = model_type
        
        # Load the appropriate model class
        self._load_model_class()
        
        # Get model configuration based on model_type
        self.model_config = self._get_model_config()
        
        # Set knowledge base path
        self.kb_path = kb_path or get_knowledge_base_path()
        
        # Initialize knowledge base
        self.knowledge_base = load_knowledge_base(self.kb_path, model_type)
    
    def _load_model_class(self):
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
    
    def _get_model_config(self):
        """
        Get model configuration based on model_type.
        
        Returns:
            dict: Model configuration
        """
        if self.model_type == "decision_tree":
            return ModelConfigs.get_decision_tree_config()
        elif self.model_type == "random_forest":
            return ModelConfigs.get_random_forest_config()
        elif self.model_type == "xgboost":
            return ModelConfigs.get_xgboost_config()
        else:
            raise ValueError(f"No configuration available for model type: {self.model_type}")
    
    def _convert_param_config_to_grid(self, param_config, X_shape):
        """
        Convert parameter configuration format to the format expected by optimize_hyperparameters.
        
        Args:
            param_config: Parameter configuration from model config
            X_shape: Shape of the feature matrix to calculate dependent values
            
        Returns:
            dict: Parameter grid in format expected by optimize_hyperparameters
        """
        n_samples, n_features = X_shape
        param_grid = {}
        
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
    
    def optimize(self, X, y, n_iter=10, test_size=0.2, random_state=42, verbose=True):
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
            X (array-like or DataFrame): Feature matrix for the dataset to optimize
            y (array-like or Series): Target values for the dataset
            n_iter (int): Number of iterations for hyperparameter optimization. Default is 10.
            test_size (float): Proportion of the dataset to use for testing when splitting. Default is 0.2.
            random_state (int): Random seed for reproducibility. Default is 42.
            verbose (bool): Whether to print progress information during optimization. Default is True.
            
        Returns:
            tuple: A tuple containing three elements:
                - best_params (dict): The best hyperparameters found
                - best_score (float): The performance score achieved with the best hyperparameters
                - trained_model: A trained model instance using the best hyperparameters
            
        Note:
            The method automatically updates the knowledge base with results from this
            optimization run, which will be available for future optimizations.
        """
        # Calculate meta-features for the dataset
        meta_features = calculate_dataset_meta_parameters(X, y)
        
        # Find similar datasets in the knowledge base
        similar_indices = find_similar_datasets(meta_features, self.knowledge_base, n_neighbors=3)
        
        # Get parameter grid
        param_grid = self._convert_param_config_to_grid(self.model_config.get("param_config", {}), X.shape)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # If we have similar datasets, use their configurations as starting points
        if similar_indices and verbose:
            print(f"Found {len(similar_indices)} similar datasets in knowledge base")
            
            # Retrieve best configurations from similar datasets
            best_configs = retrieve_best_configurations(
                similar_indices, self.knowledge_base, self.model_type
            )
            
            if best_configs and verbose:
                print(f"Retrieved {len(best_configs)} configurations from similar datasets")
                print("Starting optimization with these configurations as guidance...")
        
        # Perform hyperparameter optimization
        best_params, best_score, all_results = optimize_hyperparameters(
            self.model_class, param_grid, X_train, y_train,
            metric=self.model_config.get("metric", "accuracy"),
            n_iter=n_iter, test_size=test_size/2,  # Further split training data
            random_state=random_state, verbose=verbose
        )
        
        # Train final model on full training set
        model = train_final_model(self.model_class, best_params, X_train, y_train)
        
        # Evaluate on test set
        test_score = calculate_performance_score(
            model, X_test, y_test, metric=self.model_config.get("metric", "accuracy")
        )
        
        if verbose:
            print(f"Best validation score: {best_score}")
            print(f"Test score: {test_score}")
            print(f"Best parameters: {best_params}")
        
        # Update knowledge base
        self.knowledge_base = update_knowledge_base(
            self.knowledge_base, f"Optimization {len(self.knowledge_base) + 1}", meta_features,
            best_params, best_score, model_type=self.model_type
        )
        
        # Save knowledge base after optimization
        save_knowledge_base(
            self.knowledge_base, self.kb_path, self.model_type
        )
        
        return best_params, test_score, model
    
    def build_knowledge_base(self, dataset_ids=None, n_datasets=5, n_iter=10, 
                             random_state=42, verbose=True):
        """
        Build a knowledge base from multiple datasets.
        
        Args:
            dataset_ids (list, optional): List of OpenML dataset IDs
            n_datasets (int): Number of datasets to use if ids not provided
            n_iter (int): Number of iterations for optimization on each dataset
            random_state (int): Random seed
            verbose (bool): Whether to print progress
            
        Returns:
            dict: Updated knowledge base
        """
        # Get dataset IDs if not provided
        if dataset_ids is None:
            dataset_ids = get_recommended_datasets(
                task="classification", n_datasets=n_datasets
            )
        
        if verbose:
            print(f"Building knowledge base with {len(dataset_ids)} datasets")
        
        # Process each dataset
        for dataset_id in dataset_ids:
            try:
                if verbose:
                    print(f"\nProcessing dataset {dataset_id}")
                
                # Load dataset
                df, target_name, dataset_name = fetch_open_ml_data(dataset_id)
                X, y = prepare_data(df, target_name)
                
                # Calculate meta-features
                meta_features = calculate_dataset_meta_parameters(X, y)
                
                # Optimize hyperparameters
                best_params, best_score, _ = self.optimize(
                    X, y, n_iter=n_iter, random_state=random_state, verbose=verbose
                )
                
                # Update knowledge base
                self.knowledge_base = update_knowledge_base(
                    self.knowledge_base, dataset_name, meta_features,
                    best_params, best_score, dataset_id=dataset_id,
                    model_type=self.model_type
                )
                
                # Save knowledge base after each dataset
                save_knowledge_base(
                    self.knowledge_base, self.kb_path, self.model_type
                )
                
                if verbose:
                    print(f"Updated knowledge base with dataset {dataset_name}")
            
            except Exception as e:
                print(f"Error processing dataset {dataset_id}: {str(e)}")
                continue
        
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