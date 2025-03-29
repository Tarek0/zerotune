"""
Feature extraction functions for ZeroTune.

This module handles calculating meta-features for datasets, which are
used to characterize datasets for hyperparameter optimization.
"""

from typing import Dict, Union, Any
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

# Type aliases
ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]
MetaFeatures = Dict[str, float]


def _calculate_dataset_size(X: ArrayLike) -> MetaFeatures:
    """
    Calculate basic dataset dimensions.
    
    Args:
        X: Input feature matrix
        
    Returns:
        Dictionary containing dataset dimensions:
            - n_samples: Number of samples (rows)
            - n_features: Number of features (columns)
    """
    return {"n_samples": X.shape[0], "n_features": X.shape[1]}


def _calculate_class_imbalance_ratio(y: ArrayLike) -> MetaFeatures:
    """
    Calculate the class imbalance ratio of a dataset.
    
    Args:
        y: Target values (class labels)
        
    Returns:
        Dictionary containing imbalance metrics:
            - imbalance_ratio: The ratio of the majority class size to the minority class size
    """
    # Count the occurrences of each class
    class_counts = np.bincount(y)
    
    # Find the counts of majority and minority classes
    majority_class_count = np.max(class_counts)
    minority_class_count = np.min(class_counts)
    
    # Calculate the imbalance ratio
    imbalance_ratio = float(majority_class_count / minority_class_count)
    
    return {"imbalance_ratio": imbalance_ratio}


def _calculate_correlation_metrics(
    X: pd.DataFrame,
    y: pd.Series,
    correlation_cutoff: float = 0.1
) -> MetaFeatures:
    """
    Calculates correlation metrics between features and the target variable.
    
    Args:
        X: The input features
        y: The target variable
        correlation_cutoff: Minimum absolute correlation threshold to consider
            a feature as correlated with the target
        
    Returns:
        Dictionary containing correlation metrics:
            - mean_correlation: Mean absolute correlation between features and target
            - high_correlation_count: Number of features with correlation above the cutoff
            - high_correlation_ratio: Ratio of highly correlated features to total features
    """
    df = pd.DataFrame(X.copy())
    df['target'] = pd.Series(y.copy())
    correlation_matrix = df.corr(method='pearson')
    correlations_with_target = abs(correlation_matrix['target'])
    
    informative_features = correlations_with_target[correlations_with_target > correlation_cutoff].sort_values(ascending=False)
    n_informative = len(informative_features) - 1
    
    return {
        'n_highly_target_corr': float(n_informative),
        'avg_target_corr': float(correlations_with_target.mean()),
        'var_target_corr': float(correlations_with_target.var())
    }


def _calculate_feature_moments_and_variances(X: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate statistical moments and variances for numeric features.
    
    Args:
        X: Feature matrix as DataFrame
        
    Returns:
        Dictionary of feature statistics including:
            - avg_feature_m1/var_feature_m1: Mean/variance of feature means
            - avg_feature_m2/var_feature_m2: Mean/variance of squared features
            - avg_feature_m3/var_feature_m3: Mean/variance of cubed features
            - avg_feature_m4/var_feature_m4: Mean/variance of fourth power features
            - avg_row_m1/var_row_m1: Mean/variance of row means
            - avg_row_m2/var_row_m2: Mean/variance of squared row values
            - avg_row_m3/var_row_m3: Mean/variance of cubed row values
            - avg_row_m4/var_row_m4: Mean/variance of fourth power row values
        
    Note:
        Only numeric columns (int64, float64) are used for calculations.
        Non-numeric columns are ignored.
    """
    # Only use numeric columns for statistics
    X_numeric = X.select_dtypes(include=['int64', 'float64'])
    
    # Calculate moments for numeric features
    moment_1 = X_numeric.apply(lambda x: x.mean(), axis=0)
    moment_2 = X_numeric.apply(lambda x: (x ** 2).mean(), axis=0)
    moment_3 = X_numeric.apply(lambda x: (x ** 3).mean(), axis=0)
    moment_4 = X_numeric.apply(lambda x: (x ** 4).mean(), axis=0)
    
    # Calculate row-wise statistics
    row_moment_1 = X_numeric.apply(lambda x: x.mean(), axis=1)
    row_moment_2 = X_numeric.apply(lambda x: (x ** 2).mean(), axis=1)
    row_moment_3 = X_numeric.apply(lambda x: (x ** 3).mean(), axis=1)
    row_moment_4 = X_numeric.apply(lambda x: (x ** 4).mean(), axis=1)
    
    return {
        'avg_feature_m1': float(moment_1.mean()),
        'var_feature_m1': float(moment_1.var()),
        'avg_feature_m2': float(moment_2.mean()),
        'var_feature_m2': float(moment_2.var()),
        'avg_feature_m3': float(moment_3.mean()),
        'var_feature_m3': float(moment_3.var()),
        'avg_feature_m4': float(moment_4.mean()),
        'var_feature_m4': float(moment_4.var()),
        'avg_row_m1': float(row_moment_1.mean()),
        'var_row_m1': float(row_moment_1.var()),
        'avg_row_m2': float(row_moment_2.mean()),
        'var_row_m2': float(row_moment_2.var()),
        'avg_row_m3': float(row_moment_3.mean()),
        'var_row_m3': float(row_moment_3.var()),
        'avg_row_m4': float(row_moment_4.mean()),
        'var_row_m4': float(row_moment_4.var())
    }


def _calculate_row_moments_and_variances(X: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate statistical moments for each row in the dataset.
    
    Args:
        X: DataFrame containing the feature set
        
    Returns:
        Dictionary of row-wise statistics including:
            - avg_row_m1/var_row_m1: Mean/variance of row means
            - avg_row_m2/var_row_m2: Mean/variance of squared row values
            - avg_row_m3/var_row_m3: Mean/variance of cubed row values
            - avg_row_m4/var_row_m4: Mean/variance of fourth power row values
        
    Note:
        Only numeric columns (int64, float64) are used for calculations.
        Non-numeric columns are ignored.
    """
    # Only use numeric columns for statistics
    X_numeric = X.select_dtypes(include=['int64', 'float64'])
    
    # Calculate moments for each row
    moment_1 = X_numeric.apply(lambda x: x.mean(), axis=1)
    moment_2 = X_numeric.apply(lambda x: (x ** 2).mean(), axis=1)
    moment_3 = X_numeric.apply(lambda x: (x ** 3).mean(), axis=1)
    moment_4 = X_numeric.apply(lambda x: (x ** 4).mean(), axis=1)
    
    return {
        'avg_row_m1': float(moment_1.mean()),
        'var_row_m1': float(moment_1.var()),
        'avg_row_m2': float(moment_2.mean()),
        'var_row_m2': float(moment_2.var()),
        'avg_row_m3': float(moment_3.mean()),
        'var_row_m3': float(moment_3.var()),
        'avg_row_m4': float(moment_4.mean()),
        'var_row_m4': float(moment_4.var())
    }


def calculate_dataset_meta_parameters(X: ArrayLike, y: ArrayLike) -> MetaFeatures:
    """
    Comprehensive calculation of dataset meta-parameters.
    
    Args:
        X: Feature matrix
        y: Target variable
        
    Returns:
        Combined meta-parameters dictionary containing all calculated metrics
    """
    meta_parameters: MetaFeatures = {}
    meta_parameters.update(_calculate_dataset_size(X))
    meta_parameters.update(_calculate_class_imbalance_ratio(y))
    meta_parameters.update(_calculate_correlation_metrics(X, y, correlation_cutoff=0.10))
    meta_parameters.update(_calculate_feature_moments_and_variances(X))
    meta_parameters.update(_calculate_row_moments_and_variances(X))
    
    return meta_parameters 