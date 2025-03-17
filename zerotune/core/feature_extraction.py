"""
Feature extraction functions for ZeroTune.

This module handles calculating meta-features for datasets, which are
used to characterize datasets for hyperparameter optimization.
"""

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis


def _calculate_dataset_size(X):
    """
    Calculate basic dataset dimensions.
    
    Args:
        X (DataFrame or ndarray): Input feature matrix
        
    Returns:
        dict: Dictionary containing dataset dimensions:
            - n_samples: Number of samples (rows)
            - n_features: Number of features (columns)
    """
    return {"n_samples": X.shape[0], "n_features": X.shape[1]}


def _calculate_class_imbalance_ratio(y):
    """
    Calculate the class imbalance ratio of a dataset.
    
    Args:
        y (array-like): Target values (class labels).
        
    Returns:
        dict: Dictionary containing imbalance metrics:
            - imbalance_ratio: The ratio of the majority class size to the minority class size.
    """
    # Count the occurrences of each class
    class_counts = np.bincount(y)
    
    # Find the counts of majority and minority classes
    majority_class_count = np.max(class_counts)
    minority_class_count = np.min(class_counts)
    
    # Calculate the imbalance ratio
    imbalance_ratio = majority_class_count / minority_class_count
    
    return {"imbalance_ratio": imbalance_ratio}


def _calculate_correlation_metrics(X, y, correlation_cutoff=0.1):
    """
    Calculates correlation metrics between features and the target variable.
    
    Args:
        X (DataFrame): The input features.
        y (Series): The target variable.
        correlation_cutoff (float): Minimum absolute correlation threshold to consider
            a feature as correlated with the target.
        
    Returns:
        dict: Dictionary containing correlation metrics:
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
        'n_highly_target_corr': n_informative,
        'avg_target_corr': correlations_with_target.mean(),
        'var_target_corr': correlations_with_target.var()
    }


def _calculate_feature_moments_and_variances(X):
    """
    Calculate statistical moments for each feature in the dataset.
    
    Args:
        X (DataFrame): DataFrame containing the feature set.
        
    Returns:
        dict: Statistical moments for features
    """
    # Convert to DataFrame if needed
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    
    # Calculate moments for each column
    moment_1 = X.apply(lambda x: x.mean(), axis=0)
    moment_2 = X.apply(lambda x: x.var(), axis=0)
    moment_3 = X.apply(lambda x: skew(x.dropna()), axis=0)
    moment_4 = X.apply(lambda x: kurtosis(x.dropna()), axis=0)
    
    # Calculate averages and variances
    moments = {
        'avg_feature_m1': moment_1.mean(),  # Average Mean
        'var_feature_m1': moment_1.var(),   # Variance of Mean
        'avg_feature_m2': moment_2.mean(),  # Average Variance
        'var_feature_m2': moment_2.var(),   # Variance of Variance
        'avg_feature_m3': moment_3.mean(),  # Average Skewness
        'var_feature_m3': moment_3.var(),   # Variance of Skewness
        'avg_feature_m4': moment_4.mean(),  # Average Kurtosis
        'var_feature_m4': moment_4.var(),   # Variance of Kurtosis
    }
    
    return moments


def _calculate_row_moments_and_variances(X):
    """
    Calculate statistical moments for each row in the dataset.
    
    Args:
        X (DataFrame): DataFrame containing the feature set.
        
    Returns:
        dict: Statistical moments for rows
    """
    # Convert to DataFrame if needed
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    
    # Calculate moments for each row
    moment_1 = X.apply(lambda x: x.mean(), axis=1)
    moment_2 = X.apply(lambda x: x.var(), axis=1)
    moment_3 = X.apply(lambda x: skew(x.dropna()), axis=1)
    moment_4 = X.apply(lambda x: kurtosis(x.dropna()), axis=1)
    
    # Calculate averages and variances
    moments = {
        'avg_row_m1': moment_1.mean(),  # Average Mean
        'var_row_m1': moment_1.var(),   # Variance of Mean
        'avg_row_m2': moment_2.mean(),  # Average Variance
        'var_row_m2': moment_2.var(),   # Variance of Variance
        'avg_row_m3': moment_3.mean(),  # Average Skewness
        'var_row_m3': moment_3.var(),   # Variance of Skewness
        'avg_row_m4': moment_4.mean(),  # Average Kurtosis
        'var_row_m4': moment_4.var(),   # Variance of Kurtosis
    }
    
    return moments


def calculate_dataset_meta_parameters(X, y):
    """
    Comprehensive calculation of dataset meta-parameters.
    
    Args:
        X (DataFrame): Feature matrix
        y (Series): Target variable
        
    Returns:
        dict: Combined meta-parameters
    """
    meta_parameters = {}
    meta_parameters.update(_calculate_dataset_size(X))
    meta_parameters.update(_calculate_class_imbalance_ratio(y))
    meta_parameters.update(_calculate_correlation_metrics(X, y, correlation_cutoff=0.10))
    meta_parameters.update(_calculate_feature_moments_and_variances(X))
    meta_parameters.update(_calculate_row_moments_and_variances(X))
    
    return meta_parameters 