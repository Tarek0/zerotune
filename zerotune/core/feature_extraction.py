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
            - n_highly_target_corr: Number of features with correlation above the cutoff (integer)
            - avg_target_corr: Mean absolute correlation between features and target (float)
            - var_target_corr: Variance of correlations between features and target (float)
    """
    df = pd.DataFrame(X.copy())
    df['target'] = pd.Series(y.copy())
    correlation_matrix = df.corr(method='pearson')
    correlations_with_target = abs(correlation_matrix['target'])
    
    # Count features with correlation above cutoff (excluding target itself)
    n_informative = int(sum(correlations_with_target > correlation_cutoff)) - 1
    
    return {
        'n_highly_target_corr': n_informative,  # Return as integer
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
        NaN values are handled by replacing them with column means.
        Values are clipped to prevent numerical overflow.
    """
    try:
        # Only use numeric columns for statistics (include all numeric types)
        X_numeric = X.select_dtypes(include=[np.number])
        
        # Handle NaN values by replacing with column means
        for col in X_numeric.columns:
            if X_numeric[col].isna().any():
                mean_val = X_numeric[col].mean()
                if pd.isna(mean_val):  # If mean is NaN (all values are NaN)
                    X_numeric[col] = X_numeric[col].fillna(0)
                else:
                    X_numeric[col] = X_numeric[col].fillna(mean_val)
        
        # Check if we have valid data after handling NaNs
        if X_numeric.empty or X_numeric.shape[1] == 0:
            # Return 0 for all metrics if no valid data (instead of NaN)
            return {
                'avg_feature_m1': 0.0, 'var_feature_m1': 0.0,
                'avg_feature_m2': 0.0, 'var_feature_m2': 0.0,
                'avg_feature_m3': 0.0, 'var_feature_m3': 0.0,
                'avg_feature_m4': 0.0, 'var_feature_m4': 0.0,
                'avg_row_m1': 0.0, 'var_row_m1': 0.0,
                'avg_row_m2': 0.0, 'var_row_m2': 0.0,
                'avg_row_m3': 0.0, 'var_row_m3': 0.0,
                'avg_row_m4': 0.0, 'var_row_m4': 0.0
            }
        
        # Calculate moments for numeric features with error handling and clipping
        try:
            # Clip extreme values to prevent overflow (keep values in reasonable range)
            X_clipped = X_numeric.clip(-1e6, 1e6)
            
            # Calculate feature-wise moments
            moment_1 = X_clipped.apply(lambda x: x.mean(), axis=0)
            moment_2 = X_clipped.apply(lambda x: (x ** 2).mean(), axis=0)
            moment_3 = X_clipped.apply(lambda x: (x ** 3).mean(), axis=0)
            moment_4 = X_clipped.apply(lambda x: (x ** 4).mean(), axis=0)
            
            # Calculate row-wise statistics
            row_moment_1 = X_clipped.apply(lambda x: x.mean(), axis=1)
            row_moment_2 = X_clipped.apply(lambda x: (x ** 2).mean(), axis=1)
            row_moment_3 = X_clipped.apply(lambda x: (x ** 3).mean(), axis=1)
            row_moment_4 = X_clipped.apply(lambda x: (x ** 4).mean(), axis=1)
            
            # Helper function to safely calculate mean and variance with clipping
            def safe_mean_var(series, max_val=1e12):
                """Safely calculate mean and variance with clipping."""
                # Remove any inf/nan values
                clean_series = series.replace([np.inf, -np.inf], np.nan).dropna()
                if len(clean_series) == 0:
                    return 0.0, 0.0
                
                # Clip to prevent extreme values
                clipped_series = clean_series.clip(-max_val, max_val)
                
                mean_val = float(clipped_series.mean())
                var_val = float(clipped_series.var()) if len(clipped_series) > 1 else 0.0
                
                # Final safety check
                mean_val = np.clip(mean_val, -max_val, max_val)
                var_val = np.clip(var_val, 0.0, max_val)  # Variance should be non-negative
                
                return mean_val, var_val
            
            # Calculate statistics with safety checks
            avg_f_m1, var_f_m1 = safe_mean_var(moment_1)
            avg_f_m2, var_f_m2 = safe_mean_var(moment_2)
            avg_f_m3, var_f_m3 = safe_mean_var(moment_3)
            avg_f_m4, var_f_m4 = safe_mean_var(moment_4)
            
            avg_r_m1, var_r_m1 = safe_mean_var(row_moment_1)
            avg_r_m2, var_r_m2 = safe_mean_var(row_moment_2)
            avg_r_m3, var_r_m3 = safe_mean_var(row_moment_3)
            avg_r_m4, var_r_m4 = safe_mean_var(row_moment_4)
            
            return {
                'avg_feature_m1': avg_f_m1,
                'var_feature_m1': var_f_m1,
                'avg_feature_m2': avg_f_m2,
                'var_feature_m2': var_f_m2,
                'avg_feature_m3': avg_f_m3,
                'var_feature_m3': var_f_m3,
                'avg_feature_m4': avg_f_m4,
                'var_feature_m4': var_f_m4,
                'avg_row_m1': avg_r_m1,
                'var_row_m1': var_r_m1,
                'avg_row_m2': avg_r_m2,
                'var_row_m2': var_r_m2,
                'avg_row_m3': avg_r_m3,
                'var_row_m3': var_r_m3,
                'avg_row_m4': avg_r_m4,
                'var_row_m4': var_r_m4
            }
        except Exception as e:
            print(f"Error calculating feature moments: {str(e)}")
            # Return 0 for values that couldn't be calculated (instead of NaN)
            return {
                'avg_feature_m1': 0.0, 'var_feature_m1': 0.0,
                'avg_feature_m2': 0.0, 'var_feature_m2': 0.0,
                'avg_feature_m3': 0.0, 'var_feature_m3': 0.0,
                'avg_feature_m4': 0.0, 'var_feature_m4': 0.0,
                'avg_row_m1': 0.0, 'var_row_m1': 0.0,
                'avg_row_m2': 0.0, 'var_row_m2': 0.0,
                'avg_row_m3': 0.0, 'var_row_m3': 0.0,
                'avg_row_m4': 0.0, 'var_row_m4': 0.0
            }
            
    except Exception as e:
        print(f"Error in _calculate_feature_moments_and_variances: {str(e)}")
        # Return 0 for all metrics if an error occurred (instead of NaN)
        return {
            'avg_feature_m1': 0.0, 'var_feature_m1': 0.0,
            'avg_feature_m2': 0.0, 'var_feature_m2': 0.0,
            'avg_feature_m3': 0.0, 'var_feature_m3': 0.0,
            'avg_feature_m4': 0.0, 'var_feature_m4': 0.0,
            'avg_row_m1': 0.0, 'var_row_m1': 0.0,
            'avg_row_m2': 0.0, 'var_row_m2': 0.0,
            'avg_row_m3': 0.0, 'var_row_m3': 0.0,
            'avg_row_m4': 0.0, 'var_row_m4': 0.0
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
        Values are clipped to prevent numerical overflow.
    """
    try:
        # Only use numeric columns for statistics (include all numeric types)
        X_numeric = X.select_dtypes(include=[np.number])
        
        # Handle NaN values by replacing with column means
        for col in X_numeric.columns:
            if X_numeric[col].isna().any():
                mean_val = X_numeric[col].mean()
                if pd.isna(mean_val):  # If mean is NaN (all values are NaN)
                    X_numeric[col] = X_numeric[col].fillna(0)
                else:
                    X_numeric[col] = X_numeric[col].fillna(mean_val)
        
        # Check if we have valid data
        if X_numeric.empty or X_numeric.shape[1] == 0:
            return {
                'avg_row_m1': 0.0, 'var_row_m1': 0.0,
                'avg_row_m2': 0.0, 'var_row_m2': 0.0,
                'avg_row_m3': 0.0, 'var_row_m3': 0.0,
                'avg_row_m4': 0.0, 'var_row_m4': 0.0
            }
        
        # Clip extreme values to prevent overflow
        X_clipped = X_numeric.clip(-1e6, 1e6)
        
        # Calculate moments for each row with clipping
        moment_1 = X_clipped.apply(lambda x: x.mean(), axis=1)
        moment_2 = X_clipped.apply(lambda x: (x ** 2).mean(), axis=1)
        moment_3 = X_clipped.apply(lambda x: (x ** 3).mean(), axis=1)
        moment_4 = X_clipped.apply(lambda x: (x ** 4).mean(), axis=1)
        
        # Helper function to safely calculate mean and variance
        def safe_mean_var(series, max_val=1e12):
            """Safely calculate mean and variance with clipping."""
            # Remove any inf/nan values
            clean_series = series.replace([np.inf, -np.inf], np.nan).dropna()
            if len(clean_series) == 0:
                return 0.0, 0.0
            
            # Clip to prevent extreme values
            clipped_series = clean_series.clip(-max_val, max_val)
            
            mean_val = float(clipped_series.mean())
            var_val = float(clipped_series.var()) if len(clipped_series) > 1 else 0.0
            
            # Final safety check
            mean_val = np.clip(mean_val, -max_val, max_val)
            var_val = np.clip(var_val, 0.0, max_val)  # Variance should be non-negative
            
            return mean_val, var_val
        
        # Calculate statistics with safety checks
        avg_m1, var_m1 = safe_mean_var(moment_1)
        avg_m2, var_m2 = safe_mean_var(moment_2)
        avg_m3, var_m3 = safe_mean_var(moment_3)
        avg_m4, var_m4 = safe_mean_var(moment_4)
        
        return {
            'avg_row_m1': avg_m1,
            'var_row_m1': var_m1,
            'avg_row_m2': avg_m2,
            'var_row_m2': var_m2,
            'avg_row_m3': avg_m3,
            'var_row_m3': var_m3,
            'avg_row_m4': avg_m4,
            'var_row_m4': var_m4
        }
        
    except Exception as e:
        print(f"Error in _calculate_row_moments_and_variances: {str(e)}")
        # Return 0 for all metrics if an error occurred
        return {
            'avg_row_m1': 0.0, 'var_row_m1': 0.0,
            'avg_row_m2': 0.0, 'var_row_m2': 0.0,
            'avg_row_m3': 0.0, 'var_row_m3': 0.0,
            'avg_row_m4': 0.0, 'var_row_m4': 0.0
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