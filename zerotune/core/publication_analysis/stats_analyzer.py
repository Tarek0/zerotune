"""
Statistical Analysis Module for Publication Results

This module performs statistical analysis for publication-ready benchmark results,
including paired t-tests for significance testing between different methods.

Key Comparisons:
1. Zero-shot vs Random hyperparameters
2. Zero-shot vs Standard Optuna TPE  
3. Warm-started vs Standard Optuna TPE
"""

import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
from typing import Dict, List, Tuple, Optional
import ast


class PublicationStatsAnalyzer:
    """
    Statistical analyzer for publication benchmark results.
    
    Performs paired t-tests to evaluate significant differences between methods
    and generates statistical summaries for publication tables.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize the statistical analyzer.
        
        Args:
            alpha: Significance level for statistical tests (default: 0.05)
        """
        self.alpha = alpha
    
    def perform_publication_comparisons(self, df_results: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Perform statistical comparison for publication analysis.
        
        Only performs Zero-shot vs Random comparison.
        For Optuna methods, convergence analysis is used instead.
        
        Args:
            df_results: DataFrame with benchmark results containing performance scores
            
        Returns:
            Dictionary with comparison results for Zero-shot vs Random
        """
        comparison_results = {}
        
        # Check which columns are available for analysis
        available_columns = df_results.columns.tolist()
        
        # Zero-shot vs Random comparison
        if 'auc_predicted' in available_columns and 'auc_random' in available_columns:
            print("🔍 Performing ZT vs Random comparison...")
            comparison_results['zeroshot_vs_random'] = self._perform_paired_comparison(
                df_results, 'auc_predicted', 'auc_random', 
                'ZT', 'Random'
            )
        
        # Note: Removed Optuna comparisons (zeroshot_vs_standard_optuna, warmstart_vs_standard_optuna)
        # For Optuna methods, we focus on convergence analysis instead of per-dataset comparisons
        
        return comparison_results
    
    def _perform_paired_comparison(
        self, 
        df: pd.DataFrame, 
        column1: str, 
        column2: str,
        method1_name: str,
        method2_name: str
    ) -> pd.DataFrame:
        """
        Perform paired t-test comparison between two methods.
        
        Args:
            df: DataFrame with results
            column1: Column name for first method
            column2: Column name for second method  
            method1_name: Display name for first method
            method2_name: Display name for second method
            
        Returns:
            DataFrame with statistical test results
        """
        print(f"--- Statistical Analysis: {method1_name} vs {method2_name} ---")
        
        results = []
        
        # Filter out rows with missing data
        valid_data = df.dropna(subset=[column1, column2])
        
        if len(valid_data) == 0:
            print(f"⚠️  No valid paired data found for {method1_name} vs {method2_name}")
            return pd.DataFrame()
        
        for i, row in valid_data.iterrows():
            try:
                # Handle both single values and list representations
                scores1 = self._extract_scores(row[column1])
                scores2 = self._extract_scores(row[column2])
                
                # Check if we have individual seed scores available (better than summary stats)
                scores1_col = f"{column1}_scores"
                scores2_col = f"{column2}_scores"
                
                if scores1_col in row.index and scores2_col in row.index and not pd.isna(row[scores1_col]) and not pd.isna(row[scores2_col]):
                    # Use individual seed scores if available
                    scores1 = self._extract_scores(row[scores1_col])
                    scores2 = self._extract_scores(row[scores2_col])
                
                if scores1 is None or scores2 is None:
                    continue
                
                # Ensure we have paired data
                if len(scores1) != len(scores2):
                    print(f"⚠️  Mismatched score lengths for dataset {row.get('dataset_id', i)}: {len(scores1)} vs {len(scores2)}")
                    continue
                
                # Calculate means
                mean1 = np.mean(scores1)
                mean2 = np.mean(scores2)
                
                # Perform statistical test
                if len(scores1) > 1 and len(scores2) > 1:
                    # We have raw scores - use paired t-test
                    t_stat, p_value = ttest_rel(scores1, scores2)
                elif len(scores1) == 1 and len(scores2) == 1:
                    # We have single aggregated scores - check if we have std and n_seeds for proper t-test
                    std1_col = None
                    std2_col = None
                    n_seeds_col = None
                    
                    # Look for standard deviation columns
                    for col in row.index:
                        # For column1 (e.g., 'auc_predicted' -> 'auc_predicted_std')
                        if col == f"{column1}_std":
                            std1_col = col
                        # For column2 (e.g., 'auc_random' -> 'auc_random_std')  
                        elif col == f"{column2}_std":
                            std2_col = col
                        elif 'n_seeds' in col.lower():
                            n_seeds_col = col
                    
                    # If we have std and n_seeds, perform two-sample t-test with summary statistics
                    if std1_col and std2_col and n_seeds_col and not pd.isna(row[std1_col]) and not pd.isna(row[std2_col]):
                        try:
                            from scipy.stats import ttest_ind_from_stats
                            
                            std1 = row[std1_col]
                            std2 = row[std2_col] 
                            n_seeds = int(row[n_seeds_col])
                            
                            # Perform two-sample t-test with summary statistics (Welch's t-test)
                            t_stat, p_value = ttest_ind_from_stats(
                                mean1=mean1, std1=std1, nobs1=n_seeds,
                                mean2=mean2, std2=std2, nobs2=n_seeds,
                                equal_var=False  # Welch's t-test (unequal variances)
                            )
                        except Exception as e:
                            print(f"⚠️  Error in summary statistics t-test for dataset {row.get('dataset_id', i)}: {str(e)}")
                            t_stat, p_value = np.nan, np.nan
                    else:
                        # Single value case - no statistical test possible
                        t_stat, p_value = np.nan, np.nan
                else:
                    # Single value case - no statistical test possible
                    t_stat, p_value = np.nan, np.nan
                
                # Calculate percentage uplift
                uplift_pct = ((mean1 - mean2) / mean2) * 100 if mean2 != 0 else np.nan
                
                # Determine significance and best method
                is_significant = p_value < self.alpha if not np.isnan(p_value) else False
                best_method = method1_name if mean1 > mean2 else method2_name
                
                result = {
                    'dataset_id': row.get('dataset_id', f'dataset_{i}'),
                    'dataset_name': row.get('dataset_name', f'Dataset_{i}'),
                    f'{method1_name}_mean': mean1,
                    f'{method2_name}_mean': mean2,
                    'best_method': best_method,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': is_significant,
                    'uplift_pct': uplift_pct,
                    'n_samples': len(scores1)
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"⚠️  Error processing row {i}: {str(e)}")
                continue
        
        if not results:
            print(f"❌ No valid results generated for {method1_name} vs {method2_name}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Print summary statistics
        self._print_comparison_summary(results_df, method1_name, method2_name)
        
        return results_df
    
    def _extract_scores(self, value) -> Optional[List[float]]:
        """
        Extract numerical scores from various input formats.
        
        Args:
            value: Input value (could be float, string representation of list, etc.)
            
        Returns:
            List of float scores or None if extraction fails
        """
        if pd.isna(value):
            return None
        
        # If it's already a number, return as single-item list
        if isinstance(value, (int, float)):
            return [float(value)]
        
        # If it's a string, try to parse as list
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    return [float(x) for x in parsed]
                else:
                    return [float(parsed)]
            except (ValueError, SyntaxError):
                try:
                    return [float(value)]
                except ValueError:
                    return None
        
        # If it's already a list
        if isinstance(value, list):
            try:
                return [float(x) for x in value]
            except (ValueError, TypeError):
                return None
        
        return None
    
    def _print_comparison_summary(self, results_df: pd.DataFrame, method1_name: str, method2_name: str):
        """Print summary statistics for a comparison."""
        if len(results_df) == 0:
            return
        
        # Calculate summary statistics
        significant_results = results_df[results_df['significant'] == True]
        positive_uplifts = results_df[results_df['uplift_pct'] > 0]
        
        mean_uplift = results_df['uplift_pct'].mean()
        std_uplift = results_df['uplift_pct'].std()
        
        method1_wins = len(results_df[results_df['best_method'] == method1_name])
        total_comparisons = len(results_df)
        win_rate = (method1_wins / total_comparisons) * 100
        
        print(f"📊 Summary Statistics:")
        print(f"   Total datasets: {total_comparisons}")
        print(f"   {method1_name} wins: {method1_wins}/{total_comparisons} ({win_rate:.1f}%)")
        print(f"   Significant differences: {len(significant_results)}/{total_comparisons} ({len(significant_results)/total_comparisons*100:.1f}%)")
        print(f"   Average uplift: {mean_uplift:.2f}% ± {std_uplift:.2f}%")
        print(f"   Positive uplifts: {len(positive_uplifts)}/{total_comparisons} ({len(positive_uplifts)/total_comparisons*100:.1f}%)")
        print()
    
    def generate_statistical_summary(self, comparison_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate an overall statistical summary across all comparisons.
        
        Args:
            comparison_results: Dictionary of comparison results from perform_publication_comparisons
            
        Returns:
            DataFrame with summary statistics for each comparison
        """
        summary_data = []
        
        for comparison_name, results_df in comparison_results.items():
            if len(results_df) == 0:
                continue
            
            # Extract method names from comparison name
            if comparison_name == 'zeroshot_vs_random':
                method1, method2 = 'ZT', 'Random'
            else:
                # Default fallback for any other comparison types
                method1, method2 = 'Method1', 'Method2'
            
            # Calculate summary metrics
            total_datasets = len(results_df)
            method1_wins = len(results_df[results_df['best_method'] == method1])
            significant_count = len(results_df[results_df['significant'] == True])
            mean_uplift = results_df['uplift_pct'].mean()
            std_uplift = results_df['uplift_pct'].std()
            
            summary_data.append({
                'comparison': comparison_name,
                'method1': method1,
                'method2': method2,
                'total_datasets': total_datasets,
                'method1_wins': method1_wins,
                'win_rate_pct': (method1_wins / total_datasets) * 100,
                'significant_differences': significant_count,
                'significance_rate_pct': (significant_count / total_datasets) * 100,
                'mean_uplift_pct': mean_uplift,
                'std_uplift_pct': std_uplift
            })
        
        return pd.DataFrame(summary_data) 