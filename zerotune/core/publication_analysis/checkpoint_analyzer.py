"""
Checkpoint Analysis Module for Publication Results

This module analyzes Optuna trial convergence by extracting performance scores
at specific trial checkpoints (1, 5, 10, 15, 20 trials) to evaluate how
different methods perform as optimization progresses.
"""

import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
from typing import Dict, List, Tuple, Optional, Union
import os


class CheckpointAnalyzer:
    """
    Analyzer for Optuna trial checkpoint convergence analysis.
    
    Extracts performance scores at specific trial numbers to evaluate
    convergence behavior of different optimization methods.
    """
    
    def __init__(self, checkpoints: List[int] = [1, 5, 10, 15, 20]):
        """
        Initialize the checkpoint analyzer.
        
        Args:
            checkpoints: List of trial numbers to analyze (default: [1, 5, 10, 15, 20])
        """
        self.checkpoints = checkpoints
    
    def analyze_convergence(
        self, 
        warmstart_path: str, 
        standard_path: str,
        algorithm_name: str = "DecisionTree",
        benchmark_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Analyze convergence behavior from trial data files.
        
        Args:
            warmstart_path: Path to warmstart trial data CSV
            standard_path: Path to standard trial data CSV
            algorithm_name: Name of the algorithm for display
            benchmark_data: Optional benchmark data for checkpoint 0 comparison
            
        Returns:
            Dictionary containing convergence analysis results
        """
        print(f"🔍 Analyzing {algorithm_name} convergence at checkpoints: {self.checkpoints}")
        print("-" * 60)
        
        # Load trial data
        df_trials = self.load_trial_csvs(warmstart_path, standard_path)
        
        if df_trials is None or len(df_trials) == 0:
            print("❌ No trial data available for convergence analysis")
            return {}
        
        print(f"✅ Loaded {len(df_trials)} total trials")
        print(f"Methods: {df_trials['method'].unique()}")
        print(f"Datasets: {sorted(df_trials['dataset_id'].unique())}")
        print(f"Seeds: {sorted(df_trials['seed'].unique())}")
        print()
        
        # Extract checkpoint scores
        checkpoint_results = self.extract_checkpoint_scores(df_trials)
        
        if not checkpoint_results:
            print("❌ Could not extract checkpoint scores")
            return {}
        
        # Generate convergence comparison tables
        convergence_tables = self.generate_convergence_tables(checkpoint_results, benchmark_data)
        
        return {
            'checkpoint_scores': checkpoint_results,
            'convergence_tables': convergence_tables,
            'trial_data': df_trials
        }
    
    def load_trial_csvs(self, warmstart_path: str, standard_path: str) -> Optional[pd.DataFrame]:
        """
        Load and combine trial data from warmstart and standard CSV files.
        
        Args:
            warmstart_path: Path to warmstart trial data CSV
            standard_path: Path to standard trial data CSV
            
        Returns:
            Combined DataFrame with trial data or None if loading fails
        """
        try:
            dfs = []
            
            if os.path.exists(warmstart_path):
                df_warmstart = pd.read_csv(warmstart_path)
                # Ensure method column exists
                if 'method' not in df_warmstart.columns:
                    df_warmstart['method'] = 'warmstart'
                dfs.append(df_warmstart)
                print(f"✅ Loaded {len(df_warmstart)} warmstart trials")
            else:
                print(f"⚠️  Warmstart trials file not found: {warmstart_path}")
            
            if os.path.exists(standard_path):
                df_standard = pd.read_csv(standard_path)
                # Ensure method column exists
                if 'method' not in df_standard.columns:
                    df_standard['method'] = 'standard'
                dfs.append(df_standard)
                print(f"✅ Loaded {len(df_standard)} standard trials")
            else:
                print(f"⚠️  Standard trials file not found: {standard_path}")
            
            if not dfs:
                print("❌ No trial data files found")
                return None
            
            # Combine all trial data
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # Ensure required columns exist
            required_columns = ['number', 'value', 'dataset_id', 'seed', 'method']
            missing_columns = [col for col in required_columns if col not in combined_df.columns]
            
            if missing_columns:
                print(f"❌ Missing required columns: {missing_columns}")
                return None
            
            return combined_df
            
        except Exception as e:
            print(f"❌ Error loading trial data: {str(e)}")
            return None
    
    def extract_checkpoint_scores(self, df_trials: pd.DataFrame) -> Dict[str, Dict]:
        """
        Extract performance scores at specific checkpoints.
        
        Args:
            df_trials: DataFrame with trial data
            
        Returns:
            Dictionary with checkpoint scores organized by method, dataset, and seed
        """
        print("📊 Extracting checkpoint scores...")
        
        checkpoint_data = {}
        
        # Group by method, dataset, and seed
        grouped = df_trials.groupby(['method', 'dataset_id', 'seed'])
        
        total_groups = len(grouped)
        processed_groups = 0
        
        for (method, dataset_id, seed), group_df in grouped:
            processed_groups += 1
            
            # Sort by trial number
            group_df = group_df.sort_values('number')
            
            # Initialize data structure if needed
            if method not in checkpoint_data:
                checkpoint_data[method] = {}
            if dataset_id not in checkpoint_data[method]:
                checkpoint_data[method][dataset_id] = {}
            if seed not in checkpoint_data[method][dataset_id]:
                checkpoint_data[method][dataset_id][seed] = {}
            
            # Extract scores at each checkpoint
            for checkpoint in self.checkpoints:
                # Get trials up to this checkpoint
                trials_up_to_checkpoint = group_df[group_df['number'] < checkpoint]
                
                if len(trials_up_to_checkpoint) == 0:
                    # No trials available at this checkpoint
                    best_score = np.nan
                else:
                    # Get the best score achieved so far
                    best_score = trials_up_to_checkpoint['value'].max()
                
                checkpoint_data[method][dataset_id][seed][checkpoint] = best_score
            
            # Progress indicator
            if processed_groups % 10 == 0 or processed_groups == total_groups:
                print(f"   Processed {processed_groups}/{total_groups} method-dataset-seed combinations")
        
        print(f"✅ Extracted checkpoint scores for {len(checkpoint_data)} methods")
        return checkpoint_data
    
    def generate_convergence_tables(self, checkpoint_data: Dict, benchmark_data: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
        """
        Generate convergence comparison tables from checkpoint data.
        
        Args:
            checkpoint_data: Dictionary with checkpoint scores
            benchmark_data: Optional benchmark data for checkpoint 0 comparison
            
        Returns:
            Dictionary with convergence tables
        """
        print("📈 Generating convergence tables...")
        
        tables = {}
        
        # 1. Method comparison table (average across all datasets and seeds)
        tables['method_comparison'] = self._create_method_comparison_table(checkpoint_data)
        
        # 2. Per-dataset convergence table
        tables['per_dataset_convergence'] = self._create_per_dataset_table(checkpoint_data)
        
        # 3. Statistical summary table
        tables['statistical_summary'] = self._create_statistical_summary_table(checkpoint_data)
        
        # 4. Aggregated comparison table (Best % and Sig % format)
        tables['aggregated_comparison'] = self._create_aggregated_comparison_table(checkpoint_data, benchmark_data)
        
        return tables
    
    def _create_method_comparison_table(self, checkpoint_data: Dict) -> pd.DataFrame:
        """Create a table comparing methods across all checkpoints."""
        print("   Creating method comparison table...")
        
        comparison_data = []
        
        for checkpoint in self.checkpoints:
            row_data = {'checkpoint': checkpoint}
            
            for method in checkpoint_data.keys():
                # Collect all scores for this method at this checkpoint
                all_scores = []
                
                for dataset_id in checkpoint_data[method]:
                    for seed in checkpoint_data[method][dataset_id]:
                        score = checkpoint_data[method][dataset_id][seed].get(checkpoint, np.nan)
                        if not np.isnan(score):
                            all_scores.append(score)
                
                if all_scores:
                    mean_score = np.mean(all_scores)
                    std_score = np.std(all_scores)
                    row_data[f'{method}_mean'] = mean_score
                    row_data[f'{method}_std'] = std_score
                    row_data[f'{method}_count'] = len(all_scores)
                else:
                    row_data[f'{method}_mean'] = np.nan
                    row_data[f'{method}_std'] = np.nan
                    row_data[f'{method}_count'] = 0
            
            comparison_data.append(row_data)
        
        return pd.DataFrame(comparison_data)
    
    def _create_per_dataset_table(self, checkpoint_data: Dict) -> pd.DataFrame:
        """Create a table showing convergence per dataset."""
        print("   Creating per-dataset convergence table...")
        
        dataset_data = []
        
        # Get all unique datasets
        all_datasets = set()
        for method in checkpoint_data:
            all_datasets.update(checkpoint_data[method].keys())
        
        for dataset_id in sorted(all_datasets):
            for checkpoint in self.checkpoints:
                row_data = {
                    'dataset_id': dataset_id,
                    'checkpoint': checkpoint
                }
                
                for method in checkpoint_data.keys():
                    if dataset_id in checkpoint_data[method]:
                        # Collect scores across all seeds for this dataset
                        scores = []
                        for seed in checkpoint_data[method][dataset_id]:
                            score = checkpoint_data[method][dataset_id][seed].get(checkpoint, np.nan)
                            if not np.isnan(score):
                                scores.append(score)
                        
                        if scores:
                            row_data[f'{method}_mean'] = np.mean(scores)
                            row_data[f'{method}_std'] = np.std(scores)
                        else:
                            row_data[f'{method}_mean'] = np.nan
                            row_data[f'{method}_std'] = np.nan
                    else:
                        row_data[f'{method}_mean'] = np.nan
                        row_data[f'{method}_std'] = np.nan
                
                dataset_data.append(row_data)
        
        return pd.DataFrame(dataset_data)
    
    def _create_statistical_summary_table(self, checkpoint_data: Dict) -> pd.DataFrame:
        """Create statistical summary comparing methods at each checkpoint."""
        print("   Creating statistical summary table...")
        
        summary_data = []
        methods = list(checkpoint_data.keys())
        
        if len(methods) < 2:
            print("   ⚠️  Need at least 2 methods for statistical comparison")
            return pd.DataFrame()
        
        # Compare each pair of methods at each checkpoint
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i >= j:  # Avoid duplicate comparisons
                    continue
                
                for checkpoint in self.checkpoints:
                    # Collect paired scores
                    scores1, scores2 = [], []
                    
                    # Find common dataset-seed combinations
                    common_combinations = set()
                    if method1 in checkpoint_data and method2 in checkpoint_data:
                        datasets1 = set(checkpoint_data[method1].keys())
                        datasets2 = set(checkpoint_data[method2].keys())
                        common_datasets = datasets1.intersection(datasets2)
                        
                        for dataset_id in common_datasets:
                            seeds1 = set(checkpoint_data[method1][dataset_id].keys())
                            seeds2 = set(checkpoint_data[method2][dataset_id].keys())
                            common_seeds = seeds1.intersection(seeds2)
                            
                            for seed in common_seeds:
                                score1 = checkpoint_data[method1][dataset_id][seed].get(checkpoint, np.nan)
                                score2 = checkpoint_data[method2][dataset_id][seed].get(checkpoint, np.nan)
                                
                                if not np.isnan(score1) and not np.isnan(score2):
                                    scores1.append(score1)
                                    scores2.append(score2)
                    
                    if len(scores1) > 0:
                        mean1, mean2 = np.mean(scores1), np.mean(scores2)
                        std1, std2 = np.std(scores1), np.std(scores2)
                        
                        # Calculate uplift
                        uplift = ((mean1 - mean2) / mean2) * 100 if mean2 != 0 else np.nan
                        
                        # Determine winner
                        winner = method1 if mean1 > mean2 else method2
                        
                        summary_data.append({
                            'checkpoint': checkpoint,
                            'method1': method1,
                            'method2': method2,
                            'method1_mean': mean1,
                            'method1_std': std1,
                            'method2_mean': mean2,
                            'method2_std': std2,
                            'uplift_pct': uplift,
                            'winner': winner,
                            'n_comparisons': len(scores1)
                        })
        
        return pd.DataFrame(summary_data)
    
    def _create_aggregated_comparison_table(self, checkpoint_data: Dict, benchmark_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create an aggregated comparison table with Best % and Sig % columns.
        
        Format matches: ZT2 vs TPE with checkpoints 0, 1, 5, 10, 20
        - Checkpoint 0: Zero-shot vs Random (from benchmark data)
        - Other checkpoints: Warm-started Optuna vs Standard Optuna
        
        Args:
            checkpoint_data: Checkpoint scores from Optuna trials
            benchmark_data: Benchmark results for checkpoint 0
            
        Returns:
            DataFrame with aggregated comparison results
        """
        print("   Creating aggregated comparison table...")
        
        aggregated_data = []
        
        # Checkpoint 0: Zero-shot vs Random (from benchmark data)
        if benchmark_data is not None and 'auc_predicted' in benchmark_data.columns and 'auc_random' in benchmark_data.columns:
            zeroshot_scores = benchmark_data['auc_predicted'].dropna().tolist()
            random_scores = benchmark_data['auc_random'].dropna().tolist()
            
            if len(zeroshot_scores) == len(random_scores) and len(zeroshot_scores) > 0:
                # Calculate Best %
                zeroshot_wins = sum(1 for z, r in zip(zeroshot_scores, random_scores) if z > r)
                tpe_wins = len(zeroshot_scores) - zeroshot_wins
                
                zeroshot_best_pct = (zeroshot_wins / len(zeroshot_scores)) * 100
                tpe_best_pct = (tpe_wins / len(zeroshot_scores)) * 100
                
                # Calculate Sig %
                try:
                    t_stat, p_value = ttest_rel(zeroshot_scores, random_scores)
                    is_significant = p_value < 0.05
                    
                    if is_significant and np.mean(zeroshot_scores) > np.mean(random_scores):
                        zeroshot_sig_pct = 100.0
                        tpe_sig_pct = 0.0
                    elif is_significant and np.mean(random_scores) > np.mean(zeroshot_scores):
                        zeroshot_sig_pct = 0.0
                        tpe_sig_pct = 100.0
                    else:
                        zeroshot_sig_pct = 0.0
                        tpe_sig_pct = 0.0
                except:
                    zeroshot_sig_pct = 0.0
                    tpe_sig_pct = 0.0
                
                aggregated_data.append({
                    'checkpoint': 0,
                    'zt_best_pct': zeroshot_best_pct,
                    'zt_sig_pct': zeroshot_sig_pct,
                    'tpe_best_pct': tpe_best_pct,
                    'tpe_sig_pct': tpe_sig_pct
                })
        
        # Checkpoints 1, 5, 10, 20: Warm-started vs Standard Optuna
        for checkpoint in [1, 5, 10, 20]:  # Skip 15 to match your example
            if checkpoint not in self.checkpoints:
                continue
                
            warmstart_scores = []
            standard_scores = []
            
            # Collect per-dataset scores
            if 'warmstart' in checkpoint_data and 'standard' in checkpoint_data:
                # Get all datasets that have both warmstart and standard data
                common_datasets = set(checkpoint_data['warmstart'].keys()) & set(checkpoint_data['standard'].keys())
                
                for dataset_id in common_datasets:
                    # Collect scores across all seeds for this dataset
                    warmstart_dataset_scores = []
                    standard_dataset_scores = []
                    
                    for seed in checkpoint_data['warmstart'][dataset_id]:
                        score = checkpoint_data['warmstart'][dataset_id][seed].get(checkpoint, np.nan)
                        if not np.isnan(score):
                            warmstart_dataset_scores.append(score)
                    
                    for seed in checkpoint_data['standard'][dataset_id]:
                        score = checkpoint_data['standard'][dataset_id][seed].get(checkpoint, np.nan)
                        if not np.isnan(score):
                            standard_dataset_scores.append(score)
                    
                    # Take mean across seeds for this dataset
                    if warmstart_dataset_scores and standard_dataset_scores:
                        warmstart_scores.append(np.mean(warmstart_dataset_scores))
                        standard_scores.append(np.mean(standard_dataset_scores))
            
            if len(warmstart_scores) == len(standard_scores) and len(warmstart_scores) > 0:
                # Calculate Best %
                zt_wins = sum(1 for w, s in zip(warmstart_scores, standard_scores) if w > s)
                tpe_wins = len(warmstart_scores) - zt_wins
                
                zt_best_pct = (zt_wins / len(warmstart_scores)) * 100
                tpe_best_pct = (tpe_wins / len(warmstart_scores)) * 100
                
                # Calculate Sig %
                try:
                    t_stat, p_value = ttest_rel(warmstart_scores, standard_scores)
                    is_significant = p_value < 0.05
                    
                    if is_significant and np.mean(warmstart_scores) > np.mean(standard_scores):
                        zt_sig_pct = 100.0
                        tpe_sig_pct = 0.0
                    elif is_significant and np.mean(standard_scores) > np.mean(warmstart_scores):
                        zt_sig_pct = 0.0
                        tpe_sig_pct = 100.0
                    else:
                        zt_sig_pct = 0.0
                        tpe_sig_pct = 0.0
                except:
                    zt_sig_pct = 0.0
                    tpe_sig_pct = 0.0
                
                aggregated_data.append({
                    'checkpoint': checkpoint,
                    'zt_best_pct': zt_best_pct,
                    'zt_sig_pct': zt_sig_pct,
                    'tpe_best_pct': tpe_best_pct,
                    'tpe_sig_pct': tpe_sig_pct
                })
            else:
                # No data available for this checkpoint
                aggregated_data.append({
                    'checkpoint': checkpoint,
                    'zt_best_pct': np.nan,
                    'zt_sig_pct': np.nan,
                    'tpe_best_pct': np.nan,
                    'tpe_sig_pct': np.nan
                })
        
        return pd.DataFrame(aggregated_data)
    
    def print_convergence_summary(self, convergence_results: Dict):
        """Print a summary of the convergence analysis."""
        if not convergence_results:
            return
        
        print("\n" + "=" * 60)
        print("📈 CONVERGENCE ANALYSIS SUMMARY")
        print("=" * 60)
        
        if 'method_comparison' in convergence_results.get('convergence_tables', {}):
            method_table = convergence_results['convergence_tables']['method_comparison']
            
            print("🏆 Method Performance at Each Checkpoint:")
            print()
            
            methods = [col.replace('_mean', '') for col in method_table.columns if col.endswith('_mean')]
            
            for _, row in method_table.iterrows():
                checkpoint = int(row['checkpoint'])
                print(f"   Trial {checkpoint}:")
                
                method_scores = []
                for method in methods:
                    mean_col = f'{method}_mean'
                    std_col = f'{method}_std'
                    
                    if mean_col in row and not pd.isna(row[mean_col]):
                        mean_score = row[mean_col]
                        std_score = row[std_col] if std_col in row and not pd.isna(row[std_col]) else 0
                        method_scores.append((method, mean_score, std_score))
                
                # Sort by performance
                method_scores.sort(key=lambda x: x[1], reverse=True)
                
                for rank, (method, mean_score, std_score) in enumerate(method_scores, 1):
                    icon = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
                    print(f"     {icon} {method}: {mean_score:.4f} ± {std_score:.4f}")
                
                print()
        
        print("✅ Convergence analysis complete!")
        print("🎯 Use these results to understand optimization convergence behavior.") 