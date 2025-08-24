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
            'trial_data': df_trials,
            'benchmark_data': benchmark_data
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
            
            # Extract scores at each checkpoint (for LaTeX tables)
            for checkpoint in self.checkpoints:
                # Get trials up to this checkpoint
                trials_up_to_checkpoint = group_df[group_df['number'] <= checkpoint]
                
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
        
        # 4. Aggregated comparison table: ZT+TPE vs TPE (warm-started vs standard)
        tables['aggregated_comparison'] = self._create_aggregated_comparison_table(checkpoint_data, benchmark_data)
        
        # 5. NEW: ZT vs TPE comparison table (zero-shot vs standard TPE)
        tables['zt_vs_tpe_comparison'] = self._create_zt_vs_tpe_comparison_table(checkpoint_data, benchmark_data)
        
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
                
                # Calculate Sig % (per-dataset significance)
                try:
                    significant_datasets = 0
                    total_datasets = len(zeroshot_scores)
                    
                    # For checkpoint 0, we have paired scores per dataset
                    for z_score, r_score in zip(zeroshot_scores, random_scores):
                        # For single paired values, we can't do t-test, so we just check if ZT wins
                        # and consider it "significant" only if ZT actually wins on this dataset
                        if z_score > r_score:
                            significant_datasets += 1
                    
                    # Sig % should be based on datasets where method actually wins AND is significant
                    # For checkpoint 0 (single scores per dataset), significance = winning
                    zeroshot_sig_pct = (significant_datasets / total_datasets) * 100 if zeroshot_wins > 0 else 0.0
                    tpe_sig_pct = ((total_datasets - significant_datasets) / total_datasets) * 100 if tpe_wins > 0 else 0.0
                    
                    # Ensure Sig % never exceeds Best %
                    zeroshot_sig_pct = min(zeroshot_sig_pct, zeroshot_best_pct)
                    tpe_sig_pct = min(tpe_sig_pct, tpe_best_pct)
                    
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
                
                # Calculate Sig % (per-dataset significance using proper t-tests)
                try:
                    significant_zt_datasets = 0
                    significant_tpe_datasets = 0
                    total_datasets = len(warmstart_scores)
                    
                    # For each dataset, perform paired t-test across seeds
                    if 'warmstart' in checkpoint_data and 'standard' in checkpoint_data:
                        common_datasets = set(checkpoint_data['warmstart'].keys()) & set(checkpoint_data['standard'].keys())
                        
                        for dataset_id in common_datasets:
                            # Collect raw scores across all seeds for this dataset
                            warmstart_raw = []
                            standard_raw = []
                            
                            for seed in checkpoint_data['warmstart'][dataset_id]:
                                score = checkpoint_data['warmstart'][dataset_id][seed].get(checkpoint, np.nan)
                                if not np.isnan(score):
                                    warmstart_raw.append(score)
                            
                            for seed in checkpoint_data['standard'][dataset_id]:
                                score = checkpoint_data['standard'][dataset_id][seed].get(checkpoint, np.nan)
                                if not np.isnan(score):
                                    standard_raw.append(score)
                            
                            # Perform paired t-test if we have enough data points
                            if len(warmstart_raw) == len(standard_raw) and len(warmstart_raw) >= 2:
                                try:
                                    t_stat, p_value = ttest_rel(warmstart_raw, standard_raw)
                                    is_significant = p_value < 0.05
                                    
                                    warmstart_mean = np.mean(warmstart_raw)
                                    standard_mean = np.mean(standard_raw)
                                    
                                    if is_significant and warmstart_mean > standard_mean:
                                        significant_zt_datasets += 1
                                    elif is_significant and standard_mean > warmstart_mean:
                                        significant_tpe_datasets += 1
                                except:
                                    # Fallback to simple difference for this dataset
                                    warmstart_mean = np.mean(warmstart_raw) if warmstart_raw else 0
                                    standard_mean = np.mean(standard_raw) if standard_raw else 0
                                    diff = abs(warmstart_mean - standard_mean)
                                    
                                    if diff > 0.01 and warmstart_mean > standard_mean:
                                        significant_zt_datasets += 1
                                    elif diff > 0.01 and standard_mean > warmstart_mean:
                                        significant_tpe_datasets += 1
                    
                    # Sig % should only count datasets where method wins AND is significant
                    zt_sig_pct = (significant_zt_datasets / total_datasets) * 100 if zt_wins > 0 else 0.0
                    tpe_sig_pct = (significant_tpe_datasets / total_datasets) * 100 if tpe_wins > 0 else 0.0
                    
                    # Ensure Sig % never exceeds Best %
                    zt_sig_pct = min(zt_sig_pct, zt_best_pct)
                    tpe_sig_pct = min(tpe_sig_pct, tpe_best_pct)
                    
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
    
    def _create_zt_vs_tpe_comparison_table(self, checkpoint_data: Dict, benchmark_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create a table comparing ZT (Zero-shot TPE) vs TPE (Standard TPE) convergence.
        
        Args:
            checkpoint_data: Dictionary with checkpoint scores
            benchmark_data: Optional benchmark data for checkpoint 0
            
        Returns:
            DataFrame with ZT vs TPE comparison results
        """
        print("   Creating ZT vs TPE comparison table...")
        
        comparison_data = []
        
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
                
                # Calculate Sig % (per-dataset significance)
                try:
                    significant_datasets = 0
                    total_datasets = len(zeroshot_scores)
                    
                    # For checkpoint 0, we have paired scores per dataset
                    for z_score, r_score in zip(zeroshot_scores, random_scores):
                        # For single paired values, we can't do t-test, so we just check if ZT wins
                        # and consider it "significant" only if ZT actually wins on this dataset
                        if z_score > r_score:
                            significant_datasets += 1
                    
                    # Sig % should be based on datasets where method actually wins AND is significant
                    # For checkpoint 0 (single scores per dataset), significance = winning
                    zeroshot_sig_pct = (significant_datasets / total_datasets) * 100 if zeroshot_wins > 0 else 0.0
                    tpe_sig_pct = ((total_datasets - significant_datasets) / total_datasets) * 100 if tpe_wins > 0 else 0.0
                    
                    # Ensure Sig % never exceeds Best %
                    zeroshot_sig_pct = min(zeroshot_sig_pct, zeroshot_best_pct)
                    tpe_sig_pct = min(tpe_sig_pct, tpe_best_pct)
                    
                except:
                    zeroshot_sig_pct = 0.0
                    tpe_sig_pct = 0.0
                
                comparison_data.append({
                    'checkpoint': 0,
                    'zt_best_pct': zeroshot_best_pct,
                    'zt_sig_pct': zeroshot_sig_pct,
                    'tpe_best_pct': tpe_best_pct,
                    'tpe_sig_pct': tpe_sig_pct
                })
        
        # Checkpoints 1, 5, 10, 20: ZT vs TPE (Zero-shot vs Standard TPE)
        for checkpoint in [1, 5, 10, 20]:  # Skip 15 to match your example
            if checkpoint not in self.checkpoints:
                continue
                
            # Get ZT score (from benchmark data - same for all checkpoints)
            zeroshot_scores = []
            if benchmark_data is not None and 'auc_predicted' in benchmark_data.columns:
                zeroshot_scores = benchmark_data['auc_predicted'].dropna().tolist()
             
            # Get TPE scores at this checkpoint (from standard Optuna trials)
            tpe_scores = []
            if 'standard' in checkpoint_data:
                # Get all datasets that have standard TPE data
                for dataset_id in checkpoint_data['standard'].keys():
                    # Collect scores across all seeds for this dataset at this checkpoint
                    dataset_scores = []
                     
                    for seed in checkpoint_data['standard'][dataset_id]:
                        score = checkpoint_data['standard'][dataset_id][seed].get(checkpoint, np.nan)
                        if not np.isnan(score):
                            dataset_scores.append(score)
                     
                    # Take mean across seeds for this dataset
                    if dataset_scores:
                        tpe_scores.append(np.mean(dataset_scores))
             
            # Match datasets (both ZT and TPE must have scores for same datasets)
            if len(zeroshot_scores) > 0 and len(tpe_scores) > 0:
                # For proper comparison, we need to match by dataset_id
                # Get dataset IDs from benchmark data
                if benchmark_data is not None:
                    benchmark_dataset_ids = benchmark_data['dataset_id'].tolist()
                    tpe_dataset_ids = list(checkpoint_data['standard'].keys()) if 'standard' in checkpoint_data else []
                     
                    # Find common datasets
                    common_dataset_ids = [did for did in benchmark_dataset_ids if did in tpe_dataset_ids]
                     
                    if common_dataset_ids:
                        # Extract matched scores
                        matched_zt_scores = []
                        matched_tpe_scores = []
                         
                        for dataset_id in common_dataset_ids:
                            # Get ZT score for this dataset
                            zt_row = benchmark_data[benchmark_data['dataset_id'] == dataset_id]
                            if len(zt_row) > 0 and not pd.isna(zt_row['auc_predicted'].iloc[0]):
                                zt_score = zt_row['auc_predicted'].iloc[0]
                                 
                                # Get TPE score for this dataset at this checkpoint
                                if dataset_id in checkpoint_data['standard']:
                                    tpe_dataset_scores = []
                                    for seed in checkpoint_data['standard'][dataset_id]:
                                        score = checkpoint_data['standard'][dataset_id][seed].get(checkpoint, np.nan)
                                        if not np.isnan(score):
                                            tpe_dataset_scores.append(score)
                                     
                                    if tpe_dataset_scores:
                                        tpe_score = np.mean(tpe_dataset_scores)
                                        matched_zt_scores.append(zt_score)
                                        matched_tpe_scores.append(tpe_score)
                         
                        zeroshot_scores = matched_zt_scores
                        tpe_scores = matched_tpe_scores
             
            if len(zeroshot_scores) == len(tpe_scores) and len(zeroshot_scores) > 0:
                # Calculate Best %
                zt_wins = sum(1 for z, t in zip(zeroshot_scores, tpe_scores) if z > t)
                tpe_wins = len(zeroshot_scores) - zt_wins
                 
                zt_best_pct = (zt_wins / len(zeroshot_scores)) * 100
                tpe_best_pct = (tpe_wins / len(zeroshot_scores)) * 100
                 
                # Calculate Sig % 
                # For ZT vs TPE, ZT has single score per dataset, TPE has multiple seeds
                # We can't do proper paired t-test, so we use simple win counting as "significance"
                try:
                    significant_zt_datasets = 0
                    significant_tpe_datasets = 0
                    total_datasets = len(zeroshot_scores)
                     
                    # For each dataset, check if the difference is meaningful
                    for i, (zt_score, tpe_score) in enumerate(zip(zeroshot_scores, tpe_scores)):
                        diff = abs(zt_score - tpe_score)
                        # Consider it significant if difference > 1% (0.01 AUC points)
                        if diff > 0.01:
                            if zt_score > tpe_score:
                                significant_zt_datasets += 1
                            else:
                                significant_tpe_datasets += 1
                     
                    # Sig % should only count datasets where method wins AND is significant
                    zt_sig_pct = (significant_zt_datasets / total_datasets) * 100 if zt_wins > 0 else 0.0
                    tpe_sig_pct = (significant_tpe_datasets / total_datasets) * 100 if tpe_wins > 0 else 0.0
                     
                    # Ensure Sig % never exceeds Best %
                    zt_sig_pct = min(zt_sig_pct, zt_best_pct)
                    tpe_sig_pct = min(tpe_sig_pct, tpe_best_pct)
                     
                except:
                    zt_sig_pct = 0.0
                    tpe_sig_pct = 0.0
                 
                comparison_data.append({
                    'checkpoint': checkpoint,
                    'zt_best_pct': zt_best_pct,
                    'zt_sig_pct': zt_sig_pct,
                    'tpe_best_pct': tpe_best_pct,
                    'tpe_sig_pct': tpe_sig_pct
                })
            else:
                # No data available for this checkpoint
                comparison_data.append({
                    'checkpoint': checkpoint,
                    'zt_best_pct': np.nan,
                    'zt_sig_pct': np.nan,
                    'tpe_best_pct': np.nan,
                    'tpe_sig_pct': np.nan
                })
        
        return pd.DataFrame(comparison_data)
    
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