"""
Results Processor for Publication Analysis

This module provides the main orchestrator class for processing benchmark results
and generating publication-ready statistical analysis and LaTeX tables.
"""

import os
import pandas as pd
from typing import Optional, Dict, Any
from datetime import datetime

from .stats_analyzer import PublicationStatsAnalyzer
from .checkpoint_analyzer import CheckpointAnalyzer
from .latex_generator import LatexTableGenerator


class PublicationResultsProcessor:
    """
    Main orchestrator for publication analysis pipeline.
    
    Processes benchmark CSV files and optional trial data to generate:
    - Statistical comparisons with paired t-tests
    - Checkpoint convergence analysis
    - Summary statistics and significance analysis
    - Publication-ready LaTeX tables
    - Publication-ready output files
    """
    
    def __init__(self, alpha: float = 0.05, output_dir: str = "publication_outputs", checkpoints: list = [1, 5, 10, 15, 20]):
        """
        Initialize the results processor.
        
        Args:
            alpha: Significance level for statistical tests (default: 0.05)
            output_dir: Directory for output files (default: "publication_outputs")
            checkpoints: Trial checkpoints for convergence analysis (default: [1, 5, 10, 15, 20])
        """
        self.alpha = alpha
        self.output_dir = output_dir
        self.checkpoints = checkpoints
        self.stats_analyzer = PublicationStatsAnalyzer(alpha=alpha)
        self.checkpoint_analyzer = CheckpointAnalyzer(checkpoints=checkpoints)
        self.latex_generator = LatexTableGenerator(decimal_places=3, significance_alpha=alpha)
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    def process_benchmark_results(
        self, 
        csv_path: str,
        warmstart_trials_path: Optional[str] = None,
        standard_trials_path: Optional[str] = None,
        algorithm_name: str = "DecisionTree",
        include_convergence_analysis: bool = True,
        generate_latex_tables: bool = True
    ) -> Dict[str, Any]:
        """
        Process benchmark results and generate publication analysis.
        
        Args:
            csv_path: Path to benchmark results CSV file
            warmstart_trials_path: Optional path to warmstart trial data CSV
            standard_trials_path: Optional path to standard trial data CSV  
            algorithm_name: Name of the algorithm for output files (default: "DecisionTree")
            include_convergence_analysis: Whether to perform convergence analysis (default: True)
            generate_latex_tables: Whether to generate LaTeX tables (default: True)
            
        Returns:
            Dictionary containing all analysis results
        """
        # Create timestamped subdirectory for this analysis run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_dir = os.path.join(self.output_dir, f"{algorithm_name}_{timestamp}")
        os.makedirs(analysis_dir, exist_ok=True)
        
        print("🚀 Starting Publication Analysis Pipeline")
        print("=" * 60)
        print(f"Algorithm: {algorithm_name}")
        print(f"Benchmark CSV: {csv_path}")
        print(f"Analysis Directory: {analysis_dir}")
        
        if warmstart_trials_path:
            print(f"Warmstart Trials: {warmstart_trials_path}")
        if standard_trials_path:
            print(f"Standard Trials: {standard_trials_path}")
        
        if include_convergence_analysis:
            print(f"Convergence Checkpoints: {self.checkpoints}")
        
        if generate_latex_tables:
            print("📝 LaTeX table generation: Enabled")
        
        print("-" * 60)
        
        # Load benchmark results
        print("📊 Loading benchmark results...")
        df_results = self.load_benchmark_csv(csv_path)
        
        if df_results is None or len(df_results) == 0:
            print("❌ No benchmark data loaded. Cannot proceed with analysis.")
            return {}
        
        print(f"✅ Loaded {len(df_results)} benchmark results")
        print(f"Available columns: {list(df_results.columns)}")
        print()
        
        # Perform statistical analysis
        print("🔬 Performing statistical analysis...")
        comparison_results = self.stats_analyzer.perform_publication_comparisons(df_results)
        
        if not comparison_results:
            print("❌ No statistical comparisons could be performed.")
            return {}
        
        # Generate statistical summary
        print("📈 Generating statistical summary...")
        summary_stats = self.stats_analyzer.generate_statistical_summary(comparison_results)
        
        # Perform convergence analysis if trial data is available
        convergence_results = {}
        if include_convergence_analysis and warmstart_trials_path and standard_trials_path:
            print("\n🔍 Performing convergence analysis...")
            convergence_results = self.checkpoint_analyzer.analyze_convergence(
                warmstart_trials_path, 
                standard_trials_path,
                algorithm_name
            )
            
            if convergence_results:
                self.checkpoint_analyzer.print_convergence_summary(convergence_results)
        
        # 4. Generate LaTeX tables if requested
        latex_files = {}
        if generate_latex_tables:
            print("\n📝 Generating LaTeX tables...")
            latex_files = self.latex_generator.generate_all_tables(
                comparison_results,
                convergence_results,
                algorithm_name,
                analysis_dir
            )
        
        # Save results to files
        output_files = {}
        
        # Save detailed comparison results
        for comparison_name, results_df in comparison_results.items():
            if len(results_df) > 0:
                filename = f"{algorithm_name}_{comparison_name}_detailed_{timestamp}.csv"
                filepath = os.path.join(analysis_dir, filename)
                results_df.to_csv(filepath, index=False)
                print(f"💾 Saved detailed results: {filepath}")
                output_files[f'detailed_{comparison_name}'] = filename
        
        # Save statistical summary
        if len(summary_stats) > 0:
            summary_filename = f"{algorithm_name}_statistical_summary_{timestamp}.csv"
            summary_filepath = os.path.join(analysis_dir, summary_filename)
            summary_stats.to_csv(summary_filepath, index=False)
            print(f"💾 Saved statistical summary: {summary_filepath}")
            output_files['statistical_summary'] = summary_filename
        
        # Save convergence analysis results
        if convergence_results and 'convergence_tables' in convergence_results:
            for table_name, table_df in convergence_results['convergence_tables'].items():
                if len(table_df) > 0:
                    filename = f"{algorithm_name}_convergence_{table_name}_{timestamp}.csv"
                    filepath = os.path.join(analysis_dir, filename)
                    table_df.to_csv(filepath, index=False)
                    print(f"💾 Saved convergence table: {filepath}")
                    output_files[f'convergence_{table_name}'] = filename
        
        # Add LaTeX files to output files
        for table_type, latex_path in latex_files.items():
            output_files[f'latex_{table_type}'] = os.path.basename(latex_path)
        
        # Prepare return data
        results = {
            'benchmark_data': df_results,
            'comparison_results': comparison_results,
            'statistical_summary': summary_stats,
            'convergence_results': convergence_results,
            'latex_files': latex_files,
            'algorithm_name': algorithm_name,
            'timestamp': timestamp,
            'output_files': output_files
        }
        
        # Print final summary
        self._print_analysis_summary(comparison_results, summary_stats, convergence_results, latex_files, algorithm_name, analysis_dir)
        
        return results
    
    def load_benchmark_csv(self, csv_path: str) -> Optional[pd.DataFrame]:
        """
        Load benchmark results from CSV file.
        
        Args:
            csv_path: Path to the benchmark CSV file
            
        Returns:
            DataFrame with benchmark results or None if loading fails
        """
        try:
            if not os.path.exists(csv_path):
                print(f"❌ Benchmark CSV file not found: {csv_path}")
                return None
            
            df = pd.read_csv(csv_path)
            
            # Basic validation
            if len(df) == 0:
                print(f"❌ Benchmark CSV file is empty: {csv_path}")
                return None
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading benchmark CSV {csv_path}: {str(e)}")
            return None
    
    def load_trial_data(self, warmstart_path: str, standard_path: str) -> Optional[pd.DataFrame]:
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
                dfs.append(df_warmstart)
                print(f"✅ Loaded {len(df_warmstart)} warmstart trials")
            else:
                print(f"⚠️  Warmstart trials file not found: {warmstart_path}")
            
            if os.path.exists(standard_path):
                df_standard = pd.read_csv(standard_path)
                dfs.append(df_standard)
                print(f"✅ Loaded {len(df_standard)} standard trials")
            else:
                print(f"⚠️  Standard trials file not found: {standard_path}")
            
            if not dfs:
                print("❌ No trial data files found")
                return None
            
            # Combine all trial data
            combined_df = pd.concat(dfs, ignore_index=True)
            print(f"✅ Combined trial data: {len(combined_df)} total trials")
            
            return combined_df
            
        except Exception as e:
            print(f"❌ Error loading trial data: {str(e)}")
            return None
    
    def _print_analysis_summary(
        self, 
        comparison_results: Dict[str, pd.DataFrame], 
        summary_stats: pd.DataFrame,
        convergence_results: Dict,
        latex_files: Dict[str, str],
        algorithm_name: str,
        analysis_dir: str
    ):
        """Print final analysis summary."""
        print("\n" + "=" * 60)
        print(f"📊 PUBLICATION ANALYSIS SUMMARY - {algorithm_name}")
        print("=" * 60)
        
        if len(summary_stats) == 0:
            print("❌ No statistical analysis results available")
            return
        
        print(f"🔬 Statistical Comparisons Performed: {len(comparison_results)}")
        
        # Statistical summary
        for _, row in summary_stats.iterrows():
            comparison = row['comparison'].replace('_', ' ').title()
            method1 = row['method1']
            method2 = row['method2']
            
            print(f"\n📈 {comparison}:")
            print(f"   {method1} vs {method2}")
            print(f"   Datasets analyzed: {int(row['total_datasets'])}")
            print(f"   {method1} wins: {int(row['method1_wins'])}/{int(row['total_datasets'])} ({row['win_rate_pct']:.1f}%)")
            print(f"   Significant differences: {int(row['significant_differences'])} ({row['significance_rate_pct']:.1f}%)")
            print(f"   Average uplift: {row['mean_uplift_pct']:.2f}% ± {row['std_uplift_pct']:.2f}%")
        
        # Convergence summary
        if convergence_results and 'convergence_tables' in convergence_results:
            print(f"\n🔍 Convergence Analysis:")
            tables = convergence_results['convergence_tables']
            print(f"   Generated {len(tables)} convergence tables")
            print(f"   Checkpoints analyzed: {self.checkpoints}")
            
            if 'method_comparison' in tables:
                method_table = tables['method_comparison']
                methods = [col.replace('_mean', '') for col in method_table.columns if col.endswith('_mean')]
                print(f"   Methods compared: {', '.join(methods)}")
        
        # LaTeX tables summary
        if latex_files:
            print(f"\n📝 LaTeX Tables Generated:")
            for table_type, filepath in latex_files.items():
                table_name = table_type.replace('_', ' ').title()
                print(f"   {table_name}: {os.path.basename(filepath)}")
        
        total_files = len(comparison_results) + len(convergence_results.get('convergence_tables', {})) + len(latex_files)
        print(f"\n✅ Analysis complete! Generated {total_files} output files.")
        print("🎯 LaTeX tables are ready for publication use!")
        print(f"📁 All files saved in: {analysis_dir}")
    
    def get_available_comparisons(self, df_results: pd.DataFrame) -> Dict[str, bool]:
        """
        Check which statistical comparisons are possible with the available data.
        
        Args:
            df_results: DataFrame with benchmark results
            
        Returns:
            Dictionary indicating which comparisons are available
        """
        columns = df_results.columns.tolist()
        
        return {
            'zeroshot_vs_random': 'auc_predicted' in columns and 'auc_random' in columns,
            'zeroshot_vs_standard_optuna': 'auc_predicted' in columns and 'auc_optuna_standard' in columns,
            'warmstart_vs_standard_optuna': 'auc_optuna_warmstart' in columns and 'auc_optuna_standard' in columns
        } 