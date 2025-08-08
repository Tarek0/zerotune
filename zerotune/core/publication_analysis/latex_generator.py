"""
LaTeX Table Generator for Publication Results

This module generates publication-ready LaTeX tables from statistical analysis results,
with proper formatting including bold for best scores and underlines for significant differences.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import os


class LatexTableGenerator:
    """
    Generator for publication-ready LaTeX tables.
    
    Creates formatted LaTeX tables with:
    - Bold formatting for best scores
    - Underline formatting for statistically significant differences
    - Proper decimal precision and alignment
    - Professional table styling
    """
    
    def __init__(self, decimal_places: int = 3, significance_alpha: float = 0.05):
        """
        Initialize the LaTeX table generator.
        
        Args:
            decimal_places: Number of decimal places for scores (default: 3)
            significance_alpha: Significance threshold for underlining (default: 0.05)
        """
        self.decimal_places = decimal_places
        self.significance_alpha = significance_alpha
    
    def generate_all_tables(
        self, 
        comparison_results: Dict[str, pd.DataFrame],
        convergence_results: Optional[Dict] = None,
        algorithm_name: str = "DecisionTree",
        output_dir: str = "publication_outputs"
    ) -> Dict[str, str]:
        """
        Generate all LaTeX tables from analysis results.
        
        Args:
            comparison_results: Dictionary of statistical comparison results
            convergence_results: Optional convergence analysis results
            algorithm_name: Name of the algorithm for file naming
            output_dir: Output directory for LaTeX files
            
        Returns:
            Dictionary mapping table types to generated file paths
        """
        print("📝 Generating LaTeX tables...")
        print("-" * 40)
        
        generated_files = {}
        
        # 1. Separate comparison tables (Zero-shot vs Random, Zero-shot vs Optuna, Warm-start vs Optuna)
        if comparison_results:
            separate_tables = self.generate_separate_comparison_tables(
                comparison_results, algorithm_name, output_dir
            )
            generated_files.update(separate_tables)
        
        # 2. Convergence analysis table
        if convergence_results and 'convergence_tables' in convergence_results:
            convergence_table_path = self.generate_convergence_table(
                convergence_results['convergence_tables'], algorithm_name, output_dir
            )
            if convergence_table_path:
                generated_files['convergence_analysis'] = convergence_table_path
        
        # 3. Statistical summary table
        if comparison_results:
            stats_table_path = self.generate_statistical_summary_table(
                comparison_results, algorithm_name, output_dir
            )
            if stats_table_path:
                generated_files['statistical_summary'] = stats_table_path
        
        print(f"✅ Generated {len(generated_files)} LaTeX tables")
        return generated_files
    
    def generate_main_comparison_table(
        self, 
        comparison_results: Dict[str, pd.DataFrame],
        algorithm_name: str,
        output_dir: str
    ) -> Optional[str]:
        """
        Generate main comparison table showing all methods across datasets.
        
        Args:
            comparison_results: Statistical comparison results
            algorithm_name: Algorithm name for file naming
            output_dir: Output directory
            
        Returns:
            Path to generated LaTeX file or None if generation fails
        """
        print("   Creating main comparison table...")
        
        try:
            # Combine all comparison data
            all_data = []
            
            # Get dataset information from any available comparison
            dataset_info = {}
            for comp_name, results_df in comparison_results.items():
                if len(results_df) > 0:
                    for _, row in results_df.iterrows():
                        dataset_id = row.get('dataset_id', 'Unknown')
                        dataset_name = row.get('dataset_name', f'Dataset_{dataset_id}')
                        dataset_info[dataset_id] = dataset_name
            
            # Extract performance data for each method
            performance_data = {}
            significance_data = {}
            
            for dataset_id, dataset_name in dataset_info.items():
                row_data = {
                    'dataset_id': dataset_id,
                    'dataset_name': dataset_name
                }
                
                # Extract scores from each comparison
                for comp_name, results_df in comparison_results.items():
                    dataset_rows = results_df[results_df['dataset_id'] == dataset_id]
                    
                    if len(dataset_rows) > 0:
                        row = dataset_rows.iloc[0]
                        
                        # Extract method scores based on comparison type
                        if comp_name == 'zeroshot_vs_random':
                            row_data['Zero-shot'] = row.get('Zero-shot_mean', np.nan)
                            row_data['Random'] = row.get('Random_mean', np.nan)
                            significance_data[(dataset_id, 'Zero-shot', 'Random')] = row.get('significant', False)
                            
                        elif comp_name == 'zeroshot_vs_standard_optuna':
                            row_data['Zero-shot'] = row.get('Zero-shot_mean', np.nan)
                            row_data['Standard Optuna TPE'] = row.get('Standard Optuna TPE_mean', np.nan)
                            significance_data[(dataset_id, 'Zero-shot', 'Standard Optuna TPE')] = row.get('significant', False)
                            
                        elif comp_name == 'warmstart_vs_standard_optuna':
                            row_data['Warm-started Optuna TPE'] = row.get('Warm-started Optuna TPE_mean', np.nan)
                            row_data['Standard Optuna TPE'] = row.get('Standard Optuna TPE_mean', np.nan)
                            significance_data[(dataset_id, 'Warm-started Optuna TPE', 'Standard Optuna TPE')] = row.get('significant', False)
                
                all_data.append(row_data)
            
            if not all_data:
                print("   ⚠️  No data available for main comparison table")
                return None
            
            # Create DataFrame
            df = pd.DataFrame(all_data)
            
            # Identify available methods
            method_columns = [col for col in df.columns if col not in ['dataset_id', 'dataset_name']]
            
            # Generate LaTeX table
            latex_content = self._create_main_comparison_latex(df, method_columns, significance_data, algorithm_name)
            
            # Save to file
            filename = f"{algorithm_name}_main_comparison_table.tex"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w') as f:
                f.write(latex_content)
            
            print(f"   💾 Saved main comparison table: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"   ❌ Error generating main comparison table: {str(e)}")
            return None
    
    def generate_convergence_table(
        self, 
        convergence_tables: Dict[str, pd.DataFrame],
        algorithm_name: str,
        output_dir: str
    ) -> Optional[str]:
        """
        Generate convergence analysis table showing method performance at checkpoints.
        
        Args:
            convergence_tables: Convergence analysis tables
            algorithm_name: Algorithm name for file naming
            output_dir: Output directory
            
        Returns:
            Path to generated LaTeX file or None if generation fails
        """
        print("   Creating convergence analysis table...")
        
        try:
            if 'method_comparison' not in convergence_tables:
                print("   ⚠️  Method comparison table not available for convergence analysis")
                return None
            
            method_table = convergence_tables['method_comparison']
            
            if len(method_table) == 0:
                print("   ⚠️  Empty method comparison table")
                return None
            
            # Generate LaTeX table
            latex_content = self._create_convergence_latex(method_table, algorithm_name)
            
            # Save to file
            filename = f"{algorithm_name}_convergence_analysis_table.tex"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w') as f:
                f.write(latex_content)
            
            print(f"   💾 Saved convergence analysis table: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"   ❌ Error generating convergence table: {str(e)}")
            return None
    
    def generate_statistical_summary_table(
        self, 
        comparison_results: Dict[str, pd.DataFrame],
        algorithm_name: str,
        output_dir: str
    ) -> Optional[str]:
        """
        Generate statistical summary table with win rates and significance.
        
        Args:
            comparison_results: Statistical comparison results
            algorithm_name: Algorithm name for file naming
            output_dir: Output directory
            
        Returns:
            Path to generated LaTeX file or None if generation fails
        """
        print("   Creating statistical summary table...")
        
        try:
            # Calculate summary statistics for each comparison
            summary_data = []
            
            for comp_name, results_df in comparison_results.items():
                if len(results_df) == 0:
                    continue
                
                # Extract method names
                if comp_name == 'zeroshot_vs_random':
                    method1, method2 = 'Zero-shot', 'Random'
                elif comp_name == 'zeroshot_vs_standard_optuna':
                    method1, method2 = 'Zero-shot', 'Standard Optuna TPE'
                elif comp_name == 'warmstart_vs_standard_optuna':
                    method1, method2 = 'Warm-started Optuna TPE', 'Standard Optuna TPE'
                else:
                    continue
                
                # Calculate statistics
                total_datasets = len(results_df)
                method1_wins = len(results_df[results_df['best_method'] == method1])
                significant_count = len(results_df[results_df['significant'] == True])
                mean_uplift = results_df['uplift_pct'].mean()
                
                summary_data.append({
                    'comparison': f"{method1} vs {method2}",
                    'method1': method1,
                    'method2': method2,
                    'total_datasets': total_datasets,
                    'method1_wins': method1_wins,
                    'win_rate_pct': (method1_wins / total_datasets) * 100,
                    'significant_differences': significant_count,
                    'significance_rate_pct': (significant_count / total_datasets) * 100,
                    'mean_uplift_pct': mean_uplift
                })
            
            if not summary_data:
                print("   ⚠️  No summary data available")
                return None
            
            summary_df = pd.DataFrame(summary_data)
            
            # Generate LaTeX table
            latex_content = self._create_statistical_summary_latex(summary_df, algorithm_name)
            
            # Save to file
            filename = f"{algorithm_name}_statistical_summary_table.tex"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w') as f:
                f.write(latex_content)
            
            print(f"   💾 Saved statistical summary table: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"   ❌ Error generating statistical summary table: {str(e)}")
            return None
    
    def generate_separate_comparison_tables(
        self, 
        comparison_results: Dict[str, pd.DataFrame],
        algorithm_name: str,
        output_dir: str
    ) -> Dict[str, str]:
        """
        Generate separate comparison tables for each method comparison.
        
        Creates three separate LaTeX tables:
        1. Zero-shot vs Random
        2. Zero-shot vs Optuna TPE (Standard)
        3. Optuna Warm-start vs Optuna TPE (Standard)
        
        Args:
            comparison_results: Statistical comparison results
            algorithm_name: Algorithm name for file naming
            output_dir: Output directory
            
        Returns:
            Dictionary mapping comparison names to generated LaTeX file paths
        """
        print("   Creating separate comparison tables...")
        
        generated_files = {}
        
        # Define the three comparisons we want to generate
        table_configs = [
            {
                'comparison_key': 'zeroshot_vs_random',
                'title': f'{algorithm_name}: Zero-shot vs Random',
                'label': f'tab:{algorithm_name.lower()}_zeroshot_vs_random',
                'filename': f'{algorithm_name}_zeroshot_vs_random_table.tex',
                'method1_name': 'Zero-shot',
                'method2_name': 'Random',
                'method1_col': 'Zero-shot_mean',
                'method2_col': 'Random_mean'
            },
            {
                'comparison_key': 'zeroshot_vs_standard_optuna',
                'title': f'{algorithm_name}: Zero-shot vs Optuna TPE',
                'label': f'tab:{algorithm_name.lower()}_zeroshot_vs_optuna',
                'filename': f'{algorithm_name}_zeroshot_vs_optuna_table.tex',
                'method1_name': 'Zero-shot',
                'method2_name': 'Optuna TPE',
                'method1_col': 'Zero-shot_mean',
                'method2_col': 'Standard Optuna TPE_mean'
            },
            {
                'comparison_key': 'warmstart_vs_standard_optuna',
                'title': f'{algorithm_name}: Optuna Warm-start vs Optuna TPE',
                'label': f'tab:{algorithm_name.lower()}_warmstart_vs_optuna',
                'filename': f'{algorithm_name}_warmstart_vs_optuna_table.tex',
                'method1_name': 'Optuna Warm-start',
                'method2_name': 'Optuna TPE',
                'method1_col': 'Warm-started Optuna TPE_mean',
                'method2_col': 'Standard Optuna TPE_mean'
            }
        ]
        
        for config in table_configs:
            comparison_key = config['comparison_key']
            
            if comparison_key not in comparison_results:
                print(f"   ⚠️  Skipping {comparison_key} - no data available")
                continue
                
            results_df = comparison_results[comparison_key]
            if len(results_df) == 0:
                print(f"   ⚠️  Skipping {comparison_key} - empty results")
                continue
            
            try:
                latex_content = self._create_individual_comparison_latex(
                    results_df, config
                )
                
                # Save to file
                output_path = os.path.join(output_dir, config['filename'])
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(latex_content)
                
                generated_files[comparison_key] = output_path
                print(f"   💾 Saved {comparison_key} table: {output_path}")
                
            except Exception as e:
                print(f"   ❌ Failed to generate {comparison_key} table: {e}")
                continue
        
        return generated_files

    def _create_main_comparison_latex(
        self, 
        df: pd.DataFrame, 
        method_columns: List[str], 
        significance_data: Dict,
        algorithm_name: str
    ) -> str:
        """Create LaTeX content for main comparison table."""
        
        # Table header
        num_cols = len(method_columns) + 1  # +1 for dataset name
        col_spec = "l" + "c" * len(method_columns)
        
        latex_lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{algorithm_name} Performance Comparison Across Methods}}",
            f"\\label{{tab:{algorithm_name.lower()}_comparison}}",
            f"\\begin{{tabular}}{{{col_spec}}}",
            "\\toprule"
        ]
        
        # Header row
        header = "Dataset & " + " & ".join(method_columns) + " \\\\"
        latex_lines.append(header)
        latex_lines.append("\\midrule")
        
        # Data rows
        for _, row in df.iterrows():
            dataset_name = str(row['dataset_name']).replace('_', '\\_')
            
            # Get scores for each method
            method_scores = []
            for method in method_columns:
                score = row.get(method, np.nan)
                if pd.isna(score):
                    method_scores.append("--")
                else:
                    method_scores.append(score)
            
            # Find best score for bolding
            valid_scores = [s for s in method_scores if s != "--"]
            best_score = max(valid_scores) if valid_scores else None
            
            # Format each method score
            formatted_scores = []
            for i, (method, score) in enumerate(zip(method_columns, method_scores)):
                if score == "--":
                    formatted_scores.append("--")
                else:
                    # Format score
                    score_str = f"{score:.{self.decimal_places}f}"
                    
                    # Bold if best score
                    if best_score is not None and abs(score - best_score) < 1e-6:
                        score_str = f"\\textbf{{{score_str}}}"
                    
                    # Underline if significant (check against other methods)
                    dataset_id = row['dataset_id']
                    for other_method in method_columns:
                        if other_method != method:
                            sig_key = (dataset_id, method, other_method)
                            if significance_data.get(sig_key, False):
                                score_str = f"\\underline{{{score_str}}}"
                                break
                    
                    formatted_scores.append(score_str)
            
            # Create row
            data_row = f"{dataset_name} & " + " & ".join(formatted_scores) + " \\\\"
            latex_lines.append(data_row)
        
        # Table footer
        latex_lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\begin{tablenotes}",
            "\\small",
            "\\item \\textbf{Bold}: Best performance for the dataset.",
            f"\\item \\underline{{Underlined}}: Statistically significant difference (p < {self.significance_alpha}).",
            "\\item Higher scores indicate better performance (AUC).",
            "\\end{tablenotes}",
            "\\end{table}"
        ])
        
        return "\n".join(latex_lines)
    
    def _create_convergence_latex(self, method_table: pd.DataFrame, algorithm_name: str) -> str:
        """Create LaTeX content for convergence analysis table."""
        
        # Get method names
        method_columns = [col.replace('_mean', '') for col in method_table.columns if col.endswith('_mean')]
        
        # Table header
        col_spec = "c" + "c" * len(method_columns)
        
        latex_lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{algorithm_name} Convergence Analysis: Performance at Trial Checkpoints}}",
            f"\\label{{tab:{algorithm_name.lower()}_convergence}}",
            f"\\begin{{tabular}}{{{col_spec}}}",
            "\\toprule"
        ]
        
        # Header row
        header = "Trials & " + " & ".join(method_columns) + " \\\\"
        latex_lines.append(header)
        latex_lines.append("\\midrule")
        
        # Data rows
        for _, row in method_table.iterrows():
            checkpoint = int(row['checkpoint'])
            
            # Get scores for each method
            method_scores = []
            for method in method_columns:
                mean_col = f'{method}_mean'
                std_col = f'{method}_std'
                
                if mean_col in row and not pd.isna(row[mean_col]):
                    mean_score = row[mean_col]
                    std_score = row[std_col] if std_col in row and not pd.isna(row[std_col]) else 0
                    method_scores.append((mean_score, std_score))
                else:
                    method_scores.append(None)
            
            # Find best score for bolding
            valid_means = [s[0] for s in method_scores if s is not None]
            best_mean = max(valid_means) if valid_means else None
            
            # Format each method score
            formatted_scores = []
            for i, score_data in enumerate(method_scores):
                if score_data is None:
                    formatted_scores.append("--")
                else:
                    mean_score, std_score = score_data
                    score_str = f"{mean_score:.{self.decimal_places}f} ± {std_score:.{self.decimal_places}f}"
                    
                    # Bold if best score
                    if best_mean is not None and abs(mean_score - best_mean) < 1e-6:
                        score_str = f"\\textbf{{{score_str}}}"
                    
                    formatted_scores.append(score_str)
            
            # Create row
            data_row = f"{checkpoint} & " + " & ".join(formatted_scores) + " \\\\"
            latex_lines.append(data_row)
        
        # Table footer
        latex_lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\begin{tablenotes}",
            "\\small",
            "\\item \\textbf{Bold}: Best performance at the given checkpoint.",
            "\\item Values shown as mean ± standard deviation across all datasets and seeds.",
            "\\item Higher scores indicate better performance (AUC).",
            "\\end{tablenotes}",
            "\\end{table}"
        ])
        
        return "\n".join(latex_lines)
    
    def _create_statistical_summary_latex(self, summary_df: pd.DataFrame, algorithm_name: str) -> str:
        """Create LaTeX content for statistical summary table."""
        
        latex_lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{algorithm_name} Statistical Summary: Method Comparisons}}",
            f"\\label{{tab:{algorithm_name.lower()}_stats_summary}}",
            "\\begin{tabular}{lccccc}",
            "\\toprule",
            "Comparison & Datasets & Win Rate & Significant & Avg. Uplift \\\\"
        ]
        
        latex_lines.append("\\midrule")
        
        # Data rows
        for _, row in summary_df.iterrows():
            comparison = str(row['comparison']).replace('_', '\\_')
            total_datasets = int(row['total_datasets'])
            method1_wins = int(row['method1_wins'])
            win_rate = row['win_rate_pct']
            significant_count = int(row['significant_differences'])
            significance_rate = row['significance_rate_pct']
            mean_uplift = row['mean_uplift_pct']
            
            # Format win rate
            win_rate_str = f"{method1_wins}/{total_datasets} ({win_rate:.1f}\\%)"
            
            # Bold win rate if > 50%
            if win_rate > 50:
                win_rate_str = f"\\textbf{{{win_rate_str}}}"
            
            # Format significance
            sig_str = f"{significant_count} ({significance_rate:.1f}\\%)"
            
            # Format uplift
            uplift_str = f"{mean_uplift:+.2f}\\%"
            if mean_uplift > 0:
                uplift_str = f"\\textbf{{{uplift_str}}}"
            
            data_row = f"{comparison} & {total_datasets} & {win_rate_str} & {sig_str} & {uplift_str} \\\\"
            latex_lines.append(data_row)
        
        # Table footer
        latex_lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\begin{tablenotes}",
            "\\small",
            "\\item \\textbf{Bold}: Favorable results (win rate > 50\\% or positive uplift).",
            f"\\item Significant: p-value < {self.significance_alpha} (paired t-test).",
            "\\item Uplift: Percentage improvement of first method over second method.",
            "\\end{tablenotes}",
            "\\end{table}"
        ])
        
        return "\n".join(latex_lines) 

    def _create_individual_comparison_latex(
        self, 
        results_df: pd.DataFrame, 
        config: Dict[str, str]
    ) -> str:
        """
        Create LaTeX content for an individual comparison table.
        
        Args:
            results_df: DataFrame with comparison results
            config: Configuration dictionary with table settings
            
        Returns:
            LaTeX table content as string
        """
        lines = []
        
        # Table header
        lines.extend([
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{config['title']}}}",
            f"\\label{{{config['label']}}}",
            "\\begin{tabular}{lcccc}",
            "\\toprule",
            f"Dataset & {config['method1_name']} & {config['method2_name']} & Uplift & P-value \\\\",
            "\\midrule"
        ])
        
        # Sort by dataset name for consistency
        sorted_df = results_df.sort_values('dataset_name')
        
        # Table rows
        for _, row in sorted_df.iterrows():
            dataset_name = str(row['dataset_name']).replace('_', '\\_')
            method1_score = row[config['method1_col']]
            method2_score = row[config['method2_col']]
            uplift = row.get('uplift_pct', 0.0)
            p_value = row.get('p_value', 1.0)
            is_significant = row.get('significant', False)
            
            # Format scores
            method1_str = f"{method1_score:.{self.decimal_places}f}"
            method2_str = f"{method2_score:.{self.decimal_places}f}"
            
            # Determine which method is better and apply formatting
            if method1_score > method2_score:
                method1_str = f"\\textbf{{{method1_str}}}"
                if is_significant:
                    method1_str = f"\\underline{{{method1_str}}}"
            else:
                method2_str = f"\\textbf{{{method2_str}}}"
                if is_significant:
                    method2_str = f"\\underline{{{method2_str}}}"
            
            # Format uplift with sign
            uplift_str = f"{uplift:+.1f}\\%"
            if uplift > 0:
                uplift_str = f"\\textcolor{{green}}{{{uplift_str}}}"
            elif uplift < 0:
                uplift_str = f"\\textcolor{{red}}{{{uplift_str}}}"
            
            # Format p-value
            if p_value < 0.001:
                p_str = "< 0.001"
            else:
                p_str = f"{p_value:.3f}"
            
            if is_significant:
                p_str = f"\\textbf{{{p_str}}}"
            
            lines.append(f"{dataset_name} & {method1_str} & {method2_str} & {uplift_str} & {p_str} \\\\")
        
        # Table footer
        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\begin{tablenotes}",
            "\\small",
            "\\item \\textbf{Bold}: Better performing method for the dataset.",
            "\\item \\underline{Underlined}: Statistically significant difference (p < 0.05).",
            "\\item \\textcolor{green}{Green}/\\textcolor{red}{Red}: Positive/Negative uplift.",
            "\\item Higher AUC scores indicate better performance.",
            "\\end{tablenotes}",
            "\\end{table}"
        ])
        
        return '\n'.join(lines) 