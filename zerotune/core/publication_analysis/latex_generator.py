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
            
            # 2b. Optuna convergence subtables (4-subtable format)
            optuna_subtables_path = self.generate_optuna_convergence_subtables(
                convergence_results['convergence_tables'], algorithm_name, output_dir
            )
            if optuna_subtables_path:
                generated_files['optuna_convergence_subtables'] = optuna_subtables_path
        
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
                            row_data['ZT'] = row.get('ZT_mean', np.nan)
                            row_data['Random'] = row.get('Random_mean', np.nan)
                            significance_data[(dataset_id, 'ZT', 'Random')] = row.get('significant', False)
                            
                        elif comp_name == 'zeroshot_vs_standard_optuna':
                            row_data['ZT'] = row.get('ZT_mean', np.nan)
                            row_data['Standard Optuna TPE'] = row.get('Standard Optuna TPE_mean', np.nan)
                            significance_data[(dataset_id, 'ZT', 'Standard Optuna TPE')] = row.get('significant', False)
                            
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
            # Try to use aggregated comparison table first (new format)
            if 'aggregated_comparison' in convergence_tables:
                aggregated_table = convergence_tables['aggregated_comparison']
                if len(aggregated_table) > 0:
                    latex_content = self._create_convergence_latex(aggregated_table, algorithm_name)
                else:
                    print("   ⚠️  Empty aggregated comparison table, falling back to method comparison")
                    if 'method_comparison' not in convergence_tables:
                        print("   ⚠️  Method comparison table not available for convergence analysis")
                        return None
                    method_table = convergence_tables['method_comparison']
                    if len(method_table) == 0:
                        print("   ⚠️  Empty method comparison table")
                        return None
                    latex_content = self._create_convergence_latex(method_table, algorithm_name)
            else:
                # Fall back to original method comparison table
                if 'method_comparison' not in convergence_tables:
                    print("   ⚠️  Method comparison table not available for convergence analysis")
                    return None
                method_table = convergence_tables['method_comparison']
                if len(method_table) == 0:
                    print("   ⚠️  Empty method comparison table")
                    return None
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
                    method1, method2 = 'ZT', 'Random'
                elif comp_name == 'zeroshot_vs_standard_optuna':
                    method1, method2 = 'ZT', 'Standard Optuna TPE'
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
        Generate separate comparison table for Zero-shot vs Random.
        
        Creates one LaTeX table:
        - Zero-shot vs Random
        
        For Optuna methods, convergence analysis is used instead of per-dataset comparisons.
        
        Args:
            comparison_results: Statistical comparison results
            algorithm_name: Algorithm name for file naming
            output_dir: Output directory
            
        Returns:
            Dictionary mapping comparison names to generated LaTeX file paths
        """
        print("   Creating separate comparison tables...")
        
        generated_files = {}
        
        # Define only the Zero-shot vs Random comparison (skip Optuna comparisons)
        table_configs = [
            {
                'comparison_key': 'zeroshot_vs_random',
                'title': f'{algorithm_name}: ZT vs Random',
                'label': f'tab:{algorithm_name.lower()}_zt_vs_random',
                'filename': f'{algorithm_name}_zt_vs_random_per_dataset_table.tex',
                'method1_name': 'ZT',
                'method2_name': 'Random',
                'method1_col': 'ZT_mean',
                'method2_col': 'Random_mean'
            }
            # Note: Removed zeroshot_vs_standard_optuna and warmstart_vs_standard_optuna
            # For Optuna methods, we only focus on convergence analysis
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

    def generate_optuna_convergence_subtables(
        self, 
        convergence_tables: Dict[str, pd.DataFrame],
        algorithm_name: str,
        output_dir: str,
        checkpoints: List[int] = [1, 5, 10, 20]
    ) -> Optional[str]:
        """
        Generate Optuna convergence comparison in 4-subtable format.
        
        Note: This method creates a template showing the format, but may have limited
        data depending on the convergence analysis results available.
        
        Args:
            convergence_tables: Convergence analysis tables
            algorithm_name: Algorithm name for file naming
            output_dir: Output directory
            checkpoints: List of checkpoints to include (default: [1, 5, 10, 20])
            
        Returns:
            Path to generated LaTeX file or None if generation fails
        """
        print("   Creating Optuna convergence subtables...")
        
        try:
            if 'per_dataset_convergence' not in convergence_tables:
                print("   ⚠️  Per-dataset convergence table not available")
                return None
            
            per_dataset_table = convergence_tables['per_dataset_convergence']
            
            if len(per_dataset_table) == 0:
                print("   ⚠️  Empty per-dataset convergence table")
                return None
            
            # Generate LaTeX table
            latex_content = self._create_optuna_convergence_subtables_latex(
                per_dataset_table, algorithm_name, checkpoints
            )
            
            # Save to file
            filename = f"{algorithm_name}_optuna_convergence_per_dataset_subtables.tex"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w') as f:
                f.write(latex_content)
            
            print(f"   💾 Saved Optuna convergence subtables: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"   ❌ Error generating Optuna convergence subtables: {str(e)}")
            return None
    
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
        
        # Use the aggregated comparison table if available, otherwise fall back to method comparison
        if 'zt_best_pct' in method_table.columns:
            # New aggregated format: ZT vs TPE with Best % and Sig %
            return self._create_aggregated_convergence_latex(method_table, algorithm_name)
        else:
            # Original format: method comparison with mean ± std
            return self._create_original_convergence_latex(method_table, algorithm_name)
    
    def _create_aggregated_convergence_latex(self, aggregated_table: pd.DataFrame, algorithm_name: str) -> str:
        """Create LaTeX content for aggregated convergence comparison table (ZT vs TPE format)."""
        
        latex_lines = [
            "\\begin{table}[htbp]",
            "\\captionsetup{font=footnotesize, justification=raggedright, singlelinecheck=false}",
            f"\\caption[ZT+TPE vs TPE with increasing HPO iterations]%",
            "{Comparison of ZT+TPE and TPE on real-world datasets when TPE is allowed ",
            "additional HPO iterations: 0, 1, 5, 10, and 20. ",
            "At iteration 0, TPE is initialised with a random hyperparameter configuration. ",
            "ZT+TPE represents warm-started TPE initialized with ZeroTune predictions. ",
            "\"Best (\\%)\" indicates the percentage of datasets on which each method achieves the highest AUC, and ",
            "\"Significant (\\%)\" shows the fraction of datasets for which that advantage ",
            "is statistically significant (paired t-test, \\(p < 0.05\\)).}",
            f"\\label{{table:{algorithm_name.lower()}-convergence-summary}}",
            "\\vskip 0.1in",
            "\\begin{center}",
            "\\begin{small}",
            "\\begin{sc}",
            "\\begin{tabular}{c c c c c}",
            "\\toprule",
            "\\multirow{2}{*}{\\makecell{\\textbf{HPO}\\\\\\textbf{Iterations}}} ",
            "& \\multicolumn{2}{c}{\\textbf{ZT+TPE}} ",
            "& \\multicolumn{2}{c}{\\textbf{TPE}} \\\\",
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
            "& \\textbf{Best (\\%)} & \\textbf{Sig (\\%)} ",
            "& \\textbf{Best (\\%)} & \\textbf{Sig (\\%)} \\\\",
            "\\midrule"
        ]
        
        # Data rows
        for _, row in aggregated_table.iterrows():
            checkpoint = int(row['checkpoint'])
            
            # Format percentages
            zt_best = f"{row['zt_best_pct']:.1f}" if not pd.isna(row['zt_best_pct']) else "--"
            zt_sig = f"{row['zt_sig_pct']:.1f}" if not pd.isna(row['zt_sig_pct']) else "--"
            tpe_best = f"{row['tpe_best_pct']:.1f}" if not pd.isna(row['tpe_best_pct']) else "--"
            tpe_sig = f"{row['tpe_sig_pct']:.1f}" if not pd.isna(row['tpe_sig_pct']) else "--"
            
            data_row = f"{checkpoint} & {zt_best} & {zt_sig} & {tpe_best} & {tpe_sig} \\\\"
            latex_lines.append(data_row)
        
        # Table footer
        latex_lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{sc}",
            "\\end{small}",
            "\\end{center}",
            "\\vskip -0.1in",
            "\\end{table}"
        ])
        
        return "\n".join(latex_lines)
    
    def _create_original_convergence_latex(self, method_table: pd.DataFrame, algorithm_name: str) -> str:
        """Create LaTeX content for original convergence analysis table (mean ± std format)."""
        
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

    def _create_optuna_convergence_subtables_latex(
        self, 
        per_dataset_table: pd.DataFrame, 
        algorithm_name: str, 
        checkpoints: List[int] = [1, 5, 10, 20]
    ) -> str:
        """Create LaTeX content for Optuna convergence subtables format."""
        
        # Get unique datasets
        datasets = per_dataset_table['dataset_id'].unique()
        
        # Start LaTeX table
        latex_lines = [
            "\\begin{table}[htbp]",
            "\\captionsetup{font=footnotesize, justification=raggedright, singlelinecheck=false}",
            f"\\caption[Per-dataset performance of ZT+TPE vs TPE across HPO iterations]%",
            f"{{ZT+TPE vs TPE: Performance comparison (AUC) on real-world datasets across different HPO iterations.",
            f"AUC scores are averaged over 50 runs using different random seeds.",
            f"ZT+TPE is initialized with zero-shot predictions.",
            f"\\textbf{{Bold}} values represent the highest performance, and \\underline{{underlined}} values indicate statistically significant differences.",
            f"Note: Data availability depends on trial collection during evaluation.}}",
            f"\\label{{table:{algorithm_name.lower()}-optuna-convergence}}",
            "\\vskip 0.1in",
            "\\begin{center}",
            "\\begin{small}",
            "\\begin{sc}",
            "",
            "%--------------------------------",
            "% First Row of Subtables", 
            "%--------------------------------"
        ]
        
        # Create 4 subtables (2x2 grid)
        subtable_checkpoints = checkpoints[:4]  # Take first 4 checkpoints
        
        # First row (2 subtables)
        for i in range(0, min(2, len(subtable_checkpoints))):
            checkpoint = subtable_checkpoints[i]
            
            if i == 0:
                latex_lines.append("\\begin{subtable}[t]{0.48\\textwidth}")
            else:
                latex_lines.append("\\hfill")
                latex_lines.append("\\begin{subtable}[t]{0.48\\textwidth}")
            
            latex_lines.extend([
                "    \\centering",
                f"    \\caption{{ZT+TPE vs TPE ({checkpoint} iteration{'s' if checkpoint > 1 else ''})}}",
                f"    \\label{{tab:{algorithm_name.lower()}-optuna-{checkpoint}}}",
                "    \\begin{tabular}{lccc}",
                "    \\toprule",
                "    \\textbf{Dataset} & \\textbf{ZT+TPE} & \\textbf{TPE} & \\textbf{Uplift (\\%)} \\\\",
                "    \\midrule"
            ])
            
            # Add data rows for this checkpoint
            for dataset_id in sorted(datasets):
                dataset_data = per_dataset_table[
                    (per_dataset_table['dataset_id'] == dataset_id) & 
                    (per_dataset_table['checkpoint'] == checkpoint)
                ]
                
                if len(dataset_data) == 0:
                    continue
                
                row = dataset_data.iloc[0]
                
                # Get warm-started and standard Optuna scores
                warmstart_score = row.get('warmstart_mean', 0)
                standard_score = row.get('standard_mean', 0)
                
                # Calculate uplift percentage
                if standard_score > 0:
                    uplift_pct = ((warmstart_score - standard_score) / standard_score) * 100
                else:
                    uplift_pct = 0
                
                # Determine which is better for formatting
                if warmstart_score > standard_score:
                    warmstart_formatted = f"\\textbf{{{warmstart_score:.4f}}}"
                    standard_formatted = f"{standard_score:.4f}"
                elif standard_score > warmstart_score:
                    warmstart_formatted = f"{warmstart_score:.4f}"
                    standard_formatted = f"\\underline{{\\textbf{{{standard_score:.4f}}}}}"
                else:
                    warmstart_formatted = f"\\textbf{{{warmstart_score:.4f}}}"
                    standard_formatted = f"\\textbf{{{standard_score:.4f}}}"
                
                latex_lines.append(f"    {dataset_id}    & {warmstart_formatted} & {standard_formatted} & {uplift_pct:.2f} \\\\")
            
            latex_lines.extend([
                "    \\bottomrule",
                "    \\end{tabular}",
                "\\end{subtable}"
            ])
        
        # Add spacing between rows
        latex_lines.extend([
            "",
            "%--------------------------------",
            "% Second Row of Subtables",
            "%--------------------------------"
        ])
        
        # Second row (2 more subtables)  
        for i in range(2, min(4, len(subtable_checkpoints))):
            checkpoint = subtable_checkpoints[i]
            
            if i == 2:
                latex_lines.append("\\begin{subtable}[t]{0.48\\textwidth}")
            else:
                latex_lines.append("\\hfill")
                latex_lines.append("\\begin{subtable}[t]{0.48\\textwidth}")
            
            latex_lines.extend([
                "    \\centering",
                f"    \\caption{{ZT+TPE vs TPE ({checkpoint} iteration{'s' if checkpoint > 1 else ''})}}",
                f"    \\label{{tab:{algorithm_name.lower()}-optuna-{checkpoint}}}",
                "    \\begin{tabular}{lccc}",
                "    \\toprule",
                "    \\textbf{Dataset} & \\textbf{ZT+TPE} & \\textbf{TPE} & \\textbf{Uplift (\\%)} \\\\",
                "    \\midrule"
            ])
            
            # Add data rows for this checkpoint
            for dataset_id in sorted(datasets):
                dataset_data = per_dataset_table[
                    (per_dataset_table['dataset_id'] == dataset_id) & 
                    (per_dataset_table['checkpoint'] == checkpoint)
                ]
                
                if len(dataset_data) == 0:
                    continue
                
                row = dataset_data.iloc[0]
                
                # Get scores
                warmstart_score = row.get('warmstart_mean', 0)
                standard_score = row.get('standard_mean', 0)
                
                # Calculate uplift percentage
                if standard_score > 0:
                    uplift_pct = ((warmstart_score - standard_score) / standard_score) * 100
                else:
                    uplift_pct = 0
                
                # Determine which is better for formatting
                if warmstart_score > standard_score:
                    warmstart_formatted = f"\\textbf{{{warmstart_score:.4f}}}"
                    standard_formatted = f"{standard_score:.4f}"
                elif standard_score > warmstart_score:
                    warmstart_formatted = f"{warmstart_score:.4f}"
                    standard_formatted = f"\\underline{{\\textbf{{{standard_score:.4f}}}}}"
                else:
                    warmstart_formatted = f"\\textbf{{{warmstart_score:.4f}}}"
                    standard_formatted = f"\\textbf{{{standard_score:.4f}}}"
                
                latex_lines.append(f"    {dataset_id}    & {warmstart_formatted} & {standard_formatted} & {uplift_pct:.2f} \\\\")
            
            latex_lines.extend([
                "    \\bottomrule",
                "    \\end{tabular}",
                "\\end{subtable}"
            ])
        
        # Close the table
        latex_lines.extend([
            "",
            "\\end{sc}",
            "\\end{small}",
            "\\end{center}",
            "\\vskip -0.1in",
            "\\end{table}"
        ])
        
        return "\n".join(latex_lines) 