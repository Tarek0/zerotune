#!/usr/bin/env python3
"""
Publication Analysis Runner

Main script for running publication analysis on ML algorithm benchmark results.
Generates statistical comparisons, convergence analysis, and LaTeX tables for thesis/publication use.

Usage:
    python publication_analysis.py DecisionTree --csv benchmarks/benchmark_results_dt_kb_v1_full_optuna_20250808_143536.csv --warmstart benchmarks/optuna_trials_warmstart_dt_kb_v1_*.csv --standard benchmarks/optuna_trials_standard_dt_kb_v1_*.csv

    python publication_analysis.py RandomForest --csv benchmarks/benchmark_results_rf_kb_v1_full_optuna_20250808_*.csv

    python publication_analysis.py XGBoost --csv benchmarks/benchmark_results_xgb_kb_v1_full_optuna_20250808_*.csv --no-convergence

Features:
- Statistical comparisons with paired t-tests
- Convergence analysis at checkpoints (1, 5, 10, 15, 20 trials)
- LaTeX table generation with proper formatting
- Publication-ready interactive charts (HTML format)
- Timestamped output directories for organization
- Algorithm-agnostic analysis pipeline
"""

import argparse
import sys
import os
import glob
from datetime import datetime
from zerotune.core.publication_analysis import PublicationResultsProcessor


def find_latest_file(pattern):
    """Find the most recent file matching the given pattern."""
    files = glob.glob(pattern)
    if not files:
        return None
    # Sort by modification time, newest first
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def validate_files(csv_path, warmstart_path, standard_path, require_trials=True):
    """Validate that required files exist."""
    if not os.path.exists(csv_path):
        print(f"❌ Error: Benchmark CSV file not found: {csv_path}")
        return False
    
    if require_trials:
        if warmstart_path and not os.path.exists(warmstart_path):
            print(f"❌ Error: Warmstart trials file not found: {warmstart_path}")
            return False
        
        if standard_path and not os.path.exists(standard_path):
            print(f"❌ Error: Standard trials file not found: {standard_path}")
            return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run publication analysis on ML algorithm benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full analysis with LaTeX tables and interactive charts (default)
  python publication_analysis.py DecisionTree \\
    --csv benchmarks/benchmark_results_dt_kb_v1_full_optuna_20250808_143536.csv \\
    --warmstart benchmarks/optuna_trials_warmstart_dt_kb_v1_full_optuna_20250808_142352.csv \\
    --standard benchmarks/optuna_trials_standard_dt_kb_v1_full_optuna_20250808_142352.csv

  # Auto-detect latest files (recommended)
  python publication_analysis.py DecisionTree --auto-detect

  # Generate charts only for specific datasets
  python publication_analysis.py DecisionTree --auto-detect --chart-datasets 917 1049 1111

  # Skip charts (LaTeX only)
  python publication_analysis.py RandomForest --auto-detect --no-charts

  # Skip convergence analysis (basic comparisons only)
  python publication_analysis.py RandomForest --auto-detect --no-convergence
        """
    )
    
    parser.add_argument(
        'algorithm',
        choices=['DecisionTree', 'RandomForest', 'XGBoost'],
        help='ML algorithm to analyze'
    )
    
    parser.add_argument(
        '--csv',
        required=False,
        help='Path to benchmark results CSV file (supports wildcards)'
    )
    
    parser.add_argument(
        '--warmstart',
        help='Path to warmstart Optuna trials CSV file (supports wildcards)'
    )
    
    parser.add_argument(
        '--standard',
        help='Path to standard Optuna trials CSV file (supports wildcards)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='publication_outputs',
        help='Output directory for results (default: publication_outputs)'
    )
    
    parser.add_argument(
        '--checkpoints',
        nargs='+',
        type=int,
        default=[1, 5, 10, 15, 20],
        help='Convergence analysis checkpoints (default: 1 5 10 15 20)'
    )
    
    parser.add_argument(
        '--no-convergence',
        action='store_true',
        help='Skip convergence analysis (useful when trial data is not available)'
    )
    
    parser.add_argument(
        '--no-latex',
        action='store_true',
        help='Skip LaTeX table generation'
    )
    
    parser.add_argument(
        '--no-charts',
        action='store_true',
        help='Skip publication chart generation'
    )
    
    parser.add_argument(
        '--chart-datasets',
        nargs='+',
        type=int,
        help='Generate charts only for specific dataset IDs (default: all datasets)'
    )
    
    parser.add_argument(
        '--auto-detect',
        action='store_true',
        help='Auto-detect the latest benchmark and trial files for the specified algorithm'
    )
    
    args = parser.parse_args()
    
    # Auto-detect files if requested
    if args.auto_detect:
        algorithm_prefix = {
            'DecisionTree': 'dt',
            'RandomForest': 'rf', 
            'XGBoost': 'xgb'
        }[args.algorithm]
        
        csv_pattern = f"benchmarks/benchmark_results_{algorithm_prefix}_kb_v1_full_optuna_*.csv"
        warmstart_pattern = f"benchmarks/optuna_trials_warmstart_{algorithm_prefix}_kb_v1_full_optuna_*.csv"
        standard_pattern = f"benchmarks/optuna_trials_standard_{algorithm_prefix}_kb_v1_full_optuna_*.csv"
        
        args.csv = find_latest_file(csv_pattern)
        args.warmstart = find_latest_file(warmstart_pattern)
        args.standard = find_latest_file(standard_pattern)
        
        if not args.csv:
            print(f"❌ Error: No benchmark CSV files found matching: {csv_pattern}")
            return 1
        
        print(f"🔍 Auto-detected files:")
        print(f"   📊 Benchmark CSV: {args.csv}")
        if args.warmstart:
            print(f"   🚀 Warmstart trials: {args.warmstart}")
        if args.standard:
            print(f"   📈 Standard trials: {args.standard}")
    
    # Handle wildcard patterns in file paths
    if args.csv and '*' in args.csv:
        csv_files = glob.glob(args.csv)
        if not csv_files:
            print(f"❌ Error: No files found matching pattern: {args.csv}")
            return 1
        args.csv = find_latest_file(args.csv)
    
    if args.warmstart and '*' in args.warmstart:
        args.warmstart = find_latest_file(args.warmstart)
    
    if args.standard and '*' in args.standard:
        args.standard = find_latest_file(args.standard)
    
    # Validate required arguments
    if not args.csv:
        print("❌ Error: --csv argument is required (or use --auto-detect)")
        return 1
    
    # Determine if convergence analysis should be performed
    include_convergence = not args.no_convergence and args.warmstart and args.standard
    
    # Validate files exist
    if not validate_files(args.csv, args.warmstart, args.standard, require_trials=include_convergence):
        return 1
    
    # Print configuration
    print("🚀 Publication Analysis Configuration")
    print("=" * 60)
    print(f"Algorithm: {args.algorithm}")
    print(f"Benchmark CSV: {args.csv}")
    if args.warmstart:
        print(f"Warmstart trials: {args.warmstart}")
    if args.standard:
        print(f"Standard trials: {args.standard}")
    print(f"Output directory: {args.output_dir}")
    print(f"Convergence analysis: {'✅ Enabled' if include_convergence else '❌ Disabled'}")
    if include_convergence:
        print(f"Checkpoints: {args.checkpoints}")
    print(f"LaTeX generation: {'✅ Enabled' if not args.no_latex else '❌ Disabled'}")
    print(f"Chart generation: {'✅ Enabled' if not args.no_charts else '❌ Disabled'}")
    if not args.no_charts and args.chart_datasets:
        print(f"Chart datasets filter: {args.chart_datasets}")
    print("-" * 60)
    
    try:
        # Initialize the publication analysis processor
        processor = PublicationResultsProcessor(
            output_dir=args.output_dir,
            checkpoints=args.checkpoints
        )
        
        # Run the analysis
        results = processor.process_benchmark_results(
            csv_path=args.csv,
            warmstart_trials_path=args.warmstart if include_convergence else None,
            standard_trials_path=args.standard if include_convergence else None,
            algorithm_name=args.algorithm,
            include_convergence_analysis=include_convergence,
            generate_latex_tables=not args.no_latex,
            generate_charts=not args.no_charts,
            chart_dataset_ids=args.chart_datasets
        )
        
        print("\n🎉 Publication analysis completed successfully!")
        print(f"📁 Results saved in timestamped directory within: {args.output_dir}")
        
        if results and 'output_files' in results:
            total_files = len(results['output_files'])
            chart_count = len([f for f in results['output_files'].keys() if f.startswith('chart_')])
            latex_count = len([f for f in results['output_files'].keys() if f.startswith('latex_')])
            
            print(f"📄 Generated {total_files} output files:")
            if latex_count > 0:
                print(f"   📝 {latex_count} LaTeX tables")
            if chart_count > 0:
                print(f"   📊 {chart_count} PDF charts")
            
            # Show the analysis directory for easy access
            if 'timestamp' in results:
                analysis_dir = os.path.join(args.output_dir, f"{args.algorithm}_{results['timestamp']}")
                print(f"   📂 Open charts: {analysis_dir}/plots/*.pdf")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 