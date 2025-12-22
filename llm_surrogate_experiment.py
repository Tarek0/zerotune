#!/usr/bin/env python3
"""
LLM Surrogate HPO Experiment Script.

Benchmarks the LLM Surrogate sampler against traditional HPO methods (TPE, Random)
on the ZeroTune test datasets.

Usage:
    # Run quick test on single dataset
    python llm_surrogate_experiment.py test
    
    # Run full benchmark
    python llm_surrogate_experiment.py benchmark --n_trials 20 --n_seeds 5
    
    # Run single dataset comparison
    python llm_surrogate_experiment.py single --dataset credit-g --n_trials 20
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

# Load environment variables from .env file
from dotenv import load_dotenv
# Try multiple locations for .env
load_dotenv()  # Current directory
load_dotenv(Path(__file__).parent / ".env")  # Project root
load_dotenv(Path(__file__).parent / "zerotune" / ".env")  # zerotune subdirectory

import numpy as np
import pandas as pd
import optuna
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Suppress Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from zerotune.core.feature_extraction import calculate_dataset_meta_parameters
from zerotune.core.optimization import optimize_hyperparameters
from zerotune.core.data_loading import fetch_open_ml_data, prepare_data


# Decision Tree parameter grid (same as used in original ZeroTune)
DECISION_TREE_PARAM_GRID = {
    'max_depth': (1, 20, 'int'),
    'min_samples_split': (2, 50, 'int'),
    'min_samples_leaf': (1, 25, 'int'),
    'criterion': ['gini', 'entropy'],
}

# Full evaluation dataset IDs (unseen during KB training - from original ZeroTune paper)
TEST_DATASET_IDS = [
    917,    # fri_c1_1000_25
    1049,   # pc4
    1111,   # KDDCup09_appetency
    1464,   # blood-transfusion-service-center
    1494,   # qsar-biodeg
    1510,   # wdbc
    1558,   # bank-marketing
    4534,   # PhishingWebsites
    23381,  # dresses-sales
    40536,  # SpeedDating
]

# Small test subset for quick validation
TEST_DATASET_IDS_SMALL = [917, 1464, 1510]

# Knowledge base path
KB_PATH = "knowledge_base/kb_decision_tree_dt_kb_v1_full.json"


def run_single_optimization(
    X: np.ndarray,
    y: np.ndarray,
    sampler_name: str,
    meta_features: dict,
    n_trials: int = 20,
    seed: int = 42,
    verbose: bool = False,
    n_zerotune_initial: int = 0,
    n_random_initial: int = 0,
) -> dict:
    """
    Run a single optimization with specified sampler.
    
    Args:
        n_zerotune_initial: For LLM sampler - trials using ZeroTune prediction (default: 0)
        n_random_initial: For LLM sampler - trials using random exploration (default: 0)
    
    Returns:
        Dictionary with results including best_score, best_params, 
        trial_history, and timing.
    """
    # Create sampler
    if sampler_name == 'random':
        sampler = optuna.samplers.RandomSampler(seed=seed)
    elif sampler_name == 'tpe':
        sampler = optuna.samplers.TPESampler(seed=seed)
    elif sampler_name == 'llm':
        from zerotune.core.llm_surrogate import LLMSurrogateSampler
        sampler = LLMSurrogateSampler(
            meta_features=meta_features,
            kb_path=KB_PATH,
            model_type='decision_tree',
            llm_model='gpt-5-mini-2025-08-07',
            seed=seed,
            n_zerotune_initial=n_zerotune_initial,
            n_random_initial=n_random_initial,
        )
    else:
        raise ValueError(f"Unknown sampler: {sampler_name}")
    
    # Run optimization
    start_time = time.time()
    
    best_params, best_score, all_results, df_trials = optimize_hyperparameters(
        model_class=DecisionTreeClassifier,
        param_grid=DECISION_TREE_PARAM_GRID,
        X_train=X,
        y_train=y,
        metric='roc_auc',
        n_iter=n_trials,
        random_state=seed,
        verbose=verbose,
        sampler=sampler,
        model_random_state=seed,
    )
    
    elapsed_time = time.time() - start_time
    
    # Extract trial scores
    trial_scores = df_trials['value'].tolist() if 'value' in df_trials else []
    
    # Build result
    result = {
        'sampler': sampler_name,
        'seed': seed,
        'best_score': best_score,
        'best_params': best_params,
        'n_trials': n_trials,
        'elapsed_time': elapsed_time,
        'trial_scores': trial_scores,
    }
    
    # Add reasoning history for LLM sampler
    if sampler_name == 'llm' and hasattr(sampler, 'reasoning_history'):
        result['reasoning_history'] = sampler.reasoning_history
    
    return result


def build_trial_dataframe(results: list) -> pd.DataFrame:
    """
    Build trial-level DataFrame compatible with PublicationChartGenerator.
    
    Expected columns: dataset_id, method, seed, number, value
    """
    rows = []
    for r in results:
        if 'trial_scores' not in r or not r['trial_scores']:
            continue
        
        for trial_num, score in enumerate(r['trial_scores']):
            rows.append({
                'dataset_id': r.get('dataset_id'),
                'dataset_name': r.get('dataset_name', ''),
                'method': r['sampler'],
                'seed': r['seed'],
                'number': trial_num,
                'value': score,
            })
    
    return pd.DataFrame(rows)


def run_dataset_benchmark(
    dataset_id: int,
    n_trials: int = 20,
    n_seeds: int = 5,
    samplers: list = None,
    verbose: bool = True,
    n_zerotune_initial: int = 0,
    n_random_initial: int = 0,
) -> pd.DataFrame:
    """
    Run benchmark on a single dataset across multiple seeds and samplers.
    
    Args:
        n_zerotune_initial: For LLM sampler - trials using ZeroTune prediction
        n_random_initial: For LLM sampler - trials using random exploration
    """
    if samplers is None:
        samplers = ['random', 'tpe', 'llm']
    
    # Load dataset
    print(f"\n{'='*60}")
    print(f"Loading dataset {dataset_id}...")
    df, target_name, dataset_name = fetch_open_ml_data(dataset_id)
    X, y = prepare_data(df, target_name)
    print(f"Dataset: {dataset_name} ({len(y)} samples, {X.shape[1]} features)")
    
    # Calculate meta-features
    meta_features = calculate_dataset_meta_parameters(X, y)
    print(f"Imbalance ratio: {meta_features.get('imbalance_ratio', 'N/A'):.2f}")
    
    results = []
    
    for sampler_name in samplers:
        print(f"\n--- Running {sampler_name.upper()} sampler ---")
        
        for seed in range(n_seeds):
            if verbose:
                print(f"  Seed {seed + 1}/{n_seeds}...", end=' ', flush=True)
            
            try:
                result = run_single_optimization(
                    X=X, y=y,
                    sampler_name=sampler_name,
                    meta_features=meta_features,
                    n_trials=n_trials,
                    seed=seed,
                    n_zerotune_initial=n_zerotune_initial,
                    n_random_initial=n_random_initial,
                    verbose=False,
                )
                result['dataset_id'] = dataset_id
                result['dataset_name'] = dataset_name
                results.append(result)
                
                if verbose:
                    print(f"score={result['best_score']:.4f}, time={result['elapsed_time']:.1f}s")
                    
            except Exception as e:
                print(f"ERROR: {e}")
                results.append({
                    'dataset_id': dataset_id,
                    'dataset_name': dataset_name,
                    'sampler': sampler_name,
                    'seed': seed,
                    'best_score': None,
                    'error': str(e),
                })
    
    return pd.DataFrame(results)


def analyze_results(df: pd.DataFrame) -> None:
    """Print analysis of benchmark results."""
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    # Group by sampler
    summary = df.groupby('sampler')['best_score'].agg(['mean', 'std', 'min', 'max'])
    print("\nScore by Sampler:")
    print(summary.round(4))
    
    # Compute win rates
    if len(df['sampler'].unique()) > 1:
        print("\nWin Rates (per dataset-seed):")
        
        # Pivot to compare
        pivot = df.pivot_table(
            values='best_score',
            index=['dataset_id', 'seed'],
            columns='sampler',
            aggfunc='first'
        )
        
        samplers = df['sampler'].unique()
        for s1 in samplers:
            for s2 in samplers:
                if s1 < s2 and s1 in pivot.columns and s2 in pivot.columns:
                    wins = (pivot[s1] > pivot[s2]).sum()
                    total = len(pivot)
                    print(f"  {s1} vs {s2}: {wins}/{total} ({100*wins/total:.1f}%)")
    
    # Timing
    print("\nAverage Time per Optimization (seconds):")
    timing = df.groupby('sampler')['elapsed_time'].mean()
    for sampler, t in timing.items():
        print(f"  {sampler}: {t:.1f}s")


def cmd_test(args):
    """Quick test with single dataset and few trials."""
    print("Running quick test...")
    
    # Check for API key
    if 'OPENAI_API_KEY' not in os.environ:
        print("WARNING: OPENAI_API_KEY not set. LLM sampler will fail.")
        print("Set it with: export OPENAI_API_KEY=sk-...")
    
    df = run_dataset_benchmark(
        dataset_id=31,  # credit-g
        n_trials=5,
        n_seeds=1,
        samplers=['random', 'tpe'],  # Skip LLM for quick test
        verbose=True,
    )
    
    analyze_results(df)
    print("\nQuick test completed!")


def cmd_test_llm(args):
    """Test LLM sampler specifically."""
    print("Testing LLM Surrogate Sampler...")
    
    if 'OPENAI_API_KEY' not in os.environ:
        print("ERROR: OPENAI_API_KEY not set.")
        print("Set it with: export OPENAI_API_KEY=sk-...")
        return
    
    df = run_dataset_benchmark(
        dataset_id=31,  # credit-g
        n_trials=5,
        n_seeds=1,
        samplers=['llm'],
        verbose=True,
    )
    
    # Print reasoning history
    if len(df) > 0 and 'reasoning_history' in df.columns:
        history = df.iloc[0].get('reasoning_history', [])
        if history:
            print("\n--- LLM Reasoning History ---")
            for entry in history[:3]:
                print(f"\nTrial {entry.get('trial', '?')}:")
                print(f"  Strategy: {entry.get('strategy', 'N/A')}")
                print(f"  Reasoning: {entry.get('reasoning', 'N/A')[:200]}...")
    
    analyze_results(df)


def cmd_benchmark(args):
    """Run full benchmark across all datasets."""
    dataset_ids = TEST_DATASET_IDS_SMALL if args.small else TEST_DATASET_IDS
    
    # Get hybrid sampler parameters
    n_zt = getattr(args, 'n_zerotune', 0)
    n_rand = getattr(args, 'n_random', 0)
    
    print(f"Running benchmark: {len(dataset_ids)} datasets, {args.n_trials} trials, {args.n_seeds} seeds")
    if n_zt > 0 or n_rand > 0:
        print(f"Hybrid LLM config: {n_zt} ZeroTune + {n_rand} random + {args.n_trials - n_zt - n_rand} LLM trials")
    
    all_results = []
    all_raw_results = []
    
    for dataset_id in dataset_ids:
        df = run_dataset_benchmark(
            dataset_id=dataset_id,
            n_trials=args.n_trials,
            n_seeds=args.n_seeds,
            samplers=args.samplers.split(',') if args.samplers else None,
            verbose=True,
            n_zerotune_initial=n_zt,
            n_random_initial=n_rand,
        )
        all_results.append(df)
        all_raw_results.extend(df.to_dict('records'))
    
    # Combine results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Build trial-level data for charts
    trial_df = build_trial_dataframe(all_raw_results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"benchmarks/llm_surrogate_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary CSV
    summary_path = f"{output_dir}/summary.csv"
    combined_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")
    
    # Save trial-level CSV for chart generation
    trials_path = f"{output_dir}/trials.csv"
    trial_df.to_csv(trials_path, index=False)
    print(f"Trial data saved to: {trials_path}")
    
    # Save full results with reasoning as JSON
    json_path = f"{output_dir}/full_results.json"
    with open(json_path, 'w') as f:
        # Filter out non-serializable fields
        serializable = []
        for r in all_raw_results:
            clean = {k: v for k, v in r.items() if k != 'best_params' or isinstance(v, (dict, list, str, int, float, bool, type(None)))}
            serializable.append(clean)
        json.dump(serializable, f, indent=2, default=str)
    print(f"Full results saved to: {json_path}")
    
    # Generate convergence charts
    if args.charts and len(trial_df) > 0:
        generate_convergence_charts(trial_df, combined_df, output_dir)
    
    # Analyze
    analyze_results(combined_df)


def generate_convergence_charts(trial_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: str):
    """Generate convergence charts using existing PublicationChartGenerator."""
    try:
        from zerotune.core.publication_analysis.chart_generator import PublicationChartGenerator
        
        print("\n📊 Generating convergence charts...")
        
        chart_gen = PublicationChartGenerator()
        
        # Build benchmark data format expected by chart generator
        benchmark_data = summary_df.groupby(['dataset_id', 'dataset_name']).agg({
            'best_score': 'mean'
        }).reset_index()
        benchmark_data = benchmark_data.rename(columns={'best_score': 'auc_predicted'})
        
        # Generate per-dataset charts
        convergence_results = {'trial_data': trial_df}
        
        chart_files = chart_gen.generate_convergence_charts(
            convergence_results=convergence_results,
            benchmark_data=benchmark_data,
            algorithm_name="LLM_Surrogate",
            output_dir=output_dir,
            show_confidence_intervals=True,
            only_increasing=True,
            remove_headers=False,
            include_warmstart=False,
            chart_version="v1"
        )
        
        print(f"✅ Generated {len(chart_files)} charts in {output_dir}/plots/")
        
    except ImportError as e:
        print(f"⚠️ Chart generation skipped: {e}")
    except Exception as e:
        print(f"⚠️ Chart generation failed: {e}")


def cmd_single(args):
    """Run on single dataset."""
    n_zt = getattr(args, 'n_zerotune', 0)
    n_rand = getattr(args, 'n_random', 0)
    
    if n_zt > 0 or n_rand > 0:
        print(f"Hybrid LLM config: {n_zt} ZeroTune + {n_rand} random + {args.n_trials - n_zt - n_rand} LLM trials")
    
    df = run_dataset_benchmark(
        dataset_id=args.dataset_id,
        n_trials=args.n_trials,
        n_seeds=args.n_seeds,
        samplers=args.samplers.split(',') if args.samplers else None,
        verbose=True,
        n_zerotune_initial=n_zt,
        n_random_initial=n_rand,
    )
    
    analyze_results(df)


def main():
    parser = argparse.ArgumentParser(description="LLM Surrogate HPO Benchmark")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Quick test (no LLM)')
    
    # Test LLM command
    test_llm_parser = subparsers.add_parser('test-llm', help='Test LLM sampler')
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Full benchmark')
    bench_parser.add_argument('--n_trials', type=int, default=20)
    bench_parser.add_argument('--n_seeds', type=int, default=5)
    bench_parser.add_argument('--samplers', type=str, default='random,tpe,llm')
    bench_parser.add_argument('--small', action='store_true', help='Use small dataset subset (3 datasets)')
    bench_parser.add_argument('--charts', action='store_true', help='Generate convergence charts')
    bench_parser.add_argument('--n_zerotune', type=int, default=0,
                              help='LLM hybrid: initial trials using ZeroTune prediction (default: 0)')
    bench_parser.add_argument('--n_random', type=int, default=0,
                              help='LLM hybrid: trials using random exploration before LLM (default: 0)')
    
    # Single dataset command
    single_parser = subparsers.add_parser('single', help='Single dataset')
    single_parser.add_argument('--dataset_id', type=int, default=31)
    single_parser.add_argument('--n_trials', type=int, default=20)
    single_parser.add_argument('--n_seeds', type=int, default=3)
    single_parser.add_argument('--samplers', type=str, default='random,tpe,llm')
    single_parser.add_argument('--n_zerotune', type=int, default=0,
                               help='LLM hybrid: initial trials using ZeroTune prediction (default: 0)')
    single_parser.add_argument('--n_random', type=int, default=0,
                               help='LLM hybrid: trials using random exploration before LLM (default: 0)')
    
    args = parser.parse_args()
    
    if args.command == 'test':
        cmd_test(args)
    elif args.command == 'test-llm':
        cmd_test_llm(args)
    elif args.command == 'benchmark':
        cmd_benchmark(args)
    elif args.command == 'single':
        cmd_single(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
