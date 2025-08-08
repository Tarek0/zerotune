"""
Publication Analysis Module for ZeroTune

This module provides statistical analysis and LaTeX table generation for publication-ready
benchmark results. It is designed to be algorithm-agnostic and reusable across different
ML algorithms (Decision Trees, Random Forest, XGBoost, etc.).

Key Features:
- Statistical analysis with paired t-tests for significance testing
- Checkpoint analysis for Optuna convergence evaluation  
- LaTeX table generation with proper formatting (bold for best, underline for significance)
- Modular design for easy integration and testing

Usage:
    from zerotune.core.publication_analysis import PublicationResultsProcessor
    
    processor = PublicationResultsProcessor()
    processor.process_benchmark_results(
        csv_path="benchmarks/benchmark_results_dt_kb_v1_full_optuna_20250808_115150.csv",
        warmstart_trials_path="benchmarks/optuna_trials_warmstart_dt_kb_v1_full_optuna_20250808_115150.csv",
        standard_trials_path="benchmarks/optuna_trials_standard_dt_kb_v1_full_optuna_20250808_115150.csv"
    )
"""

from .results_processor import PublicationResultsProcessor
from .stats_analyzer import PublicationStatsAnalyzer
from .checkpoint_analyzer import CheckpointAnalyzer
from .latex_generator import LatexTableGenerator

__all__ = [
    'PublicationResultsProcessor',
    'PublicationStatsAnalyzer',
    'CheckpointAnalyzer',
    'LatexTableGenerator',
]

__version__ = '1.0.0' 