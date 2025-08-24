# Publication Charts Guide

## Overview

The publication analysis pipeline now automatically generates **publication-ready PDF charts** alongside LaTeX tables. Charts recreate the logic from your original plotting code and show convergence performance across optimization iterations.

## 🚀 Quick Start (Super Simple!)

Just run your publication analysis as usual - charts are generated automatically:

```bash
# Auto-detect latest files and generate everything (recommended)
python publication_analysis.py DecisionTree --auto-detect
```

That's it! You'll get:
- ✅ Statistical analysis and LaTeX tables
- ✅ PDF convergence charts for each dataset

## 📊 Generated Charts

### Per-Dataset Convergence Charts
- Shows convergence for ZT2 + TPE vs TPE for each specific dataset
- Includes confidence intervals and ZT2 baseline performance line
- Clear dataset ID in filename for easy identification
- Continuous iteration points (1, 2, 3, ..., 20) for smooth curves
- Examples: 
  - `DecisionTree_dataset_917_convergence_20250822_124210.pdf`
  - `DecisionTree_dataset_1049_convergence_20250822_124216.pdf`
  - `DecisionTree_dataset_1111_convergence_20250822_124221.pdf`



## 🎛️ Customization Options

### Generate Charts for Specific Datasets Only
```bash
python publication_analysis.py DecisionTree --auto-detect --chart-datasets 917 1049 1111
```

### Skip Charts (LaTeX Only)
```bash
python publication_analysis.py DecisionTree --auto-detect --no-charts
```

### Skip Convergence Analysis (Basic Comparisons Only)
```bash
python publication_analysis.py DecisionTree --auto-detect --no-convergence
```

## 📁 Output Structure

After running analysis, you'll find charts in the timestamped output directory:

```
publication_outputs/
└── DecisionTree_20250822_115925/
    ├── plots/                          # PDF charts (one per dataset)
    │   ├── DecisionTree_dataset_917_convergence_*.pdf
    │   ├── DecisionTree_dataset_1049_convergence_*.pdf
    │   ├── DecisionTree_dataset_1111_convergence_*.pdf
    │   └── ... (one chart per dataset)
    ├── *.tex                           # LaTeX tables
    └── csv_data/                       # Raw data files
        ├── *_detailed_*.csv
        ├── *_convergence_*.csv
        └── *_statistical_summary_*.csv
```

## 🎯 Chart Features

- **Publication-Ready**: High-quality PDF format for papers and presentations
- **Clean Design**: No titles but with axis labels (remove_headers=True, show_axis_labels=True)
- **Proper Axis Labels**: "Iteration Number" (X-axis) and "Performance (AUC)" (Y-axis)
- **Smooth Curves**: Continuous iteration points (1, 2, 3, 4, ..., 20) for smooth convergence
- **Monotonic Performance**: Shows only increasing performance curves (only_increasing=True)
- **Confidence Intervals**: Statistical uncertainty bands with appropriate width for 50 seeds
- **Baseline Lines**: ZT2 zero-shot performance as reference line
- **Average AUC Values**: Plots actual performance values, not win percentages
- **Method Comparison**: ZT2 + TPE (warmstart) vs TPE (standard) vs ZT2 (zero-shot)

## 💡 Pro Tips

1. **Use `--auto-detect`** - It finds the latest matching files automatically
2. **Charts are publication-ready** - PDF files can be directly included in papers
3. **Filter datasets** with `--chart-datasets` for faster generation during development
4. **All analysis runs in one command** - No separate chart generation step needed

## 🔧 Under the Hood

The chart generation is fully integrated into the `PublicationResultsProcessor`:

```python
from zerotune.core.publication_analysis import PublicationResultsProcessor

processor = PublicationResultsProcessor()
results = processor.process_benchmark_results(
    csv_path="benchmarks/benchmark_results_dt_kb_v1_full_optuna_20250810_002722.csv",
    warmstart_trials_path="benchmarks/optuna_trials_warmstart_dt_kb_v1_full_optuna_20250810_002721.csv",
    standard_trials_path="benchmarks/optuna_trials_standard_dt_kb_v1_full_optuna_20250810_002721.csv",
    algorithm_name="DecisionTree",
    generate_charts=True,  # Default: True
    chart_dataset_ids=None  # Default: all datasets
)

# Charts are in results['chart_files']
```

## 📈 Chart Data Sources

- **Trial Data**: From Optuna warmstart/standard CSV files
- **Benchmark Data**: From benchmark results CSV files  
- **Chart Iterations**: Continuous points [1, 2, 3, 4, ..., 20] for smooth curves
- **LaTeX Table Checkpoints**: Discrete points [1, 5, 10, 15, 20] for statistical analysis

The charts recreate your original plotting logic with:
- Cumulative maximum calculation across seeds (only_increasing=True by default)
- Proper confidence intervals (standard statistical width, 20% opacity, show_confidence_intervals=True by default)
- Clean design with axis labels but no titles (remove_headers=True, show_axis_labels=True by default)
- Average AUC values across all runs at each HPO iteration
- ZT2 baseline performance as reference line
- Continuous iteration points (1, 2, 3, ..., 20) for smooth convergence curves

**Important**: Charts use continuous iterations (1-20) for smooth curves, while LaTeX tables use discrete checkpoints (1, 5, 10, 15, 20) for statistical analysis. 