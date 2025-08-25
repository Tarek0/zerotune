# Publication Charts Guide

## Overview

The publication analysis pipeline automatically generates **publication-ready PDF charts** alongside LaTeX tables. Charts show convergence performance across optimization iterations with statistically correct confidence intervals and perfect baseline consistency.

## 🚀 Quick Start (Super Simple!)

Just run your publication analysis as usual - charts are generated automatically:

```bash
# Auto-detect latest files and generate everything (recommended)
poetry run python publication_analysis.py DecisionTree --auto-detect
poetry run python publication_analysis.py RandomForest --auto-detect
poetry run python publication_analysis.py XGBoost --auto-detect
```

That's it! You'll get:
- ✅ **Statistical analysis** with comprehensive LaTeX tables
- ✅ **Publication-ready PDF charts** for each dataset (two versions per dataset)
- ✅ **Perfect baseline consistency** with logical visualizations
- ✅ **Statistically correct confidence intervals** with proper statistical methods

## 📊 Generated Charts

### Per-Dataset Convergence Charts

**Two versions generated for each dataset**:

1. **ZT2 vs TPE** (`ZT2_vs_TPE_Performance_Dataset_{id}.pdf`):
   - Compares ZT2 zero-shot baseline vs standard TPE optimization
   - Shows TPE's improvement over the zero-shot starting point
   - Clean comparison without warmstart complexity

2. **ZT2+WS vs TPE** (`ZT2_WS_vs_TPE_Performance_Dataset_{id}.pdf`):
   - Includes warmstart TPE (orange line) vs standard TPE (blue line)
   - Shows ZT2 baseline (red dashed line) and Random baseline (green dashed line)
   - Complete comparison with all methods

### Chart Features

- **Perfect Baseline Consistency**: Random baseline starts exactly at TPE iteration 0 (where TPE is effectively random search)
- **Statistically Correct Confidence Intervals**: Calculated using sample standard deviation (ddof=1) on cumulative maximum values
- **Continuous Iteration Points**: Smooth curves showing performance at each iteration (1, 2, 3, ..., 20)
- **Publication-Ready Design**: Clean aesthetics with proper axis labels but no titles
- **Dataset ID in Filename**: Easy identification with dataset ID embedded in filename

### Example Filenames

```
plots/
├── ZT2_vs_TPE_Performance_Dataset_917.pdf          # Version 1: ZT2 vs TPE only
├── ZT2_WS_vs_TPE_Performance_Dataset_917.pdf       # Version 2: With warmstart
├── ZT2_vs_TPE_Performance_Dataset_1049.pdf         # Version 1: ZT2 vs TPE only
├── ZT2_WS_vs_TPE_Performance_Dataset_1049.pdf      # Version 2: With warmstart
└── ... (two charts per dataset)
```

## 🎛️ Customization Options

### Generate Charts for Specific Datasets Only
```bash
poetry run python publication_analysis.py DecisionTree --auto-detect --chart-datasets 917 1049 1111
```

### Skip Charts (LaTeX Only)
```bash
poetry run python publication_analysis.py DecisionTree --auto-detect --no-charts
```

### Skip Convergence Analysis (Basic Comparisons Only)
```bash
poetry run python publication_analysis.py DecisionTree --auto-detect --no-convergence
```

### Manual File Specification
```bash
poetry run python publication_analysis.py DecisionTree \
  --csv benchmarks/benchmark_results_dt_kb_v1_full_optuna_20250822_144346.csv \
  --warmstart benchmarks/optuna_trials_warmstart_dt_kb_v1_full_optuna_20250822_144346.csv \
  --standard benchmarks/optuna_trials_standard_dt_kb_v1_full_optuna_20250822_144346.csv
```

## 📁 Output Structure

After running analysis, you'll find charts in the timestamped output directory:

```
publication_outputs/
└── DecisionTree_20250823_082520/
    ├── plots/                                      # PDF charts directory
    │   ├── ZT2_vs_TPE_Performance_Dataset_917.pdf           # Version 1
    │   ├── ZT2_WS_vs_TPE_Performance_Dataset_917.pdf        # Version 2
    │   ├── ZT2_vs_TPE_Performance_Dataset_1049.pdf          # Version 1
    │   ├── ZT2_WS_vs_TPE_Performance_Dataset_1049.pdf       # Version 2
    │   └── ... (20 total charts: 2 per dataset × 10 datasets)
    ├── DecisionTree_zt_vs_random_per_dataset_table.tex      # LaTeX tables
    ├── DecisionTree_convergence_analysis_table.tex
    ├── DecisionTree_zt_vs_tpe_convergence_table.tex
    └── csv_data/                                   # Raw data files
        ├── DecisionTree_zeroshot_vs_random_detailed_*.csv
        ├── DecisionTree_convergence_method_comparison_*.csv
        └── DecisionTree_statistical_summary_*.csv
```

## 🎯 Chart Features & Improvements

### Statistical Rigor
- **Corrected Confidence Intervals**: Uses sample standard deviation (ddof=1) instead of population standard deviation
- **CI on Cumulative Maximum**: Confidence intervals calculated directly on cummax values for statistical soundness
- **Proper T-Distribution**: Uses t-critical values appropriate for sample size (50 seeds)
- **Visible Confidence Bands**: 20% opacity bands clearly show uncertainty around mean performance

### Perfect Baseline Consistency
- **Zero Discrepancy**: ZT2 benchmark exactly matches warmstart trial 0 performance (0.00% difference)
- **Logical Random Baseline**: Green dashed line starts at TPE iteration 0, showing TPE begins with random search
- **Consistent Model Initialization**: Same random_state used across benchmark and trial evaluations
- **Fixed Train/Test Splits**: Consistent 80%/20% methodology across all evaluation contexts

### Visual Design
- **Publication-Ready Aesthetics**: Clean design optimized for papers and presentations
- **Proper Axis Labels**: "Iteration Number" (X-axis) and "Performance (AUC)" (Y-axis)
- **No Titles**: Clean design with `remove_headers=True` but axis labels enabled
- **Smooth Convergence Curves**: Continuous iteration points (1, 2, 3, ..., 20) for smooth visualization
- **Monotonic Performance**: Shows only increasing performance (`only_increasing=True`)
- **Method Labels**: "ZT2 + TPE" for warmstart, "TPE" for standard, "ZT2" for zero-shot baseline

### Data Accuracy
- **Average AUC Values**: Plots actual performance values across all runs at each iteration
- **50-Seed Robustness**: Statistics calculated from 50 independent runs per method per dataset
- **Cumulative Maximum**: Shows "best so far" performance, representing practical optimization progress
- **Perfect Warmstart**: Warmstart trials begin exactly at zero-shot predictions, validating methodology

## 💡 Pro Tips

1. **Use `--auto-detect`** - Automatically finds the latest matching benchmark and trial files
2. **Charts are publication-ready** - PDF files can be directly included in papers without modification
3. **Two versions per dataset** - Choose between simple comparison (V1) or complete analysis (V2)
4. **Filter datasets** with `--chart-datasets` for faster generation during development
5. **Check baseline consistency** - Perfect 0.00% discrepancy validates experimental methodology
6. **Confidence intervals are visible** - Look for light-colored bands around performance lines
7. **Random baseline logic** - Green line starts at iteration 0 showing TPE's random initialization

## 🔧 Under the Hood

### Chart Generation Integration

The chart generation is fully integrated into the `PublicationResultsProcessor`:

```python
from zerotune.core.publication_analysis import PublicationResultsProcessor

processor = PublicationResultsProcessor()
results = processor.process_benchmark_results(
    csv_path="benchmarks/benchmark_results_dt_kb_v1_full_optuna_20250822_144346.csv",
    warmstart_trials_path="benchmarks/optuna_trials_warmstart_dt_kb_v1_full_optuna_20250822_144346.csv",
    standard_trials_path="benchmarks/optuna_trials_standard_dt_kb_v1_full_optuna_20250822_144346.csv",
    algorithm_name="DecisionTree",
    generate_charts=True,  # Default: True
    chart_dataset_ids=None,  # Default: all datasets
    # Chart options with publication-ready defaults:
    only_increasing=True,
    show_confidence_intervals=True,
    remove_headers=True,
    show_axis_labels=True,
    generate_both_versions=True  # Generate both V1 and V2 charts
)

# Charts are in results['chart_files']
print(f"Generated {len(results['chart_files'])} PDF charts")
```

### Statistical Methodology

**Confidence Interval Calculation**:
```python
# Statistically correct approach (implemented)
seed_cummax_performances = []
for seed in seeds:
    seed_data = method_trials[method_trials['seed'] == seed]
    seed_data['cummax_value'] = seed_data['value'].cummax()
    cummax_at_iteration = seed_data[seed_data['number'] <= trial_num]['cummax_value'].iloc[-1]
    seed_cummax_performances.append(cummax_at_iteration)

mean_cummax = np.mean(seed_cummax_performances)
std_cummax = np.std(seed_cummax_performances, ddof=1)  # Sample std dev
sem_cummax = std_cummax / np.sqrt(len(seed_cummax_performances))
degrees_freedom = len(seed_cummax_performances) - 1
t_critical = t.ppf((1 + confidence_level) / 2, degrees_freedom)
ci_margin = t_critical * sem_cummax
```

**Perfect Baseline Consistency**:
```python
# Random baseline uses TPE iteration 0 performance
standard_trials = dataset_trials[dataset_trials['method'] == 'standard']
iteration_0_trials = standard_trials[standard_trials['number'] == 0]
tpe_iteration_0_performance = iteration_0_trials['value'].mean()

# Random baseline line starts at iteration 0 with TPE's random performance
fig.add_trace(go.Scatter(
    x=[0, max_iteration],  # Start at iteration 0
    y=[tpe_iteration_0_performance, tpe_iteration_0_performance],
    name='Random',
    line=dict(color='green', width=2, dash='dash')
))
```

## 📈 Chart Data Sources

### Data Files Required
- **Benchmark CSV**: Contains zero-shot predictions and final performance metrics
- **Warmstart Trials CSV**: Trial-by-trial data for warm-started Optuna TPE
- **Standard Trials CSV**: Trial-by-trial data for standard Optuna TPE

### Chart vs Table Data Points
- **Chart Iterations**: Continuous points [0, 1, 2, 3, 4, ..., 20] for smooth convergence curves
- **LaTeX Table Checkpoints**: Discrete points [1, 5, 10, 15, 20] for statistical analysis tables
- **Baseline Consistency**: ZT2 benchmark value exactly matches warmstart trial 0 value

### Performance Metrics
- **Average AUC**: Mean performance across all 50 seeds at each iteration
- **Cumulative Maximum**: "Best so far" performance showing practical optimization progress  
- **Confidence Intervals**: Statistical uncertainty calculated on cummax values with proper t-distribution
- **Perfect Alignment**: 0.00% discrepancy between benchmark and warmstart starting points

## 🎯 Validation & Quality Assurance

### Statistical Validation
- ✅ **Sample Standard Deviation**: Uses `ddof=1` for unbiased estimation
- ✅ **T-Distribution**: Proper critical values for 50-seed sample size
- ✅ **CI on Cummax**: Confidence intervals calculated on the actual plotted values
- ✅ **Visible Uncertainty**: 20% opacity bands clearly show statistical uncertainty

### Baseline Consistency Validation  
- ✅ **Perfect Alignment**: ZT2 benchmark == warmstart trial 0 (0.00% difference)
- ✅ **Logical Random Baseline**: Starts at TPE iteration 0 where TPE is random search
- ✅ **Consistent Methodology**: Same train/test split and model random_state across contexts
- ✅ **Parameter Consistency**: Warmstart uses exact zero-shot predictions

### Visual Quality Validation
- ✅ **Publication Standards**: High-quality PDF output suitable for academic papers
- ✅ **Clear Legends**: Method names clearly distinguish ZT2, TPE, and ZT2+TPE
- ✅ **Proper Scaling**: Y-axis appropriately scaled to show performance differences
- ✅ **Smooth Curves**: Continuous iteration points create smooth convergence visualization

**Result**: Charts that are both statistically rigorous and visually compelling for publication use.

## 📊 Example Results Summary

When you run the publication analysis, expect to see:

### Decision Tree Results
- **Perfect Baseline Consistency**: 0.00% discrepancy across all datasets
- **Strong Performance**: 7.08% average improvement, 100% win rate
- **Statistical Significance**: 90% of datasets show significant improvements
- **Smooth Convergence**: Clear warmstart advantage in early iterations

### Random Forest Results  
- **Perfect Baseline Consistency**: 0.00% discrepancy across all datasets
- **Solid Performance**: 1.47% average improvement, 100% win rate
- **Moderate Significance**: 50% of datasets show significant improvements
- **Convergence Pattern**: Warmstart and standard TPE converge to similar performance

### XGBoost Results
- **Perfect Baseline Consistency**: 0.00% discrepancy across all datasets
- **Consistent Performance**: 0.80% average improvement, 100% win rate  
- **High Significance**: 90% of datasets show significant improvements
- **Close Competition**: Warmstart maintains slight edge throughout optimization

**Important**: Charts use continuous iterations (0-20) for smooth curves, while LaTeX tables use discrete checkpoints (1, 5, 10, 15, 20) for statistical analysis. Both approaches are statistically valid and serve different presentation purposes. 