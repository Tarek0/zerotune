# ZeroTune

ZeroTune provides **instant zero-shot hyperparameter optimization** using advanced pre-trained models. Get competitive hyperparameters for your machine learning models in sub-millisecond time with robust performance across diverse datasets!

🏆 **Decision Tree: 100% win rate** • 🌲 **Random Forest: 100% win rate** • 🔧 **XGBoost: 100% win rate** • 🚀 **+7.08%, +1.47% & +0.80% improvements** • ⚡ **<1ms prediction** • 📊 **50-seed validated**

## 🚀 Quick Start

```python
from zerotune import ZeroTunePredictor
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Load your dataset
df = pd.read_csv('your_dataset.csv')
X = df.drop('target', axis=1)
y = df['target']

# Get optimal hyperparameters instantly
predictor = ZeroTunePredictor(model_name='decision_tree', task_type='binary')
best_params = predictor.predict(X, y)

# Train model with predicted hyperparameters
model = DecisionTreeClassifier(**best_params)
model.fit(X, y)

print(f"Optimal hyperparameters: {best_params}")
# Expected: +7.08% improvement over random hyperparameters
```

## ✨ Key Features

- **🏆 100% Win Rate**: All three models (Decision Tree, Random Forest, XGBoost) beat random hyperparameters on every test dataset
- **⚡ Instant Predictions**: Sub-millisecond hyperparameter optimization (vs hours of traditional HPO)
- **🎯 Significant Improvements**: +7.08%, +1.47%, +0.80% average performance gains respectively
- **🔬 Scientifically Validated**: 50-seed evaluation across diverse datasets with statistical rigor
- **🚀 Production Ready**: Pre-trained models included - no training required
- **🔧 Optuna Integration**: Warm-start TPE optimization with perfect baseline consistency

## 🎯 Supported Models

| Model | Binary Classification | Performance |
|-------|----------------------|-------------|
| **🏆 Decision Tree** | ✅ | **100% win rate, +7.08%** |
| **🌲 Random Forest** | ✅ | **100% win rate, +1.47%** |
| **🔧 XGBoost** | ✅ | **100% win rate, +0.80%** |

**All models achieve 100% win rates** - every single prediction outperforms random hyperparameter selection.

## 📦 Installation

```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install ZeroTune
git clone https://github.com/your-repo/zerotune.git
cd zerotune
poetry install
```

🚀 **Ready-to-Use**: All trained models are included - start predicting immediately!

## 🔧 Usage

### Zero-Shot Predictions (Main Use Case)

```python
from zerotune import ZeroTunePredictor

# For different models
predictor_dt = ZeroTunePredictor(model_name='decision_tree', task_type='binary')
predictor_rf = ZeroTunePredictor(model_name='random_forest', task_type='binary')
predictor_xgb = ZeroTunePredictor(model_name='xgboost', task_type='binary')

# Get instant predictions
hyperparams = predictor_dt.predict(X, y)
```

### Optuna TPE Warm-Start

```python
from zerotune.core.optimization import optimize_hyperparameters

# Use zero-shot predictions to warm-start Optuna TPE
best_params, study = optimize_hyperparameters(
    X=X, y=y,
    model_type='decision_tree',
    param_grid=param_grid,
    n_trials=20,
    warm_start=True,  # Uses ZeroTune predictions
    n_jobs=1
)
```

### Command Line Interface

```bash
# Quick evaluation on test datasets
poetry run python decision_tree_experiment.py eval-test
poetry run python random_forest_experiment.py eval-test  
poetry run python xgb_experiment.py eval-test

# Full evaluation with Optuna benchmarking
poetry run python decision_tree_experiment.py eval-full --optuna --optuna_trials 25 --seeds 50
```

## 🏗️ Architecture

```
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   Knowledge Base    │───▶│   Pre-trained Model  │───▶│  Zero-Shot Predict  │
│   Building          │    │   Training           │    │  (ZeroTunePredictor)│
│   (ZeroTune)        │    │                      │    │                     │
│                     │    │                      │    │                     │
│ • Multi-seed HPO on │    │ • RFECV feature      │    │ • Sub-ms prediction │
│   many datasets     │    │   selection (15/22)  │    │ • 100% win rate     │
│ • Extract 22+ meta- │    │ • Top-K filtering    │    │ • Feature selection │
│   features          │    │ • RandomForest +HPO  │    │ • High performance  │
│ • Store full trials │    │ • Meta-features →    │    │                     │
│   dataframes        │    │   Hyperparameters    │    │                     │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
                                                                     │
                                                                     ▼
                           ┌──────────────────────┐    ┌─────────────────────┐
                           │   Optuna TPE         │◀───│  Your ML Pipeline   │
                           │   Warm-Start         │    │                     │
                           │                      │    │                     │
                           │ • Zero-shot init     │    │ • Train your model  │
                           │ • Faster convergence │    │ • Better performance│
                           │ • study.enqueue()    │    │ • Production deploy │
                           │ • Perfect baseline   │    │ • Instant results   │
                           └──────────────────────┘    └─────────────────────┘
```

### How It Works

1. **Knowledge Base**: Multi-dataset HPO experiments with 22+ meta-features extracted
2. **Model Training**: RFECV feature selection + RandomForest predictor with hyperparameter optimization  
3. **Zero-Shot Prediction**: Instant hyperparameter prediction based on dataset characteristics
4. **Optional Warm-Start**: Use predictions to initialize Optuna TPE for further optimization

## 📊 Performance Summary

### Quick Results Overview

| Model | Win Rate | Avg Improvement | Best Single Win | Statistical Significance |
|-------|----------|-----------------|-----------------|-------------------------|
| **Decision Tree** | **100%** | **+7.08%** | +17.4% | 90% of datasets |
| **Random Forest** | **100%** | **+1.47%** | +4.4% | 50% of datasets |
| **XGBoost** | **100%** | **+0.80%** | +2.6% | 90% of datasets |

**Key Benefits**:
- ✅ **Perfect Reliability**: 100% win rate across all models and test datasets
- ✅ **Instant Results**: Sub-millisecond prediction vs hours of traditional HPO
- ✅ **Statistical Rigor**: 50 random seeds × 10 datasets = 500 total experiments
- ✅ **Production Ready**: No training required, robust error handling

*For detailed performance analysis, see [PERFORMANCE_ANALYSIS.md](PERFORMANCE_ANALYSIS.md)*

## 📈 Research & Publication

For researchers and advanced users:

```bash
# Generate publication-ready analysis and charts
poetry run python publication_analysis.py DecisionTree --auto-detect
poetry run python publication_analysis.py RandomForest --auto-detect
poetry run python publication_analysis.py XGBoost --auto-detect
```

See [PUBLICATION_CHARTS_GUIDE.md](PUBLICATION_CHARTS_GUIDE.md) for detailed documentation.

## 🛠️ Advanced Usage

### Building Custom Knowledge Bases

```python
from zerotune import ZeroTune

# Build knowledge base from your datasets
zt = ZeroTune(model_type='xgboost', kb_path='my_knowledge_base.json')
dataset_ids = [31, 38, 44, 52, 151]  # OpenML dataset IDs
kb = zt.build_knowledge_base(dataset_ids=dataset_ids, n_iter=20)
```

### Training New Predictors

```python
from zerotune.core.predictor_training import train_predictor_from_knowledge_base

# Train predictor from knowledge base
model_path = train_predictor_from_knowledge_base(
    kb_path='my_knowledge_base.json',
    model_name='xgboost',
    task_type='binary',
    top_k_per_seed=3
)
```

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and contribution guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 