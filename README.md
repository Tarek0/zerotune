# ZeroTune

ZeroTune provides **instant zero-shot hyperparameter optimization** using advanced pre-trained models. Get competitive hyperparameters for your machine learning models in sub-millisecond time with robust performance across diverse datasets!

🎯 **Average AUC: 0.8610** • 🚀 **60% datasets outperform random** • ⚡ **<1ms prediction time** • 🧠 **RFECV feature selection**

## 🚀 Quick Start (Zero-Shot Predictions)

```python
from zerotune import ZeroTunePredictor
from xgboost import XGBClassifier
import pandas as pd

# Load your dataset
df = pd.read_csv('your_dataset.csv')
X = df.drop('target', axis=1)
y = df['target']

# Get optimal hyperparameters instantly
predictor = ZeroTunePredictor(model_name='xgboost', task_type='binary')
best_params = predictor.predict(X, y)

# Train model with predicted hyperparameters
model = XGBClassifier(**best_params)
model.fit(X, y)

print(f"Optimal hyperparameters: {best_params}")
```

## ✨ Features

### Zero-Shot Hyperparameter Optimization
- **Instant predictions** using pre-trained models with advanced evaluation metrics
- **No optimization time** required - get results in milliseconds
- **Competitive hyperparameters** with 60% datasets showing positive uplift vs random
- **RFECV feature selection** focuses on the most predictive meta-features
- Support for **XGBoost**, **Random Forest**, and **Decision Tree** models
- **Binary**, **multiclass**, and **regression** tasks supported
- **Custom model training** from your own knowledge bases

### Advanced Evaluation & Quality Assurance
- **NMAE (Normalized Mean Absolute Error)**: Scale-independent accuracy measurement
- **Top-K Accuracy**: Quantifies superiority over random hyperparameter selection
- **Single-seed HPO**: Efficient training data collection with robust numerical stability
- **Top-K filtering**: Uses only the best-performing trials for predictor training
- **Cross-validated feature selection**: RFECV eliminates noisy meta-features

### Knowledge Base Building (For Training New Predictors)
- Collect comprehensive HPO experiment data from multiple datasets
- Extract 22+ dataset meta-features with statistical moments and numerical stability
- Single-seed optimization for efficient and robust training data collection
- Build high-quality training datasets for new zero-shot predictors
- Support for custom dataset collections and experiment configurations

## 🎯 Supported Models

| Model | Binary Classification | Multiclass Classification | Regression |
|-------|----------------------|---------------------------|------------|
| **XGBoost** | ✅ | ✅ | ✅ |
| **Random Forest** | ✅ | ✅ | ✅ |
| **Decision Tree** | ✅ | ✅ | ✅ |

## 📦 Installation

```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install ZeroTune
git clone https://github.com/yourusername/zerotune.git
cd zerotune
poetry install
```

## 🔧 Usage

### 1. Zero-Shot Predictions (Main Use Case)

```python
from zerotune import ZeroTunePredictor

# For different models and tasks
predictor_xgb = ZeroTunePredictor(model_name='xgboost', task_type='binary')
predictor_rf = ZeroTunePredictor(model_name='random_forest', task_type='multiclass')
predictor_dt = ZeroTunePredictor(model_name='decision_tree', task_type='regression')

# Get predictions
hyperparams = predictor_xgb.predict(X, y)
```

### 2. Building Knowledge Bases (For Training New Predictors)

```python
from zerotune import ZeroTune
from zerotune.core.predictor_training import train_predictor_from_knowledge_base

# Build knowledge base from multiple datasets with multi-seed optimization
zt = ZeroTune(model_type='xgboost', kb_path='my_knowledge_base.json')
dataset_ids = [31, 38, 44, 52, 151]  # OpenML dataset IDs
kb = zt.build_knowledge_base(dataset_ids=dataset_ids, n_iter=20)

# Train a new zero-shot predictor from the knowledge base
model_path = train_predictor_from_knowledge_base(
    kb_path='my_knowledge_base.json',
    model_name='xgboost',
    task_type='binary',
    top_k_per_seed=3  # Use only top-3 trials per seed for training
)

# Knowledge base contains:
# - 22+ meta-features per dataset (statistical moments, distributions)
# - Multi-seed HPO results (10 seeds × 20 iterations = 200 trials per dataset)
# - Top-performing hyperparameters with performance scores
# - Full Optuna trials dataframe for advanced analysis
# - RFECV feature selection applied during predictor training
```

### 3. Command Line Interface

```bash
# Complete experimental workflow
poetry run python xgb_experiment.py info         # Show dataset information

# Quick development cycle (2 datasets)
poetry run python xgb_experiment.py test         # Build knowledge base
poetry run python xgb_experiment.py train-test   # Train predictor
poetry run python xgb_experiment.py eval-test    # Evaluate on unseen data

# Full production cycle (15 datasets)
poetry run python xgb_experiment.py full         # Build comprehensive KB
poetry run python xgb_experiment.py train-full   # Train robust predictor
poetry run python xgb_experiment.py eval-full    # Evaluate performance
```

## 🏗️ Architecture

```
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   Knowledge Base    │───▶│   Pre-trained Model  │───▶│  Zero-Shot Predict  │
│   Building          │    │   Training           │    │  (ZeroTunePredictor)│
│   (ZeroTune)        │    │                      │    │                     │
│                     │    │                      │    │                     │
│ • Multi-seed HPO on │    │ • RFECV feature      │    │ • Sub-ms prediction │
│   many datasets     │    │   selection (15/22)  │    │ • NMAE: 23% error   │
│ • Extract 22+ meta- │    │ • Top-K filtering    │    │ • 100% > random     │
│   features          │    │ • RandomForest +HPO  │    │ • Feature selection │
│ • Store full trials │    │ • Meta-features →    │    │ • High performance  │
│   dataframes        │    │   Hyperparameters    │    │                     │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
```

### Key Technical Components

1. **Knowledge Base Building** (`ZeroTune`):
   - Single-seed HPO with robust numerical stability
   - Comprehensive meta-feature extraction (22+ features with clipping)
   - Full Optuna trials storage for advanced analysis

2. **Predictor Training** (`train_predictor_from_knowledge_base`):
   - RFECV feature selection with GroupKFold cross-validation
   - Top-K filtering (top-3 trials per dataset)
   - Hyperparameter optimization of the predictor itself
   - Advanced evaluation metrics (NMAE, Top-K accuracy)

3. **Zero-Shot Prediction** (`ZeroTunePredictor`):
   - Instant hyperparameter prediction (<1ms)
   - Automatic feature selection application
   - Competitive performance across diverse datasets

## 📊 Performance

### Zero-Shot Predictor Quality Metrics

**Predictor Training Performance**:
- **Low prediction error** with NMAE-based evaluation across all hyperparameters
- **Intelligent feature selection** via RFECV retaining the most predictive meta-features
- **High-quality training data** using only top-performing HPO trials
- **Robust cross-validation** with GroupKFold to prevent data leakage

**Hyperparameter Prediction Quality**:
- **Continuous parameters** (learning_rate, subsample, colsample_bytree) typically show best prediction accuracy
- **Discrete parameters** (max_depth) have moderate prediction challenges
- **Wide-range parameters** (n_estimators) are most challenging but still competitive

### Real-World Evaluation on Unseen Datasets

Zero-shot predictor provides competitive performance across diverse datasets with:

- **Instant predictions** in sub-millisecond time
- **No data leakage** - evaluation on completely unseen datasets
- **Competitive AUC scores** across various domain types and dataset sizes
- **Consistent performance** from small (500 samples) to large (50K+ samples) datasets
- **Positive uplift** on majority of datasets compared to random hyperparameter selection
**Evaluation Summary**: 
- **Competitive performance** across diverse dataset types and sizes
- **Majority positive uplift** compared to random hyperparameter selection
- **Instant predictions** - sub-millisecond time vs hours of traditional HPO
- **Production ready** with robust numerical stability and error handling

### Understanding the Evaluation Metrics

**NMAE (Normalized Mean Absolute Error)**:
- Measures prediction accuracy on a 0-100% scale (lower is better)
- Scale-independent: all hyperparameters normalized to [0,1] range
- Low NMAE means predictions are close to optimal values on average

**Top-K Accuracy**:
- Percentage of predictions that outperform random hyperparameter selection
- 100% means every single prediction beats random baselines
- Demonstrates real practical value over naive approaches

**RFECV Feature Selection**:
- Recursive Feature Elimination with Cross-Validation
- Automatically identifies the most predictive meta-features (15/22 retained)
- Focuses model on statistical moments of feature and row distributions

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and contribution guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 