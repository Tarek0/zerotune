# ZeroTune

ZeroTune provides **instant zero-shot hyperparameter optimization** using pre-trained models. Get optimal hyperparameters for your machine learning models without any optimization time!

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
- **Instant predictions** using pre-trained models
- **No optimization time** required
- **High-quality hyperparameters** trained on diverse datasets
- Support for **XGBoost**, **Random Forest**, and **Decision Tree** models
- **Binary**, **multiclass**, and **regression** tasks supported
- **Custom model training** from your own knowledge bases

### Knowledge Base Building (For Training New Predictors)
- Collect HPO experiment data from multiple datasets
- Extract comprehensive dataset meta-features
- Build training datasets for new zero-shot predictors
- Support for custom dataset collections

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

# Build knowledge base from multiple datasets
zt = ZeroTune(model_type='xgboost', kb_path='my_knowledge_base.json')
dataset_ids = [31, 38, 44, 52, 151]  # OpenML dataset IDs
kb = zt.build_knowledge_base(dataset_ids=dataset_ids, n_iter=20)

# Train a new zero-shot predictor from the knowledge base
model_path = train_predictor_from_knowledge_base(
    kb_path='my_knowledge_base.json',
    model_name='xgboost',
    task_type='binary'
)

# Knowledge base contains:
# - Meta-features for each dataset  
# - Optimal hyperparameters found
# - Performance scores achieved
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
│ • Run HPO on many   │    │ • Train ML model on  │    │ • Instant prediction│
│   datasets          │    │   KB data            │    │ • No optimization   │
│ • Extract meta-     │    │ • Meta-features →    │    │ • High performance  │
│   features          │    │   Hyperparameters    │    │                     │
│ • Store results     │    │                      │    │                     │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
```

## 📊 Performance

Zero-shot predictor evaluation on 10 unseen datasets (no data leakage):

| Dataset ID | Dataset Name | Test AUC | Prediction Time |
|------------|-------------|----------|----------------|
| 1510 | wdbc | **0.9918** | **Instant** |
| 4534 | PhishingWebsites | **0.9913** | **Instant** |
| 917 | fri_c1_1000_25 | **0.9790** | **Instant** |
| 1049 | pc4 | **0.9571** | **Instant** |
| 1494 | qsar-biodeg | **0.9338** | **Instant** |
| 1558 | bank-marketing | **0.9179** | **Instant** |
| 40536 | SpeedDating | **0.8675** | **Instant** |
| 1111 | KDDCup09_appetency | **0.8447** | **Instant** |
| 1464 | blood-transfusion | **0.6809** | **Instant** |
| 23381 | dresses-sales | **0.5899** | **Instant** |

**Summary**: Average AUC **0.8754 ± 0.1304** across 10 diverse datasets (trained on only 2 datasets)

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and contribution guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 