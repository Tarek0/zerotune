# Contributing to ZeroTune

Thank you for your interest in contributing to ZeroTune! This document provides guidelines for contributing to our zero-shot hyperparameter optimization project.

## 🚀 Quick Start for Contributors

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/zerotune.git
cd zerotune

# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate virtual environment
poetry shell

# Run tests to verify setup
poetry run pytest
```

### Development Environment

- **Python**: 3.8-3.11 supported
- **Package Manager**: Poetry for dependency management
- **Testing**: pytest for unit tests
- **Code Style**: Black for formatting, pylint for linting

## 📋 Types of Contributions

### 1. **Model Support (High Priority)**
- Add new ML models (e.g., LightGBM, CatBoost)
- Improve existing model configurations
- Optimize hyperparameter ranges

### 2. **Performance Improvements**
- Knowledge base optimizations
- Meta-feature engineering
- Predictor training enhancements

### 3. **Evaluation & Benchmarking**
- New evaluation datasets
- Performance metrics improvements
- Fair benchmarking methodology

### 4. **Documentation**
- Code documentation
- Usage examples
- Performance analysis

## 🔬 Research Contributions

### Adding New Models

1. **Create model configuration** in `zerotune/core/model_configs.py`:
```python
@staticmethod
def get_your_model_config() -> ModelConfig:
    return {
        "name": "YourModel",
        "model": YourModelClass(random_state=42),
        'metric': 'roc_auc',
        'param_config': {
            'param1': {'min_value': 1, 'max_value': 100, 'param_type': "int"},
            'param2': {'percentage_splits': [0.1, 0.3, 0.5, 0.8], 'param_type': "float"}
        }
    }
```

2. **Update model registry** in `zerotune/core/config.py`

3. **Create experiment script** following the pattern of `decision_tree_experiment.py`

4. **Run comprehensive evaluation**:
   - Build knowledge base (50 HPO runs per dataset)
   - Train zero-shot predictor
   - Evaluate with 50-seed robustness testing

### Evaluation Standards

All model contributions must include:

- **Knowledge base**: 15 diverse datasets, 50 HPO trials each
- **Fair benchmarking**: Random baselines using identical hyperparameter ranges
- **Statistical validation**: 50-seed evaluation (500 total experiments)
- **Performance documentation**: Win rate, average improvement, confidence intervals

## 📊 Performance Benchmarks

### Current Performance Rankings

| Model | Win Rate | Avg Improvement | Status |
|-------|----------|-----------------|--------|
| **Decision Tree** | 100% (10/10) | +5.6% | ✅ Production |
| **Random Forest** | TBD | TBD | 🚧 In Progress |
| **XGBoost** | 30% (3/10) | +0.7% | ⚠️ Needs Improvement |

### Performance Requirements for New Models

- **Minimum**: 70% win rate, +3% average improvement
- **Good**: 80% win rate, +4% average improvement  
- **Excellent**: 90%+ win rate, +5%+ average improvement

## 🧪 Testing Guidelines

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run specific test categories
poetry run pytest tests/unit/
poetry run pytest tests/integration/
poetry run pytest tests/experiments/

# Run with coverage
poetry run pytest --cov=zerotune --cov-report=html
```

### Test Requirements

- **Unit tests**: All new functions and classes
- **Integration tests**: End-to-end workflows
- **Performance tests**: Benchmark against existing models
- **Regression tests**: Ensure no performance degradation

## 📝 Code Style

### Formatting Standards

```bash
# Format code
poetry run black zerotune/

# Check imports
poetry run isort zerotune/ --check-only

# Lint code
poetry run pylint zerotune/
```

### Documentation Standards

- **Docstrings**: All public functions and classes
- **Type hints**: All function parameters and returns
- **Comments**: Complex algorithms and business logic
- **README updates**: For new features and performance improvements

## 🤝 Contribution Process

### 1. **Issue Discussion**
- Open issue to discuss your contribution idea
- Get feedback from maintainers before starting work
- Reference existing issues when possible

### 2. **Development**
- Create feature branch: `git checkout -b feature/your-feature-name`
- Follow coding standards and testing requirements
- Document your changes thoroughly

### 3. **Performance Validation**
- Run full evaluation suite
- Compare against current benchmarks
- Include performance metrics in PR description

### 4. **Pull Request**
- Clear title and description
- Reference related issues
- Include test results and performance data
- Update documentation as needed

### 5. **Review Process**
- Code review by maintainers
- Performance validation
- CI/CD pipeline checks
- Documentation review

## 🔍 Advanced Contributions

### Research Areas

1. **Meta-Learning Improvements**
   - Advanced feature engineering
   - Transfer learning across model types
   - Multi-objective optimization

2. **Scalability**
   - Large-scale knowledge base building
   - Distributed HPO
   - Real-time prediction optimization

3. **Robustness**
   - Out-of-domain generalization
   - Dataset drift handling
   - Uncertainty quantification

### Experimental Framework

Follow our established methodology:

1. **Knowledge Base Building**: Use `ZeroTune` class with 50 HPO runs per dataset
2. **Predictor Training**: Apply RFECV feature selection and top-3 trial filtering  
3. **Fair Evaluation**: Use dataset-aware random baselines
4. **Statistical Validation**: 50-seed evaluation with confidence intervals

## 📞 Getting Help

- **Issues**: GitHub issues for bugs and feature requests
- **Discussions**: GitHub discussions for general questions
- **Email**: contact@zerotune.ai for direct contact
- **Documentation**: Check README.md and inline docstrings

## 📜 Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please:

- Be respectful and constructive in discussions
- Focus on technical merit and research validity
- Help newcomers learn and contribute effectively
- Report any concerns to maintainers

## 🏆 Recognition

Contributors will be:
- Listed in project acknowledgments
- Credited in research publications (when applicable)
- Invited to co-author performance benchmark papers
- Recognized in release notes

Thank you for helping make ZeroTune the best zero-shot hyperparameter optimization framework! 🎉 