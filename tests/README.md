# ZeroTune Tests

This directory contains tests for the ZeroTune package. The tests are written using pytest.

## Running the Tests

You can run the tests using the following command:

```bash
# Run all tests
poetry run pytest

# Run tests with coverage report
poetry run pytest --cov=zerotune

# Run a specific test file
poetry run pytest tests/test_zerotune_core.py

# Run a specific test
poetry run pytest tests/test_zerotune_core.py::test_calculate_dataset_meta_parameters
```

## Test Categories

The tests are organized into the following categories:

1. **Core Tests**: Tests for the core functionality of the ZeroTune package.
   - `test_zerotune_core.py`: Tests for dataset meta-parameter calculation, hyperparameter transformation, and model evaluation functions.

2. **Knowledge Base Tests**: Tests for the KnowledgeBase class.
   - `test_knowledge_base.py`: Tests for creating, managing, and using knowledge bases for model training.

3. **Predictor Tests**: Tests for the predictor classes.
   - `test_predictors.py`: Tests for the ZeroTunePredictor and CustomZeroTunePredictor classes.

4. **Pretrained Model Tests**: Tests for the pretrained models.
   - `test_pretrained_model.py`: Tests for the included pretrained models like the decision tree classifier.

## Adding New Tests

When adding new tests, follow these guidelines:

1. Create test files with the prefix `test_`.
2. Create test functions with the prefix `test_`.
3. Group related tests in classes with the prefix `Test`.
4. Use descriptive names for test functions that clearly indicate what is being tested.
5. Use fixtures from `conftest.py` for common setup.
6. For slow or resource-intensive tests, use the `@pytest.mark.slow` decorator.

## Test Fixtures

Common test fixtures are defined in `conftest.py`, including:

- `small_classification_dataset`: A small classification dataset for testing.
- `small_regression_dataset`: A small regression dataset for testing.
- `mock_dataset_meta_parameters`: Mock dataset meta parameters for testing.
- `decision_tree_param_config`: A parameter configuration for Decision Tree models. 