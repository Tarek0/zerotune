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
poetry run pytest tests/test_zerotune_core.py::TestZeroTuneCore::test_calculate_dataset_meta_parameters

# Run parameterized tests
poetry run pytest tests/test_predictors.py::TestPredictors::test_custom_model_with_different_features

# Run environment validation tests
poetry run pytest tests/test_environment.py
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

5. **Environment Tests**: Tests for the environment setup.
   - `test_environment.py`: Tests that validate Python version and installed dependencies.

## Edge Case Tests

The test suite includes specific tests for edge cases:

- **Empty datasets**: Tests how functions handle datasets with no rows
- **Single-row datasets**: Tests behavior with datasets containing only one sample
- **High-dimensional datasets**: Tests performance with datasets having many features
- **All-categorical datasets**: Tests handling of datasets with categorical features

## Parameterized Tests

Several tests use pytest's parametrization to test multiple configurations:

- **Multiple model types**: The same test logic applied to different model types
- **Feature combinations**: Tests with different combinations of features
- **Target parameter sets**: Tests with different sets of target parameters
- **Dependency versions**: Tests with various dependency versions

## Test Fixtures

Common test fixtures are defined in `conftest.py`, including:

- `small_classification_dataset`: A small classification dataset for testing.
- `small_regression_dataset`: A small regression dataset for testing.
- `empty_dataset`: An empty dataset for edge case testing.
- `single_row_dataset`: A dataset with just one row for edge case testing.
- `high_dimensional_dataset`: A dataset with many features for performance testing.
- `all_categorical_dataset`: A dataset with only categorical features.
- `mock_dataset_meta_parameters`: Mock dataset meta parameters for testing.
- `decision_tree_param_config`: A parameter configuration for Decision Tree models.

## Adding New Tests

When adding new tests, follow these guidelines:

1. Create test files with the prefix `test_`.
2. Create test functions with the prefix `test_`.
3. Group related tests in classes with the prefix `Test`.
4. Use descriptive names for test functions that clearly indicate what is being tested.
5. Write detailed docstrings explaining what the test is checking.
6. Use fixtures from `conftest.py` for common setup.
7. Consider using parametrized tests for similar test cases.
8. Add proper edge case tests for robustness.
9. For slow or resource-intensive tests, use the `@pytest.mark.slow` decorator. 