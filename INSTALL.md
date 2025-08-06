# Installing ZeroTune

ZeroTune can be installed in multiple ways, depending on your preference:

## Option 1: Using Poetry (Recommended)

Poetry is a modern dependency management tool for Python that simplifies package management.

```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Clone the repository and navigate to it
git clone https://github.com/yourusername/zerotune.git
cd zerotune

# Install dependencies and the package
poetry install
```

### Running with Poetry 2.0+

If you're using Poetry 2.0 or newer (check with `poetry --version`), you have two options:

```bash
# Option 1: Use poetry run (recommended and simplest approach)
poetry run zerotune demo --model xgboost

# Option 2: Manually activate the virtual environment
# First, get your virtual environment path
poetry env info

# Then activate it using source
source /path/to/your/virtualenv/bin/activate  # Replace with the path shown by poetry env info

# Now you can run zerotune commands directly
zerotune demo --model xgboost
```

### Running with Poetry 1.x

For older versions of Poetry:

```bash
# Option 1: Using Poetry run
poetry run zerotune demo --model xgboost

# Option 2: If you've activated the poetry shell
poetry shell
zerotune demo --model xgboost
```

## Option 2: Using pip

For those who prefer pip, you can install ZeroTune using the traditional setup.py:

```bash
# Clone the repository and navigate to it
git clone https://github.com/yourusername/zerotune.git
cd zerotune

# Install in development mode
pip install -e .
```

After installation, you can run ZeroTune from anywhere:

```bash
zerotune demo --model xgboost
```

## Running ZeroTune Without Installation

If you prefer not to install the package, you can still run ZeroTune directly:

```bash
# Using the Python module syntax
python -m zerotune demo --model xgboost

# Using the entry point script
./zerotune_cli.py demo --model xgboost

# Using the provided shell script
./run.sh --demo --model xgboost
```

## Verifying Installation

To verify that ZeroTune is properly installed, run:

```bash
zerotune --help
```

This should display the help message with available commands and options.

## Troubleshooting

If you encounter any issues:

1. **Command not found**: Make sure the package is installed and you're in the correct environment.
   - For Poetry 2.0+: Always use `poetry run zerotune` or manually activate the environment first
   - For pip: Ensure the installation was successful with `pip list | grep zerotune`

2. **Python Version**: Make sure your Python version is 3.8 or newer
   - Check with `python --version`

3. **Poetry Version**: Different Poetry versions have different commands
   - Check your version with `poetry --version`
   - Poetry 2.0+ no longer includes the `shell` command by default

4. **Environment Issues**: Ensure you're running from the correct environment
   - For Poetry: Check the environment with `poetry env info`
   - For pip: Make sure your PATH includes the bin directory where zerotune is installed

5. **Dependency Issues**: Try updating your dependencies
   - For Poetry: `poetry update`
   - For pip: `pip install --upgrade -e .`

For more details, refer to the README.md file and the [Poetry documentation](https://python-poetry.org/docs/). 