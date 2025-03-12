"""
Tests for the environment setup.

These tests validate that the test environment is correctly configured with
all required dependencies at the correct versions.
"""
import pytest
import sys
import importlib
import pkg_resources


def test_python_version():
    """Test that Python version is compatible with ZeroTune requirements."""
    # ZeroTune requires Python 3.8.1+
    version_info = sys.version_info
    
    # Format current Python version for display in assertion message
    current_version = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
    min_version = "3.8.1"
    
    # Check major version
    assert version_info.major == 3, f"Python major version must be 3, got {current_version}"
    
    # Check minor version
    assert version_info.minor >= 8, f"Python minor version must be >= 8, got {current_version}"
    
    # If minor version is 8, check micro version
    if version_info.minor == 8:
        assert version_info.micro >= 1, f"Python version must be >= {min_version}, got {current_version}"


@pytest.mark.parametrize("package,min_version", [
    ("pandas", "1.3.0"),
    ("numpy", "1.20.0"),
    ("scipy", "1.7.0"),
    ("sklearn", "1.0.0"),
    ("optuna", "3.0.0"),
    ("joblib", "1.1.0"),
    ("tqdm", "4.62.0"),
])
def test_dependencies_installed(package, min_version):
    """
    Test that required dependencies are installed with correct versions.
    
    Args:
        package: The package name to check
        min_version: The minimum required version
    """
    # Special case for scikit-learn
    if package == "sklearn":
        actual_package = "scikit-learn"
    else:
        actual_package = package
    
    # Check if the package is installed
    try:
        # Try to import the package
        module = importlib.import_module(package)
        assert module is not None, f"Failed to import {package}"
        
        # Check the version
        installed_version = pkg_resources.get_distribution(actual_package).version
        
        # Parse versions for comparison
        installed_parts = [int(x) for x in installed_version.split(".")]
        required_parts = [int(x) for x in min_version.split(".")]
        
        # Compare major, minor, patch versions
        for i in range(min(len(installed_parts), len(required_parts))):
            if installed_parts[i] > required_parts[i]:
                # Higher version is fine
                break
            elif installed_parts[i] < required_parts[i]:
                assert False, f"{package} version should be >= {min_version}, got {installed_version}"
        
    except (ImportError, pkg_resources.DistributionNotFound):
        assert False, f"Required dependency {package} is not installed"
    except Exception as e:
        assert False, f"Error checking {package}: {str(e)}"


def test_optional_dependencies():
    """Test for optional dependencies that enhance functionality but aren't required."""
    optional_packages = [
        "matplotlib",
        "openml"
    ]
    
    # Check if each package is available and report status
    for package in optional_packages:
        try:
            importlib.import_module(package)
            # If we get here, the package is installed
            print(f"Optional dependency '{package}' is available")
        except ImportError:
            # Package not available, but that's ok for optional dependencies
            print(f"Optional dependency '{package}' is not available")


def test_zerotune_importable():
    """Test that the zerotune package itself can be imported."""
    try:
        import zerotune
        assert zerotune.__name__ == "zerotune"
        
        # Check that key modules are importable
        from zerotune import calculate_dataset_meta_parameters
        from zerotune import KnowledgeBase
        from zerotune import ZeroTunePredictor
        
        # If we get here, imports were successful
        assert True
    except ImportError as e:
        assert False, f"Failed to import zerotune: {str(e)}" 