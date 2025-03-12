#!/usr/bin/env python
"""
Simple test script to check if OpenML is working correctly.
This script attempts to load a few datasets from OpenML to validate connectivity.
"""

import sys
import pandas as pd
import numpy as np

print("Python version:", sys.version)

try:
    import openml
    print("OpenML version:", openml.__version__)
    
    # Try to load dataset 61 (Iris)
    print("\nTrying to load Iris dataset (ID: 61)...")
    try:
        dataset = openml.datasets.get_dataset(61)
        print("  Dataset name:", dataset.name)
        print("  Default target:", dataset.default_target_attribute)
        
        X, y, categorical, names = dataset.get_data()
        print("  Data shapes:", X.shape if hasattr(X, 'shape') else type(X), 
              y.shape if hasattr(y, 'shape') else type(y))
        print("  Target sample:", y[:3])
        print("  ✓ Successfully loaded dataset")
    except Exception as e:
        print("  ✗ Error loading dataset 61:", str(e))
    
    # Try to load dataset 31 (credit-g)
    print("\nTrying to load credit-g dataset (ID: 31)...")
    try:
        dataset = openml.datasets.get_dataset(31)
        print("  Dataset name:", dataset.name)
        print("  Default target:", dataset.default_target_attribute)
        
        X, y, categorical, names = dataset.get_data()
        print("  Data shapes:", X.shape if hasattr(X, 'shape') else type(X), 
              y.shape if hasattr(y, 'shape') else type(y))
        print("  Target sample:", y[:3])
        print("  ✓ Successfully loaded dataset")
    except Exception as e:
        print("  ✗ Error loading dataset 31:", str(e))
    
    # Try to list available datasets
    print("\nTrying to list some available datasets...")
    try:
        datasets = openml.datasets.list_datasets(size=5)
        print("  Retrieved", len(datasets), "datasets:")
        for dataset_id, dataset_info in list(datasets.items())[:5]:
            print(f"  - {dataset_info['name']} (ID: {dataset_id})")
        print("  ✓ Successfully listed datasets")
    except Exception as e:
        print("  ✗ Error listing datasets:", str(e))
        
except ImportError as e:
    print("Failed to import OpenML:", str(e))
    print("Please install OpenML: pip install openml")
except Exception as e:
    print("Unexpected error:", str(e))

print("\nTest completed.") 