#!/usr/bin/env python
"""
Detailed test script to debug OpenML dataset loading issues.
This script tests different ways of accessing OpenML data to identify the issue.
"""

import sys
import pandas as pd
import numpy as np

print("Python version:", sys.version)

try:
    import openml
    print("OpenML version:", openml.__version__)
    
    # Test dataset 61 (Iris) with different methods
    print("\nTesting Iris dataset (ID: 61) with different methods...")
    try:
        dataset = openml.datasets.get_dataset(61)
        print("  Dataset name:", dataset.name)
        print("  Default target:", dataset.default_target_attribute)
        
        # Method 1: Standard get_data
        print("\n  Method 1: Standard get_data()...")
        try:
            X, y, categorical, names = dataset.get_data()
            print("    X type:", type(X))
            print("    y type:", type(y))
            if y is not None:
                print("    First few y values:", y[:5])
            else:
                print("    y is None!")
        except Exception as e:
            print("    Error:", str(e))
        
        # Method 2: Specify target
        print("\n  Method 2: Specify target attribute...")
        try:
            X, y, categorical, names = dataset.get_data(target=dataset.default_target_attribute)
            print("    X type:", type(X))
            print("    y type:", type(y))
            if y is not None:
                print("    First few y values:", y[:5])
            else:
                print("    y is None!")
        except Exception as e:
            print("    Error:", str(e))
        
        # Method 3: Get DataFrame and separate target
        print("\n  Method 3: Get as DataFrame...")
        try:
            df, _, _, _ = dataset.get_data(dataset_format="dataframe")
            print("    DataFrame shape:", df.shape)
            print("    DataFrame columns:", df.columns.tolist())
            
            # Extract target from DataFrame
            if dataset.default_target_attribute in df.columns:
                y = df[dataset.default_target_attribute]
                X = df.drop(dataset.default_target_attribute, axis=1)
                print("    X shape:", X.shape)
                print("    y shape:", y.shape)
                print("    First few y values:", y.head())
            else:
                print("    Target column not found in DataFrame!")
        except Exception as e:
            print("    Error:", str(e))
            
        # Method 4: Get as array and convert
        print("\n  Method 4: Get as array and convert manually...")
        try:
            X_arr, y_arr, categorical, names = dataset.get_data(dataset_format="array")
            print("    X array shape:", X_arr.shape if hasattr(X_arr, 'shape') else "no shape")
            print("    y array type:", type(y_arr))
            print("    attribute names:", names[:5], "...")
            
            if names and y_arr is not None:
                # Convert to DataFrame
                X_df = pd.DataFrame(X_arr, columns=names)
                y_series = pd.Series(y_arr, name=dataset.default_target_attribute)
                print("    X DataFrame shape:", X_df.shape)
                print("    y Series shape:", y_series.shape)
                print("    First few y values:", y_series.head())
            else:
                print("    Cannot convert to DataFrame!")
        except Exception as e:
            print("    Error:", str(e))
            
        # Method 5: Check raw dataset dictionary
        print("\n  Method 5: Check raw dataset dictionary...")
        try:
            data_dict = dataset.get_data_dictionary()
            if data_dict:
                print("    Data dictionary keys:", list(data_dict.keys()) if data_dict else "None")
                if 'target' in data_dict:
                    print("    Target in data dictionary:", data_dict['target'])
            else:
                print("    Data dictionary is None!")
        except Exception as e:
            print("    Error:", str(e))
    
    except Exception as e:
        print("  ✗ Failed to access dataset 61:", str(e))
        
except ImportError as e:
    print("Failed to import OpenML:", str(e))
    print("Please install OpenML: pip install openml")
except Exception as e:
    print("Unexpected error:", str(e))

print("\nTest completed.") 