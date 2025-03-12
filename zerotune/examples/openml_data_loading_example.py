"""
OpenML Dataset Loading Example.

This example demonstrates different methods for loading datasets from OpenML
and handling common issues that may arise when working with OpenML data.

The script shows five different approaches to loading the Iris dataset:
1. Standard get_data() without target specification
2. get_data() with explicit target attribute
3. Loading as a DataFrame and extracting the target
4. Loading as an array and manually converting to DataFrame/Series
5. Checking the raw dataset dictionary

This is useful for troubleshooting OpenML data loading issues in ZeroTune.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Try to import OpenML and handle import errors gracefully
try:
    import openml
    print("OpenML version:", openml.__version__)
except ImportError as e:
    print("Failed to import OpenML:", str(e))
    print("Please install OpenML: pip install openml")
    print("You can use: poetry add openml")
    sys.exit(1)

print("Python version:", sys.version)

# Test the Iris dataset (ID: 61) with different methods
print("\nTesting Iris dataset (ID: 61) with different methods...")
try:
    # First, get the dataset metadata
    dataset = openml.datasets.get_dataset(61)
    print(f"Dataset name: {dataset.name}")
    print(f"Default target attribute: {dataset.default_target_attribute}")
    print(f"Dataset features: {len(dataset.features)}")
    
    print("\n" + "="*70)
    print("METHOD 1: Standard get_data() without specifying target")
    print("="*70)
    try:
        X, y, categorical, names = dataset.get_data()
        print(f"X type: {type(X)}")
        print(f"y type: {type(y)}")
        
        if y is not None:
            print(f"First few y values: {y[:5]}")
        else:
            print("ISSUE: y is None! This occurs when the target isn't specified")
            print("SOLUTION: Use Method 2 (specify target attribute)")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    print("\n" + "="*70)
    print("METHOD 2: Specify target attribute explicitly")
    print("="*70)
    try:
        X, y, categorical, names = dataset.get_data(target=dataset.default_target_attribute)
        print(f"X type: {type(X)}")
        print(f"y type: {type(y)}")
        
        if y is not None:
            print(f"First few y values: {y[:5]}")
            print("SUCCESS: Target attribute was correctly extracted")
        else:
            print("ISSUE: y is still None! Check if the target attribute name is correct")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    print("\n" + "="*70)
    print("METHOD 3: Get as DataFrame using dataset_format parameter")
    print("="*70)
    try:
        df, _, _, _ = dataset.get_data(dataset_format="dataframe")
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame columns: {df.columns.tolist()}")
        
        # Extract target from DataFrame
        if dataset.default_target_attribute in df.columns:
            y = df[dataset.default_target_attribute]
            X = df.drop(dataset.default_target_attribute, axis=1)
            print(f"X shape: {X.shape}")
            print(f"y shape: {y.shape}")
            print(f"First few y values: {y.head()}")
            print("SUCCESS: Loaded as DataFrame and separated target column")
        else:
            print(f"ISSUE: Target column '{dataset.default_target_attribute}' not found in DataFrame!")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("NOTE: This method requires OpenML 0.10.0 or newer")
    
    print("\n" + "="*70)
    print("METHOD 4: Get as array and convert manually to DataFrame")
    print("="*70)
    try:
        X_arr, y_arr, categorical, names = dataset.get_data(dataset_format="array")
        print(f"X array shape: {X_arr.shape if hasattr(X_arr, 'shape') else 'no shape'}")
        print(f"y array type: {type(y_arr)}")
        print(f"Feature names: {names[:5]} ...")
        
        if names and y_arr is not None:
            # Convert to DataFrame and Series
            X_df = pd.DataFrame(X_arr, columns=names)
            y_series = pd.Series(y_arr, name=dataset.default_target_attribute)
            print(f"X DataFrame shape: {X_df.shape}")
            print(f"y Series shape: {y_series.shape}")
            print(f"First few y values: {y_series.head()}")
            print("SUCCESS: Converted array data to DataFrame and Series")
        else:
            print("ISSUE: Cannot convert to DataFrame/Series!")
            if y_arr is None:
                print("SOLUTION: Specify the target attribute")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    print("\n" + "="*70)
    print("METHOD 5: Check raw dataset dictionary")
    print("="*70)
    try:
        try:
            data_dict = dataset.get_data_dictionary()
            if data_dict:
                print(f"Data dictionary keys: {list(data_dict.keys()) if data_dict else 'None'}")
                if 'target' in data_dict:
                    print(f"Target in data dictionary: {data_dict['target']}")
                print("SUCCESS: Retrieved raw dataset dictionary")
            else:
                print("ISSUE: Data dictionary is None!")
        except AttributeError:
            print("ISSUE: get_data_dictionary() method not found")
            print("SOLUTION: This method may not be available in your OpenML version")
    except Exception as e:
        print(f"Error: {str(e)}")

except Exception as e:
    print(f"Failed to access dataset 61: {str(e)}")

print("\n" + "="*70)
print("SUMMARY AND RECOMMENDATIONS")
print("="*70)
print("Based on testing with the Iris dataset:")
print("1. Method 2 (specify target) is the most reliable way to load data")
print("2. Method 3 (DataFrame format) is convenient but requires newer OpenML versions")
print("3. When using with ZeroTune, ensure you specify the target attribute name")
print("4. For categorical targets, ZeroTune will convert them to integers using LabelEncoder")
print("\nTest completed.") 