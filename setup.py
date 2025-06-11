#!/usr/bin/env python
"""
Setup script for ZeroTune - One-shot hyperparameter optimization using meta-learning.
This file is provided for those who prefer to use pip instead of Poetry.
"""

import os
from setuptools import setup, find_packages

# Read README.md for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="zerotune",
    version="0.1.0",
    description="One-shot hyperparameter optimization using meta-learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Tarek",
    author_email="tareksalhi0@gmail.com",
    url="https://github.com/Tarek0/zerotune",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "xgboost>=1.7.0",
        "openml>=0.14.0",
        "joblib>=1.2.0",
        "optuna>=3.2.0",
    ],
    entry_points={
        "console_scripts": [
            "zerotune=zerotune.__main__:main",
        ],
    },
    python_requires=">=3.8,<3.12",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
) 