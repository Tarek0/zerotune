"""
ZeroTune Core Package.

This package provides the core functionality for ZeroTune, a system for
automatic hyperparameter optimization based on meta-learning.

The ZeroTune class is the main entry point for using the system.
"""

from zerotune.core.zero_tune import ZeroTune
from zerotune.core.model_configs import ModelConfigs
from zerotune.core.data_loading import (
    fetch_open_ml_data,
    prepare_data,
    load_dataset_catalog,
    get_dataset_ids,
    get_recommended_datasets
)
from zerotune.core.config import CONFIG
from zerotune.core.utils import (
    safe_json_serialize,
    load_json,
    save_json,
    is_numeric_dtype,
    select_numeric_columns,
    is_classification_task,
    make_base_params_dict,
    convert_to_dataframe
)

__all__ = [
    'ZeroTune',
    'ModelConfigs',
    'CONFIG',
    'fetch_open_ml_data',
    'prepare_data',
    'load_dataset_catalog',
    'get_dataset_ids',
    'get_recommended_datasets',
    'safe_json_serialize',
    'load_json',
    'save_json',
    'is_numeric_dtype',
    'select_numeric_columns',
    'is_classification_task',
    'make_base_params_dict',
    'convert_to_dataframe'
] 