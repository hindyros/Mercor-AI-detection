"""
Data Processing Module

This module contains functions for feature engineering and dataset creation.
Use this to create different versions of processed datasets for experimentation.
"""

from .feature_engineering import (
    extract_text_features,
    encode_topic,
    prepare_features
)

from .dataset_creator import (
    create_dataset,
    load_dataset,
    list_available_datasets
)

__all__ = [
    'extract_text_features',
    'encode_topic',
    'prepare_features',
    'create_dataset',
    'load_dataset',
    'list_available_datasets'
]

