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

from .embeddings import (
    EmbeddingExtractor,
    extract_embeddings_simple,
    average_pool
)

from .dataset_creator import (
    create_dataset,
    load_dataset,
    list_available_datasets
)

try:
    from .subset_selection import (
        select_hardest_subset,
        identify_hard_samples,
        apply_subset_selection,
        trimmed_loss_subset_selection_gurobi,
        trimmed_loss_subset_selection_scipy
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import subset_selection functions: {e}")
    # Define placeholders to avoid import errors
    select_hardest_subset = None
    identify_hard_samples = None
    apply_subset_selection = None
    trimmed_loss_subset_selection_gurobi = None
    trimmed_loss_subset_selection_scipy = None

try:
    from .benchmarking import (
        load_hc3,
        evaluate_on_benchmark,
        benchmark_model
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import benchmarking functions: {e}")
    load_hc3 = None
    evaluate_on_benchmark = None
    benchmark_model = None

__all__ = [
    'extract_text_features',
    'encode_topic',
    'prepare_features',
    'EmbeddingExtractor',
    'extract_embeddings_simple',
    'average_pool',
    'create_dataset',
    'load_dataset',
    'list_available_datasets',
    'select_hardest_subset',
    'identify_hard_samples',
    'apply_subset_selection',
    'trimmed_loss_subset_selection_gurobi',
    'trimmed_loss_subset_selection_scipy',
    'load_hc3',
    'evaluate_on_benchmark',
    'benchmark_model'
]

