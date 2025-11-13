"""
Dataset Creator Module

Functions to create, save, and load different versions of processed datasets.
"""

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from .feature_engineering import prepare_features


# Directory to store processed datasets
PROCESSED_DATA_DIR = Path(__file__).parent.parent / "processed_data"
PROCESSED_DATA_DIR.mkdir(exist_ok=True)


def create_dataset(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    dataset_name: str = "baseline",
    description: str = "",
    feature_cols: Optional[list] = None,
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    Create and save a processed dataset.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training dataframe
    test_df : pd.DataFrame, optional
        Test dataframe
    dataset_name : str
        Name for this dataset version (e.g., 'baseline', 'v1', 'no_topic', etc.)
    description : str
        Description of what makes this dataset version unique
    feature_cols : list, optional
        Specific feature columns to use (if None, uses all features)
    overwrite : bool
        Whether to overwrite if dataset already exists
        
    Returns:
    -------
    dict
        Dictionary containing dataset info and paths
    """
    dataset_path = PROCESSED_DATA_DIR / dataset_name
    dataset_path.mkdir(exist_ok=True)
    
    # Check if dataset already exists
    info_file = dataset_path / "dataset_info.pkl"
    if info_file.exists() and not overwrite:
        raise ValueError(f"Dataset '{dataset_name}' already exists. Use overwrite=True to replace it.")
    
    # Prepare features
    X_train, y_train, X_test, feature_names, topic_encoder = prepare_features(
        train_df, test_df, feature_cols=feature_cols
    )
    
    # Save dataset
    np.save(dataset_path / "X_train.npy", X_train)
    np.save(dataset_path / "y_train.npy", y_train)
    
    if X_test is not None:
        np.save(dataset_path / "X_test.npy", X_test)
    
    # Save metadata
    dataset_info = {
        'dataset_name': dataset_name,
        'description': description,
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'n_train_samples': len(y_train),
        'n_test_samples': len(X_test) if X_test is not None else 0,
        'feature_cols': feature_cols,
        'topic_encoder': topic_encoder,
        'train_ids': train_df['id'].values if 'id' in train_df.columns else None,
        'test_ids': test_df['id'].values if test_df is not None and 'id' in test_df.columns else None,
    }
    
    with open(info_file, 'wb') as f:
        pickle.dump(dataset_info, f)
    
    print(f"✓ Dataset '{dataset_name}' created successfully!")
    print(f"  Description: {description}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Train samples: {len(y_train)}")
    if X_test is not None:
        print(f"  Test samples: {len(X_test)}")
    print(f"  Saved to: {dataset_path}")
    
    return dataset_info


def load_dataset(dataset_name: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """
    Load a processed dataset.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset to load
        
    Returns:
    -------
    tuple
        (X_train, y_train, X_test, dataset_info)
    """
    dataset_path = PROCESSED_DATA_DIR / dataset_name
    
    if not dataset_path.exists():
        available = list_available_datasets()
        raise ValueError(
            f"Dataset '{dataset_name}' not found.\n"
            f"Available datasets: {available}"
        )
    
    # Load data
    X_train = np.load(dataset_path / "X_train.npy")
    y_train = np.load(dataset_path / "y_train.npy")
    
    X_test = None
    test_file = dataset_path / "X_test.npy"
    if test_file.exists():
        X_test = np.load(test_file)
    
    # Load metadata
    info_file = dataset_path / "dataset_info.pkl"
    with open(info_file, 'rb') as f:
        dataset_info = pickle.load(f)
    
    return X_train, y_train, X_test, dataset_info


def list_available_datasets() -> list:
    """
    List all available processed datasets.
    
    Returns:
    -------
    list
        List of dataset names
    """
    if not PROCESSED_DATA_DIR.exists():
        return []
    
    datasets = []
    for item in PROCESSED_DATA_DIR.iterdir():
        if item.is_dir() and (item / "dataset_info.pkl").exists():
            datasets.append(item.name)
    
    return sorted(datasets)


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """
    Get information about a dataset without loading the full data.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset
        
    Returns:
    -------
    dict
        Dataset information
    """
    dataset_path = PROCESSED_DATA_DIR / dataset_name
    info_file = dataset_path / "dataset_info.pkl"
    
    if not info_file.exists():
        raise ValueError(f"Dataset '{dataset_name}' not found.")
    
    with open(info_file, 'rb') as f:
        return pickle.load(f)


def delete_dataset(dataset_name: str) -> None:
    """
    Delete a processed dataset.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset to delete
    """
    dataset_path = PROCESSED_DATA_DIR / dataset_name
    
    if not dataset_path.exists():
        raise ValueError(f"Dataset '{dataset_name}' not found.")
    
    import shutil
    shutil.rmtree(dataset_path)
    print(f"✓ Dataset '{dataset_name}' deleted.")

