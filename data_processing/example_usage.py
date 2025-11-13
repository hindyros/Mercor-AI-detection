"""
Example: How to use the data_processing module

This script demonstrates how to create and use different dataset versions.
Run this once to create your datasets, then import them in notebooks.
"""

import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_processing import create_dataset, list_available_datasets


def main():
    # Load raw data
    print("Loading raw data...")
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}\n")
    
    # Create baseline dataset (all features)
    print("=" * 60)
    print("Creating baseline dataset...")
    print("=" * 60)
    create_dataset(
        train_df=train,
        test_df=test,
        dataset_name='baseline',
        description='Baseline features: all meta features + topic encoding'
    )
    
    # List available datasets
    print("\n" + "=" * 60)
    print("Available datasets:")
    print("=" * 60)
    datasets = list_available_datasets()
    for ds in datasets:
        print(f"  - {ds}")
    
    print("\n✓ Done! You can now load datasets in your notebooks using:")
    print("  from data_processing import load_dataset")
    print("  X_train, y_train, X_test, info = load_dataset('baseline')")


if __name__ == "__main__":
    main()

