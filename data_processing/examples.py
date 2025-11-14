"""
Examples: How to use the data_processing module

This script demonstrates how to create and use different dataset versions.
Run this once to create your datasets, then import them in notebooks.

Usage:
    python -m data_processing.examples
"""

import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_processing import create_dataset, list_available_datasets


def create_baseline_dataset():
    """Example: Create a baseline dataset with meta features only."""
    print("=" * 60)
    print("EXAMPLE 1: Creating Baseline Dataset (Meta Features Only)")
    print("=" * 60)
    
    # Load raw data from data/raw folder
    print("\nLoading raw data...")
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    train = pd.read_csv(data_dir / 'train.csv')
    test = pd.read_csv(data_dir / 'test.csv')
    
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}\n")
    
    # Create baseline dataset (all features)
    print("Creating baseline dataset...")
    create_dataset(
        train_df=train,
        test_df=test,
        dataset_name='baseline',
        description='Baseline features: all meta features + topic encoding'
    )
    
    print("\n✓ Baseline dataset created!")
    print("  Load it with: load_dataset('baseline')")


def create_dataset_with_embeddings():
    """Example: Create a dataset with both meta features and embeddings."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Creating Dataset with Embeddings")
    print("=" * 60)
    
    # Load raw data from data/raw folder
    print("\nLoading raw data...")
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    train = pd.read_csv(data_dir / 'train.csv')
    test = pd.read_csv(data_dir / 'test.csv')
    
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}\n")
    
    # Create dataset WITH embeddings
    print("Creating dataset with embeddings...")
    print("⚠️  This will take some time as embeddings need to be extracted...")
    
    create_dataset(
        train_df=train,
        test_df=test,
        dataset_name='with_embeddings',
        description='Meta features + llama-embed-nemotron-8b embeddings',
        include_embeddings=True,
        embedding_model="nvidia/llama-embed-nemotron-8b",
        batch_size=8  # Adjust based on your GPU memory
    )
    
    print("\n✓ Dataset with embeddings created!")
    print("  Load it with: load_dataset('with_embeddings')")


def list_datasets():
    """Example: List all available datasets."""
    print("\n" + "=" * 60)
    print("Available Datasets")
    print("=" * 60)
    datasets = list_available_datasets()
    if datasets:
        for ds in datasets:
            print(f"  - {ds}")
    else:
        print("  No datasets found. Create one using create_dataset()")
    
    print("\n✓ Done! You can now load datasets in your notebooks using:")
    print("  from data_processing import load_dataset")
    print("  X_train, y_train, X_test, info = load_dataset('baseline')")


def main():
    """Main function to run examples."""
    print("\n" + "=" * 60)
    print("DATA PROCESSING MODULE - EXAMPLES")
    print("=" * 60)
    print("\nThis script demonstrates how to create different dataset versions.")
    print("You can run individual examples or all of them.\n")
    
    # Example 1: Baseline dataset
    create_baseline_dataset()
    
    # Example 2: Dataset with embeddings (commented out by default as it takes time)
    # Uncomment to run:
    # create_dataset_with_embeddings()
    
    # List all available datasets
    list_datasets()


if __name__ == "__main__":
    main()

