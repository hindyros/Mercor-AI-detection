"""
Example: Create dataset with embeddings

This script demonstrates how to create a dataset with both meta features and embeddings.
"""

import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_processing import create_dataset


def main():
    # Load raw data
    print("Loading raw data...")
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}\n")
    
    # Create dataset WITH embeddings
    print("=" * 60)
    print("Creating dataset with embeddings...")
    print("=" * 60)
    print("This will take some time as embeddings need to be extracted...")
    
    create_dataset(
        train_df=train,
        test_df=test,
        dataset_name='with_embeddings',
        description='Meta features + llama-embed-nemotron-8b embeddings',
        include_embeddings=True,
        embedding_model="nvidia/llama-embed-nemotron-8b",
        batch_size=8  # Adjust based on your GPU memory
    )
    
    print("\n✓ Done! You can now load the dataset in your notebooks:")
    print("  from data_processing import load_dataset")
    print("  X_train, y_train, X_test, info = load_dataset('with_embeddings')")


if __name__ == "__main__":
    main()

