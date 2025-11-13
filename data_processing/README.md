# Data Processing Module

This module contains functions for feature engineering and creating different versions of processed datasets.

## Features

- **Meta Features**: Heuristic features extracted from text (length, word count, punctuation, etc.)
- **Embeddings**: Text embeddings using llama-embed-nemotron-8b or other models
- **Combined**: Both meta features + embeddings for multimodal learning

## Quick Start

### 1. Create a Dataset (Meta Features Only)

```python
import pandas as pd
from data_processing import create_dataset

# Load raw data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Create baseline dataset (meta features only)
create_dataset(
    train_df=train,
    test_df=test,
    dataset_name='baseline',
    description='Baseline features: all meta features + topic encoding'
)
```

### 2. Create a Dataset WITH Embeddings

```python
from data_processing import create_dataset

# Create dataset with embeddings
create_dataset(
    train_df=train,
    test_df=test,
    dataset_name='with_embeddings',
    description='Meta features + llama-embed-nemotron-8b embeddings',
    include_embeddings=True,
    embedding_model="nvidia/llama-embed-nemotron-8b",
    batch_size=8  # Adjust based on GPU memory
)
```

### 3. Load a Dataset in Your Notebook

```python
from data_processing import load_dataset

# Load the dataset
X_train, y_train, X_test, info = load_dataset('with_embeddings')

print(f"Features: {info['n_features']}")
print(f"Train samples: {info['n_train_samples']}")
print(f"Has embeddings: {info.get('include_embeddings', False)}")
if info.get('embedding_info'):
    print(f"Embedding dim: {info['embedding_info']['embedding_dim']}")
```

## Creating Different Dataset Versions

You can create multiple versions for experimentation:

```python
# Version 1: Baseline (meta features only)
create_dataset(train, test, 'baseline', 'All meta features + topic')

# Version 2: With embeddings
create_dataset(train, test, 'with_embeddings', 
               'Meta features + embeddings',
               include_embeddings=True)

# Version 3: Embeddings only (no meta features)
# You can modify feature_cols to exclude meta features
```

## Direct Usage in Notebooks

You can also use the functions directly without creating datasets:

```python
from data_processing import prepare_features

# Get features with embeddings
X_train, y_train, X_test, feature_names, topic_encoder, embedding_info = prepare_features(
    train_df=train,
    test_df=test,
    include_embeddings=True,
    embedding_model="nvidia/llama-embed-nemotron-8b"
)

# Now use X_train, y_train for training
```

## Module Structure

- `feature_engineering.py`: Core feature extraction functions
  - `extract_text_features()`: Extract meta features from text
  - `encode_topic()`: Encode topic column
  - `prepare_features()`: Complete feature preparation (supports embeddings)

- `embeddings.py`: Text embedding extraction
  - `EmbeddingExtractor`: Class for extracting embeddings
  - `extract_embeddings_simple()`: Simple function interface
  - `average_pool()`: Pooling function for embeddings

- `dataset_creator.py`: Dataset creation and management
  - `create_dataset()`: Create and save a processed dataset
  - `load_dataset()`: Load a processed dataset
  - `list_available_datasets()`: List all available datasets
  - `get_dataset_info()`: Get dataset metadata without loading
  - `delete_dataset()`: Delete a dataset

## Benefits

1. **Reusability**: Process data once, use in multiple notebooks
2. **Version Control**: Create different dataset versions for experiments
3. **Reproducibility**: Same dataset version = same features
4. **Efficiency**: No need to recompute features every time
5. **Organization**: All processed datasets in one place (`processed_data/`)
6. **Multimodal**: Support for combining meta features + embeddings

## Example Workflow

```python
# Step 1: Create datasets (run once)
from data_processing import create_dataset
import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Create baseline
create_dataset(train, test, 'baseline', 'Meta features only')

# Create with embeddings (takes longer)
create_dataset(train, test, 'with_embeddings', 
               'Meta + embeddings',
               include_embeddings=True)

# Step 2: Use in your experiment notebook
from data_processing import load_dataset

# Load baseline
X_train_base, y_train, X_test_base, info_base = load_dataset('baseline')

# Load with embeddings
X_train_full, y_train, X_test_full, info_full = load_dataset('with_embeddings')

# Compare models trained on different feature sets
```

## Notes

- Embeddings extraction requires GPU for reasonable speed
- The llama-embed-nemotron-8b model is large (~16GB), make sure you have enough memory
- Batch size can be adjusted based on your GPU memory
- Embeddings are cached in the dataset, so you only need to extract them once
