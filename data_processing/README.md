# Data Processing Module

This module contains functions for feature engineering and creating different versions of processed datasets.

## Quick Start

### 1. Create a Dataset

```python
import pandas as pd
from data_processing import create_dataset

# Load raw data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Create baseline dataset
create_dataset(
    train_df=train,
    test_df=test,
    dataset_name='baseline',
    description='Baseline features: all meta features + topic encoding'
)
```

### 2. Load a Dataset in Your Notebook

```python
from data_processing import load_dataset

# Load the dataset
X_train, y_train, X_test, info = load_dataset('baseline')

print(f"Features: {info['n_features']}")
print(f"Train samples: {info['n_train_samples']}")
print(f"Feature names: {info['feature_names']}")
```

### 3. List Available Datasets

```python
from data_processing import list_available_datasets

datasets = list_available_datasets()
print(f"Available datasets: {datasets}")
```

## Creating Different Dataset Versions

You can create multiple versions of datasets for experimentation:

```python
# Version 1: Baseline (all features)
create_dataset(train, test, 'baseline', 'All meta features + topic')

# Version 2: Without topic encoding
from data_processing.feature_engineering import extract_text_features
train_no_topic = extract_text_features(train)
test_no_topic = extract_text_features(test)
feature_cols_no_topic = [col for col in train_no_topic.columns 
                        if col not in ['id', 'topic', 'answer', 'is_cheating']]
create_dataset(train, test, 'no_topic', 'Meta features without topic encoding',
               feature_cols=feature_cols_no_topic)

# Version 3: Only text statistics
feature_cols_stats = ['text_length', 'word_count', 'char_count_no_spaces', 
                      'sentence_count', 'paragraph_count']
create_dataset(train, test, 'text_stats_only', 'Only basic text statistics',
               feature_cols=feature_cols_stats)
```

## Module Structure

- `feature_engineering.py`: Core feature extraction functions
  - `extract_text_features()`: Extract meta features from text
  - `encode_topic()`: Encode topic column
  - `prepare_features()`: Complete feature preparation pipeline

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

## Example Workflow

```python
# In a preprocessing notebook or script
import pandas as pd
from data_processing import create_dataset

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Create different versions
create_dataset(train, test, 'baseline', 'Full feature set')
create_dataset(train, test, 'minimal', 'Minimal features only', 
               feature_cols=['text_length', 'word_count', 'unique_word_ratio'])

# In your experiment notebook
from data_processing import load_dataset

X_train, y_train, X_test, info = load_dataset('baseline')
# Now use X_train, y_train, X_test for your models
```

