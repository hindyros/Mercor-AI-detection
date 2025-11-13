"""
Feature Engineering Functions

Extracted from heuristic_model.ipynb for reuse across notebooks.
Supports both meta features and embeddings.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Optional, List
from .embeddings import EmbeddingExtractor


def extract_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract comprehensive text-based features from the answer column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'answer' column
        
    Returns:
    -------
    pd.DataFrame
        DataFrame with additional feature columns
    """
    df = df.copy()
    
    # Basic text statistics
    df['text_length'] = df['answer'].str.len()
    df['word_count'] = df['answer'].str.split().str.len()
    df['char_count_no_spaces'] = df['answer'].str.replace(' ', '').str.len()
    df['sentence_count'] = df['answer'].str.count(r'[.!?]+')
    df['paragraph_count'] = df['answer'].str.count('\n\n') + 1
    
    # Average word length
    df['avg_word_length'] = df['char_count_no_spaces'] / (df['word_count'] + 1e-6)
    
    # Average sentence length
    df['avg_sentence_length'] = df['word_count'] / (df['sentence_count'] + 1e-6)
    
    # Punctuation features (escaping special regex characters)
    df['exclamation_count'] = df['answer'].str.count('!')
    df['question_count'] = df['answer'].str.count(r'\?')
    df['comma_count'] = df['answer'].str.count(',')
    df['period_count'] = df['answer'].str.count(r'\.')
    df['punctuation_ratio'] = (df['exclamation_count'] + df['question_count'] + df['period_count']) / (df['text_length'] + 1e-6)
    
    # Capitalization features
    df['uppercase_count'] = df['answer'].str.findall(r'[A-Z]').str.len()
    df['uppercase_ratio'] = df['uppercase_count'] / (df['text_length'] + 1e-6)
    
    # Special characters
    df['digit_count'] = df['answer'].str.count(r'\d')
    df['special_char_count'] = df['answer'].str.count(r'[^\w\s]')
    
    # Word complexity (long words)
    words = df['answer'].str.split()
    df['long_word_count'] = words.apply(lambda x: sum(1 for w in x if len(w) > 6))
    df['long_word_ratio'] = df['long_word_count'] / (df['word_count'] + 1e-6)
    
    # Common AI indicators (case-insensitive using findall)
    df['first_person_pronouns'] = df['answer'].str.findall(r'(?i)\b(I|me|my|myself|we|us|our|ourselves)\b').str.len()
    df['first_person_ratio'] = df['first_person_pronouns'] / (df['word_count'] + 1e-6)
    
    # Whitespace features
    df['whitespace_count'] = df['answer'].str.count(' ')
    df['whitespace_ratio'] = df['whitespace_count'] / (df['text_length'] + 1e-6)
    
    # Unique word ratio (vocabulary diversity)
    df['unique_word_count'] = words.apply(lambda x: len(set(w.lower() for w in x)) if x else 0)
    df['unique_word_ratio'] = df['unique_word_count'] / (df['word_count'] + 1e-6)
    
    return df


def encode_topic(df: pd.DataFrame, le: Optional[LabelEncoder] = None, fit: bool = True) -> Tuple[pd.DataFrame, LabelEncoder]:
    """
    Encode topic column using LabelEncoder.
    Handles unseen topics in test set by mapping them to -1.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'topic' column
    le : LabelEncoder, optional
        Pre-fitted LabelEncoder. If None and fit=True, creates new one
    fit : bool
        Whether to fit the encoder (True for training, False for test)
        
    Returns:
    -------
    pd.DataFrame, LabelEncoder
        DataFrame with encoded topic and the encoder
    """
    df = df.copy()
    
    if le is None:
        le = LabelEncoder()
    
    if fit:
        df['topic_encoded'] = le.fit_transform(df['topic'])
    else:
        # Handle unseen topics by mapping them to -1
        known_topics = set(le.classes_)
        df['topic_encoded'] = df['topic'].apply(
            lambda x: le.transform([x])[0] if x in known_topics else -1
        )
    
    return df, le


def prepare_features(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    feature_cols: Optional[List[str]] = None,
    include_embeddings: bool = False,
    embedding_model: Optional[str] = None,
    embedding_extractor: Optional[EmbeddingExtractor] = None,
    batch_size: int = 8
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], List[str], LabelEncoder, Optional[dict]]:
    """
    Prepare features for training and testing.
    Can include both meta features and embeddings.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training dataframe
    test_df : pd.DataFrame, optional
        Test dataframe
    feature_cols : List[str], optional
        Pre-defined feature columns (for consistency with test set)
    include_embeddings : bool
        Whether to include text embeddings
    embedding_model : str, optional
        HuggingFace model name for embeddings (if include_embeddings=True)
    embedding_extractor : EmbeddingExtractor, optional
        Pre-initialized embedding extractor (reuses if provided)
    batch_size : int
        Batch size for embedding extraction
        
    Returns:
    -------
    tuple
        (X_train, y_train, X_test, feature_names, topic_encoder, embedding_info)
        embedding_info contains embedding_dim and model_name if embeddings were used
    """
    # Extract text features (meta features)
    train_processed = extract_text_features(train_df)
    
    # Encode topics
    train_processed, topic_encoder = encode_topic(train_processed, fit=True)
    
    # Select feature columns (exclude id, topic, answer, is_cheating)
    if feature_cols is None:
        feature_cols = [col for col in train_processed.columns 
                       if col not in ['id', 'topic', 'answer', 'is_cheating']]
    
    X_train_meta = train_processed[feature_cols].values
    y_train = train_processed['is_cheating'].values
    
    # Extract embeddings if requested
    embedding_info = None
    X_train_embeddings = None
    X_test_embeddings = None
    
    if include_embeddings:
        # Initialize or reuse embedding extractor
        if embedding_extractor is None:
            if embedding_model is None:
                embedding_model = "nvidia/llama-embed-nemotron-8b"
            embedding_extractor = EmbeddingExtractor(
                model_name_or_path=embedding_model,
                batch_size=batch_size
            )
        
        print("Extracting embeddings for training set...")
        X_train_embeddings = embedding_extractor.extract_embeddings_from_dataframe(
            train_df, text_column='answer'
        )
        
        if test_df is not None:
            print("Extracting embeddings for test set...")
            X_test_embeddings = embedding_extractor.extract_embeddings_from_dataframe(
                test_df, text_column='answer'
            )
        
        embedding_info = {
            'embedding_dim': X_train_embeddings.shape[1],
            'model_name': embedding_model or embedding_extractor.model_name_or_path
        }
    
    # Combine meta features and embeddings
    if include_embeddings and X_train_embeddings is not None:
        X_train = np.hstack([X_train_meta, X_train_embeddings])
        feature_names = feature_cols + [f'embedding_{i}' for i in range(X_train_embeddings.shape[1])]
    else:
        X_train = X_train_meta
        feature_names = feature_cols.copy()
    
    # Process test set
    X_test = None
    if test_df is not None:
        test_processed = extract_text_features(test_df)
        test_processed, _ = encode_topic(test_processed, le=topic_encoder, fit=False)
        X_test_meta = test_processed[feature_cols].values
        
        if include_embeddings and X_test_embeddings is not None:
            X_test = np.hstack([X_test_meta, X_test_embeddings])
        else:
            X_test = X_test_meta
    
    return X_train, y_train, X_test, feature_names, topic_encoder, embedding_info

