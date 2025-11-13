"""
Feature Engineering Functions

Extracted from heuristic_model.ipynb for reuse across notebooks.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Optional, List


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
    feature_cols: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], List[str], LabelEncoder]:
    """
    Prepare features for training and testing.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training dataframe
    test_df : pd.DataFrame, optional
        Test dataframe
    feature_cols : List[str], optional
        Pre-defined feature columns (for consistency with test set)
        
    Returns:
    -------
    tuple
        (X_train, y_train, X_test, feature_names, topic_encoder)
    """
    # Extract text features
    train_processed = extract_text_features(train_df)
    
    # Encode topics
    train_processed, topic_encoder = encode_topic(train_processed, fit=True)
    
    # Select feature columns (exclude id, topic, answer, is_cheating)
    if feature_cols is None:
        feature_cols = [col for col in train_processed.columns 
                       if col not in ['id', 'topic', 'answer', 'is_cheating']]
    
    X_train = train_processed[feature_cols].values
    y_train = train_processed['is_cheating'].values
    
    X_test = None
    if test_df is not None:
        test_processed = extract_text_features(test_df)
        test_processed, _ = encode_topic(test_processed, le=topic_encoder, fit=False)
        X_test = test_processed[feature_cols].values
    
    return X_train, y_train, X_test, feature_cols, topic_encoder

