"""
Model Experiment Tracking System

Automatically logs all model training results to a centralized CSV file.
Each model training creates a new row with complete experiment details.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any
import json


# Path to the centralized results CSV
RESULTS_CSV = Path(__file__).parent.parent / "model_results.csv"


def log_model_experiment(
    model_name: str,
    notebook_name: str,
    model_type: str,
    features: str,
    data_processing: Dict[str, Any],
    training_config: Dict[str, Any],
    validation_metrics: Dict[str, float],
    test_metrics: Optional[Dict[str, float]] = None,
    hc3_benchmark: Optional[Dict[str, float]] = None,
    notes: str = "",
    **kwargs
) -> None:
    """
    Log a model training experiment to the centralized results CSV.
    
    Parameters:
    -----------
    model_name : str
        Name/identifier for this model (e.g., "XGBoost_v1", "RoBERTa_finetuned")
    notebook_name : str
        Name of the notebook where model was trained (e.g., "heuristic_model", "TransferLearning")
    model_type : str
        Type of model (e.g., "XGBoost", "RandomForest", "RoBERTa", "NeuralNetwork")
    features : str
        Description of features used (e.g., "Meta features (25)", "Meta + Embeddings (4121)")
    data_processing : dict
        Dictionary describing data processing pipeline:
        - train_size, val_size, test_size
        - split_method (e.g., "stratified_80_20", "5_fold_cv")
        - preprocessing_steps (list of steps)
        - feature_engineering (description)
        - upsampling (bool or description)
    training_config : dict
        Dictionary with training hyperparameters:
        - learning_rate, batch_size, epochs, etc.
        - regularization (dropout, weight_decay, etc.)
        - optimizer, scheduler
    validation_metrics : dict
        Dictionary with validation set metrics:
        - roc_auc, accuracy, precision, recall, f1
        - Can include other metrics
    test_metrics : dict, optional
        Dictionary with test set metrics (if evaluated)
    hc3_benchmark : dict, optional
        Dictionary with HC3 benchmark metrics (if evaluated)
    notes : str
        Additional notes about the experiment
    **kwargs
        Any additional fields to include
    """
    
    # Create results directory if it doesn't exist
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare experiment record - only include fields with actual values
    experiment = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_name': model_name,
        'notebook_name': notebook_name,
        'model_type': model_type,
        'features': features,
    }
    
    # Data processing - only add non-None values
    if data_processing.get('train_size') is not None:
        experiment['train_size'] = data_processing['train_size']
    if data_processing.get('val_size') is not None:
        experiment['val_size'] = data_processing['val_size']
    if data_processing.get('test_size') is not None:
        experiment['test_size'] = data_processing['test_size']
    if data_processing.get('split_method'):
        experiment['split_method'] = data_processing['split_method']
    if data_processing.get('preprocessing_steps'):
        preprocessing = data_processing['preprocessing_steps']
        experiment['preprocessing'] = json.dumps(preprocessing) if isinstance(preprocessing, list) else str(preprocessing)
    if data_processing.get('feature_engineering'):
        experiment['feature_engineering'] = data_processing['feature_engineering']
    if data_processing.get('upsampling') is not False:
        experiment['upsampling'] = data_processing.get('upsampling', False)
    
    # Training config - only add non-None values
    if training_config.get('learning_rate') is not None:
        experiment['learning_rate'] = training_config['learning_rate']
    if training_config.get('batch_size') is not None:
        experiment['batch_size'] = training_config['batch_size']
    if training_config.get('epochs') is not None:
        experiment['epochs'] = training_config['epochs']
    if training_config.get('dropout') is not None:
        experiment['dropout'] = training_config['dropout']
    if training_config.get('weight_decay') is not None:
        experiment['weight_decay'] = training_config['weight_decay']
    if training_config.get('optimizer'):
        experiment['optimizer'] = training_config['optimizer']
    if training_config.get('scheduler'):
        experiment['scheduler'] = training_config['scheduler']
    if training_config.get('early_stopping') is not False:
        experiment['early_stopping'] = training_config.get('early_stopping', False)
    
    # Store full training config as JSON for reference
    experiment['training_config_json'] = json.dumps(training_config)
    
    # Validation metrics - always include
    experiment['val_roc_auc'] = validation_metrics.get('roc_auc')
    experiment['val_accuracy'] = validation_metrics.get('accuracy')
    experiment['val_precision'] = validation_metrics.get('precision')
    experiment['val_recall'] = validation_metrics.get('recall')
    experiment['val_f1'] = validation_metrics.get('f1')
    if validation_metrics.get('loss') is not None:
        experiment['val_loss'] = validation_metrics['loss']
    
    # Test metrics - only add if available
    if test_metrics:
        if test_metrics.get('roc_auc') is not None:
            experiment['test_roc_auc'] = test_metrics['roc_auc']
        if test_metrics.get('accuracy') is not None:
            experiment['test_accuracy'] = test_metrics['accuracy']
        if test_metrics.get('precision') is not None:
            experiment['test_precision'] = test_metrics['precision']
        if test_metrics.get('recall') is not None:
            experiment['test_recall'] = test_metrics['recall']
        if test_metrics.get('f1') is not None:
            experiment['test_f1'] = test_metrics['f1']
    
    # HC3 benchmark - only add if available
    if hc3_benchmark:
        if hc3_benchmark.get('roc_auc') is not None:
            experiment['hc3_roc_auc'] = hc3_benchmark['roc_auc']
        if hc3_benchmark.get('accuracy') is not None:
            experiment['hc3_accuracy'] = hc3_benchmark['accuracy']
        if hc3_benchmark.get('precision') is not None:
            experiment['hc3_precision'] = hc3_benchmark['precision']
        if hc3_benchmark.get('recall') is not None:
            experiment['hc3_recall'] = hc3_benchmark['recall']
        if hc3_benchmark.get('f1') is not None:
            experiment['hc3_f1'] = hc3_benchmark['f1']
        if hc3_benchmark.get('num_samples') is not None:
            experiment['hc3_samples'] = hc3_benchmark['num_samples']
    
    # Notes
    if notes:
        experiment['notes'] = notes
    
    # Add any additional kwargs
    experiment.update(kwargs)
    
    # Load existing results or create new DataFrame
    if RESULTS_CSV.exists():
        df = pd.read_csv(RESULTS_CSV)
    else:
        df = pd.DataFrame()
    
    # Append new experiment
    new_row = pd.DataFrame([experiment])
    df = pd.concat([df, new_row], ignore_index=True)
    
    # Reorder columns for readability (core fields first, then optional)
    core_cols = ['timestamp', 'model_name', 'notebook_name', 'model_type', 'features']
    optional_cols = [c for c in df.columns if c not in core_cols]
    df = df[core_cols + sorted(optional_cols)]
    
    # Remove columns that are entirely NaN (except core columns)
    for col in optional_cols:
        if df[col].isna().all():
            df = df.drop(columns=[col])
    
    # Save to CSV
    df.to_csv(RESULTS_CSV, index=False)
    
    print(f"\n{'='*60}")
    print(f"✓ Model experiment logged: {model_name}")
    print(f"  Saved to: {RESULTS_CSV}")
    print(f"  Total experiments: {len(df)}")
    print(f"{'='*60}\n")


def get_all_results() -> pd.DataFrame:
    """Load and return all logged model results."""
    if RESULTS_CSV.exists():
        return pd.read_csv(RESULTS_CSV)
    else:
        return pd.DataFrame()


def get_best_models(metric='val_roc_auc', top_k=5) -> pd.DataFrame:
    """Get top K models by a given metric."""
    df = get_all_results()
    if df.empty:
        return df
    
    if metric not in df.columns:
        print(f"Warning: Metric '{metric}' not found in results")
        return df
    
    # Sort by metric (descending) and return top K
    df_sorted = df.sort_values(metric, ascending=False, na_last=True)
    return df_sorted.head(top_k)


def print_results_summary():
    """Print a summary of all model results."""
    df = get_all_results()
    
    if df.empty:
        print("No model results found. Train some models first!")
        return
    
    print("\n" + "="*80)
    print("MODEL EXPERIMENTS SUMMARY")
    print("="*80)
    print(f"Total experiments: {len(df)}")
    print(f"\nModels by type:")
    print(df['model_type'].value_counts().to_string())
    
    if 'val_roc_auc' in df.columns:
        print(f"\nBest Validation ROC-AUC:")
        best = df.nlargest(5, 'val_roc_auc')[['model_name', 'model_type', 'val_roc_auc', 'notebook_name']]
        print(best.to_string(index=False))
    
    if 'hc3_roc_auc' in df.columns:
        hc3_models = df[df['hc3_roc_auc'].notna()]
        if len(hc3_models) > 0:
            print(f"\nHC3 Benchmark Results ({len(hc3_models)} models):")
            hc3_summary = hc3_models[['model_name', 'model_type', 'hc3_roc_auc', 'hc3_precision', 'hc3_recall']]
            print(hc3_summary.to_string(index=False))
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # Example usage
    print_results_summary()

