"""
HC3 Benchmarking Utilities

Provides reusable functions for benchmarking models on HC3 dataset.
Supports both meta-features-only and meta+embeddings modes for flexibility.
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, Optional, Any, Tuple
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from .benchmarking import load_hc3
from .feature_engineering import extract_text_features


def benchmark_model_on_hc3(
    model,
    feature_names: list,
    sample_frac: float = 0.005,
    use_embeddings: bool = False,
    embedding_extractor=None,
    embedding_model: Optional[str] = None,
    device: Optional[str] = None,
    batch_size: int = 64,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Benchmark a trained model on HC3 dataset.
    
    This function handles feature extraction (meta features + optional embeddings)
    and evaluates the model on HC3 data.
    
    Parameters:
    -----------
    model : torch.nn.Module or sklearn model
        Trained model to evaluate
    feature_names : list
        List of feature names used during training (from dataset_info['feature_names'])
    sample_frac : float
        Fraction of HC3 data to use (default: 0.005 = 0.5%)
    use_embeddings : bool
        If True, extract and use embeddings. If False, use only meta features.
    embedding_extractor : EmbeddingExtractor, optional
        Pre-initialized embedding extractor (if None and use_embeddings=True, will create one)
    embedding_model : str, optional
        HuggingFace model name for embeddings (only used if embedding_extractor is None)
    device : str, optional
        Device for model inference ('cuda', 'mps', 'cpu')
    batch_size : int
        Batch size for model inference
    random_state : int
        Random seed for sampling HC3 data
        
    Returns:
    --------
    dict
        Dictionary with metrics: roc_auc, accuracy, precision, recall, f1, num_samples
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    print("="*60)
    print("BENCHMARKING MODEL ON HC3 DATASET")
    print("="*60)
    print(f"Mode: {'Meta features + Embeddings' if use_embeddings else 'Meta features only'}")
    print(f"Sample fraction: {sample_frac*100:.2f}%")
    print("="*60)
    
    # Step 1: Load HC3 dataset
    print("\n1. Loading HC3 dataset...")
    hc3_df_full = load_hc3(split="train", combine_topic_answer=False)
    print(f"   Loaded {len(hc3_df_full)} samples from HC3")
    
    # Sample data
    print(f"\n2. Sampling {sample_frac*100:.2f}% of the data...")
    hc3_df = hc3_df_full.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)
    print(f"   Using {len(hc3_df)} samples for benchmarking")
    print(f"   Class distribution:")
    print(f"     - Human samples (label=0): {(hc3_df['label'] == 0).sum()}")
    print(f"     - AI samples (label=1): {(hc3_df['label'] == 1).sum()}")
    
    # Step 2: Prepare data for feature extraction
    hc3_for_features = hc3_df.copy()
    hc3_for_features['answer'] = hc3_for_features['text']
    if 'topic' not in hc3_for_features.columns:
        hc3_for_features['topic'] = 'unknown'
    
    # Step 3: Extract meta features
    print("\n3. Extracting meta features...")
    hc3_meta_features = extract_text_features(hc3_for_features)
    
    # Add dummy topic encoding
    from sklearn.preprocessing import LabelEncoder
    hc3_meta_features['topic_encoded'] = 0
    
    # Step 4: Extract embeddings (if requested)
    hc3_embeddings = None
    if use_embeddings:
        print("\n4. Extracting embeddings...")
        if embedding_extractor is None:
            if embedding_model is None:
                raise ValueError("use_embeddings=True requires either embedding_extractor or embedding_model")
            print(f"   Initializing embedding extractor with model: {embedding_model}")
            from .embeddings import EmbeddingExtractor
            embedding_extractor = EmbeddingExtractor(
                model_name_or_path=embedding_model,
                device=device
            )
        
        hc3_texts = hc3_df['text'].tolist()
        hc3_embeddings = embedding_extractor.extract_embeddings(
            hc3_texts, 
            batch_size=8, 
            show_progress=True
        )
        hc3_embeddings = np.array(hc3_embeddings)
        print(f"   Extracted embeddings shape: {hc3_embeddings.shape}")
    else:
        print("\n4. Skipping embeddings (using meta features only)")
    
    # Step 5: Combine features to match training format
    print("\n5. Combining features...")
    
    # Detect model's expected feature count
    model_expected_features = None
    if isinstance(model, torch.nn.Module):
        # PyTorch model: check first layer input size
        first_layer = None
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                first_layer = module
                break
        if first_layer is not None:
            model_expected_features = first_layer.in_features
    else:
        # sklearn model: check n_features_in_ attribute
        if hasattr(model, 'n_features_in_'):
            model_expected_features = model.n_features_in_
        elif hasattr(model, 'model') and hasattr(model.model, 'n_features_in_'):
            # For wrapped models like ScaledModel
            model_expected_features = model.model.n_features_in_
    
    # Separate meta and embedding feature names
    meta_feature_names = [name for name in feature_names if not name.startswith('embedding_')]
    embedding_feature_names = [name for name in feature_names if name.startswith('embedding_')]
    
    print(f"   Meta features expected: {len(meta_feature_names)}")
    if model_expected_features:
        print(f"   Model expects: {model_expected_features} features")
    
    if use_embeddings:
        print(f"   Embedding features expected: {len(embedding_feature_names)}")
    else:
        print(f"   Embedding features: SKIPPED (using meta features only)")
    
    # Build feature array
    hc3_features_list = []
    
    # Add meta features
    for feat_name in meta_feature_names:
        if feat_name in hc3_meta_features.columns:
            hc3_features_list.append(hc3_meta_features[feat_name].values)
        else:
            print(f"   ⚠️  Warning: Feature '{feat_name}' not found, filling with zeros")
            hc3_features_list.append(np.zeros(len(hc3_meta_features)))
    
    # Determine if we need to add embeddings or padding
    meta_feature_count = len(meta_feature_names)
    
    # Add embeddings (if using embeddings)
    if use_embeddings and len(embedding_feature_names) > 0:
        if hc3_embeddings is None:
            raise ValueError("use_embeddings=True but embeddings not extracted")
        
        expected_emb_dim = len(embedding_feature_names)
        actual_emb_dim = hc3_embeddings.shape[1]
        
        if actual_emb_dim == expected_emb_dim:
            hc3_features_list.append(hc3_embeddings)
        elif actual_emb_dim > expected_emb_dim:
            print(f"   ⚠️  Truncating embeddings from {actual_emb_dim} to {expected_emb_dim} dims")
            hc3_features_list.append(hc3_embeddings[:, :expected_emb_dim])
        else:
            print(f"   ⚠️  Padding embeddings from {actual_emb_dim} to {expected_emb_dim} dims")
            padding = np.zeros((hc3_embeddings.shape[0], expected_emb_dim - actual_emb_dim))
            hc3_features_list.append(np.hstack([hc3_embeddings, padding]))
    elif not use_embeddings and model_expected_features and model_expected_features > meta_feature_count:
        # Model was trained with embeddings, but we're benchmarking without them
        # Only pad if the model actually expects more features than meta features
        needed_emb_dim = model_expected_features - meta_feature_count
        print(f"   ⚠️  Model expects {model_expected_features} features but only {meta_feature_count} meta features")
        print(f"   ⚠️  Padding {needed_emb_dim} embedding dimensions with zeros")
        padding = np.zeros((len(hc3_df), needed_emb_dim))
        hc3_features_list.append(padding)
    elif not use_embeddings and len(embedding_feature_names) > 0 and model_expected_features is None:
        # Fallback: if we can't detect model features and feature_names has embeddings,
        # assume model needs them (for PyTorch models that were trained with embeddings)
        if isinstance(model, torch.nn.Module):
            print(f"   ⚠️  PyTorch model: Padding {len(embedding_feature_names)} embedding dimensions with zeros")
            padding = np.zeros((len(hc3_df), len(embedding_feature_names)))
            hc3_features_list.append(padding)
        # For sklearn models without detected feature count, don't pad (they likely don't need embeddings)
    
    # Stack all features
    hc3_features = np.column_stack(hc3_features_list)
    
    print(f"   Final features shape: {hc3_features.shape}")
    if model_expected_features:
        print(f"   Model expects: {model_expected_features} features")
        if hc3_features.shape[1] != model_expected_features:
            raise ValueError(f"Feature dimension mismatch: got {hc3_features.shape[1]}, model expects {model_expected_features}")
    else:
        print(f"   Expected shape: ({len(hc3_df)}, {len(feature_names)})")
        if hc3_features.shape[1] != len(feature_names):
            print(f"   ⚠️  Warning: Feature count ({hc3_features.shape[1]}) doesn't match feature_names ({len(feature_names)})")
            print(f"   ⚠️  This may be OK if model was trained with different features")
    
    # Step 6: Evaluate model
    print("\n6. Evaluating model on HC3...")
    
    # Convert to tensors if PyTorch model
    if isinstance(model, torch.nn.Module):
        model.eval()
        model.to(device)
        hc3_tensor = torch.FloatTensor(hc3_features).to(device)
        hc3_labels = hc3_df['label'].values
        
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(hc3_tensor), batch_size):
                batch = hc3_tensor[i:i+batch_size]
                outputs = model(batch)
                probs = outputs.cpu().numpy().flatten()
                all_probs.extend(probs)
        
        all_probs = np.array(all_probs)
        all_predictions = (all_probs >= 0.5).astype(int)
    else:
        # sklearn model
        hc3_labels = hc3_df['label'].values
        all_probs = model.predict_proba(hc3_features)[:, 1]
        all_predictions = model.predict(hc3_features)
    
    # Calculate metrics
    metrics = {
        'roc_auc': roc_auc_score(hc3_labels, all_probs),
        'accuracy': accuracy_score(hc3_labels, all_predictions),
        'precision': precision_score(hc3_labels, all_predictions),
        'recall': recall_score(hc3_labels, all_predictions),
        'f1': f1_score(hc3_labels, all_predictions),
        'num_samples': len(hc3_df)
    }
    
    # Display results
    print("\n" + "="*60)
    print("HC3 BENCHMARK RESULTS")
    print("="*60)
    print(f"\nDataset: HC3 (Human ChatGPT Comparison Corpus)")
    print(f"Samples evaluated: {len(hc3_df)}")
    print(f"  - Human samples (label=0): {(hc3_labels == 0).sum()}")
    print(f"  - AI samples (label=1): {(hc3_labels == 1).sum()}")
    print(f"\nPerformance Metrics:")
    print(f"  ROC-AUC:    {metrics['roc_auc']:.4f}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print("="*60)
    
    return metrics

