"""
Benchmarking Module - Simple and Clean Implementation

Provides functions to load and evaluate models on benchmark datasets for AI text detection.
Currently supports HC3 (Human ChatGPT Comparison Corpus).
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    import warnings
    warnings.warn("datasets library required. Install with: pip install datasets")

try:
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def clean_text(s):
    """Simple text cleaning function."""
    if s is None:
        return ""
    return " ".join(str(s).strip().replace("\r", " ").replace("\n", " ").split())


def make_input_string(topic: str, answer: str) -> str:
    """Combine topic and answer into formatted input string."""
    topic_clean = clean_text(topic)
    answer_clean = clean_text(answer)
    if topic_clean:
        return f"TOPIC: {topic_clean}\n\nANSWER: {answer_clean}"
    return answer_clean


def _to_list(value):
    """Safely convert any value to a Python list. Never fails."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (pd.Series, np.ndarray)):
        try:
            return value.tolist()
        except Exception:
            return list(value)
    # Single value
    return [value]


def load_hc3(
    split: str = "train",
    domain: Optional[str] = None,
    combine_topic_answer: bool = False,
    cache_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Load HC3 (Human ChatGPT Comparison Corpus) dataset.
    
    Simple implementation that avoids all array/Series boolean issues.
    """
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets library required. Install with: pip install datasets")
    
    print(f"Loading HC3 dataset (split={split}, domain={domain})...")
    
    # Load dataset
    try:
        if domain:
            dataset = load_dataset("Hello-SimpleAI/HC3", domain, cache_dir=cache_dir)
        else:
            dataset = load_dataset("Hello-SimpleAI/HC3", "all", cache_dir=cache_dir)
    except Exception as e:
        print(f"Error loading HC3: {e}")
        raise
    
    # Get split
    if split == "all":
        if "train" in dataset and "test" in dataset:
            train_data = dataset["train"]
            test_data = dataset["test"]
            # Process both splits
            all_data = []
            for item in train_data:
                all_data.append(item)
            for item in test_data:
                all_data.append(item)
        else:
            all_data = list(dataset.values())[0]
    elif split in dataset:
        all_data = dataset[split]
    else:
        available_splits = list(dataset.keys())
        print(f"Split '{split}' not available. Using '{available_splits[0]}' instead.")
        all_data = dataset[available_splits[0]]
    
    # Process data - use HuggingFace dataset directly, avoid pandas conversion issues
    results = []
    
    for item in all_data:
        # Get question
        question = item.get('question', '')
        question = clean_text(question) if question else ''
        
        # Get answers - use helper function to always get a list
        human_answers = _to_list(item.get('human_answers', []))
        chatgpt_answers = _to_list(item.get('chatgpt_answers', []))
        
        # Get domain
        domain_name = str(item.get('source', 'unknown'))
        
        # Process human answers
        for answer in human_answers:
            if answer is None:
                continue
            answer_str = clean_text(str(answer))
            if not answer_str:
                continue
            
            if combine_topic_answer:
                text = make_input_string(question, answer_str)
            else:
                text = answer_str
            
            results.append({
                'text': text,
                'label': 0,
                'domain': domain_name,
                'source': 'human',
                'question': question
            })
        
        # Process ChatGPT answers
        for answer in chatgpt_answers:
            if answer is None:
                continue
            answer_str = clean_text(str(answer))
            if not answer_str:
                continue
            
            if combine_topic_answer:
                text = make_input_string(question, answer_str)
            else:
                text = answer_str
            
            results.append({
                'text': text,
                'label': 1,
                'domain': domain_name,
                'source': 'chatgpt',
                'question': question
            })
    
    result_df = pd.DataFrame(results)
    
    if len(result_df) == 0:
        raise ValueError("No data loaded from HC3 dataset")
    
    print(f"✓ Loaded {len(result_df)} samples from HC3")
    print(f"  Human samples (label=0): {(result_df['label'] == 0).sum()}")
    print(f"  AI samples (label=1): {(result_df['label'] == 1).sum()}")
    
    return result_df


def evaluate_on_benchmark(
    model,
    tokenizer,
    benchmark_df: pd.DataFrame,
    batch_size: int = 16,
    max_length: int = 512,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """Evaluate a model on a benchmark dataset."""
    if not TORCH_AVAILABLE:
        raise ImportError("torch and sklearn required for model evaluation")
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.eval()
    model.to(device)
    
    texts = benchmark_df['text'].tolist()
    labels = benchmark_df['label'].values
    
    all_probs = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
    
    all_probs = np.array(all_probs)
    predictions = (all_probs >= 0.5).astype(int)
    
    metrics = {
        'roc_auc': roc_auc_score(labels, all_probs),
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions),
        'recall': recall_score(labels, predictions),
        'f1': f1_score(labels, predictions),
        'predictions': all_probs,
        'labels': labels
    }
    
    return metrics


def benchmark_model(
    model,
    tokenizer,
    benchmarks: list = ["hc3"],
    splits: Dict[str, str] = None,
    combine_topic_answer: bool = False,
    batch_size: int = 16,
    device: Optional[str] = None
) -> pd.DataFrame:
    """Benchmark a model on benchmark datasets."""
    if splits is None:
        splits = {"hc3": "test"}
    
    results = []
    
    for benchmark_name in benchmarks:
        print(f"\n{'='*60}")
        print(f"Evaluating on {benchmark_name.upper()}")
        print(f"{'='*60}")
        
        try:
            if benchmark_name.lower() == "hc3":
                benchmark_df_full = load_hc3(
                    split=splits.get("hc3", "test"),
                    combine_topic_answer=combine_topic_answer
                )
                
                # Sample 0.5% of the data for faster benchmarking
                print(f"\n  Loaded {len(benchmark_df_full)} samples from HC3")
                print(f"  Sampling 0.5% of the data for benchmarking...")
                benchmark_df = benchmark_df_full.sample(frac=0.005, random_state=42).reset_index(drop=True)
                print(f"  Using {len(benchmark_df)} samples ({len(benchmark_df)/len(benchmark_df_full)*100:.2f}%)")
                print(f"  Class distribution:")
                print(f"    - Human samples (label=0): {(benchmark_df['label'] == 0).sum()}")
                print(f"    - AI samples (label=1): {(benchmark_df['label'] == 1).sum()}")
            else:
                print(f"Unknown benchmark: {benchmark_name}. Skipping...")
                continue
            
            metrics = evaluate_on_benchmark(
                model=model,
                tokenizer=tokenizer,
                benchmark_df=benchmark_df,
                batch_size=batch_size,
                device=device
            )
            
            results.append({
                'benchmark': benchmark_name.upper(),
                'split': splits.get(benchmark_name.lower(), "test"),
                'roc_auc': metrics['roc_auc'],
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'num_samples': len(benchmark_df)
            })
            
            print(f"\nResults for {benchmark_name.upper()}:")
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1: {metrics['f1']:.4f}")
            
        except Exception as e:
            print(f"Error evaluating on {benchmark_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return pd.DataFrame(results)
