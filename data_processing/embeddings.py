"""
Text Embeddings Extraction

Uses llama-embed model to extract embeddings from text.
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoTokenizer
from typing import List, Optional
import pandas as pd
from tqdm import tqdm


def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Average pooling with attention mask.
    
    Parameters:
    -----------
    last_hidden_states : torch.Tensor
        Token-level hidden states from the model
    attention_mask : torch.Tensor
        Attention mask indicating which tokens are valid
        
    Returns:
    -------
    torch.Tensor
        Sentence-level embeddings
    """
    last_hidden_states = last_hidden_states.to(torch.float32)
    last_hidden_states_masked = last_hidden_states.masked_fill(
        ~attention_mask[..., None].bool(), 
        0.0
    )
    embedding = last_hidden_states_masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    embedding = F.normalize(embedding, dim=-1)  # L2-normalise
    return embedding


class EmbeddingExtractor:
    """
    Class to extract embeddings from text using llama-embed model.
    """
    
    def __init__(
        self,
        model_name_or_path: str = "nvidia/llama-embed-nemotron-8b",
        device: Optional[str] = None,
        batch_size: int = 8,
        max_length: int = 4096
    ):
        """
        Initialize the embedding extractor.
        
        Parameters:
        -----------
        model_name_or_path : str
            HuggingFace model name or path
        device : str, optional
            Device to use ('cuda', 'mps', 'cpu'). If None, auto-detect.
        batch_size : int
            Batch size for processing texts
        max_length : int
            Maximum sequence length for tokenization
        """
        self.model_name_or_path = model_name_or_path
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        
        # Load tokenizer
        print(f"Loading tokenizer from {model_name_or_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            padding_side="left",
        )
        
        # Load model
        print(f"Loading model from {model_name_or_path}...")
        print("⚠️  WARNING: This is a large model (~16GB) and may take several minutes to load/download.")
        print("   If this hangs, the model may be downloading or loading into memory...")
        
        # Determine attention implementation
        # Only use flash_attention_2 if CUDA is available AND flash-attn is installed
        attn_implementation = "eager"  # Default to eager (safer, works everywhere)
        if torch.cuda.is_available():
            try:
                import flash_attn
                attn_implementation = "flash_attention_2"
                print("   Using flash_attention_2 (CUDA + flash-attn detected)")
            except ImportError:
                print("   flash-attn not available, using eager attention")
                attn_implementation = "eager"
        else:
            print(f"   Using eager attention (device: {device})")
        
        # Load model with error handling
        try:
            self.model = AutoModel.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                attn_implementation=attn_implementation,
                low_cpu_mem_usage=True,  # Helps with large models
            ).eval()
            
            self.model = self.model.to(self.device)
            print(f"✓ Model loaded successfully on device: {self.device}")
        except Exception as e:
            print(f"\n❌ ERROR loading model: {e}")
            print("\nPossible solutions:")
            print("  1. Check internet connection (model may be downloading)")
            print("  2. Ensure you have enough disk space (~16GB)")
            print("  3. Try using a smaller embedding model")
            print("  4. If on Mac, ensure MPS is available: torch.backends.mps.is_available()")
            raise
    
    def extract_embeddings(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Extract embeddings from a list of texts.
        
        Parameters:
        -----------
        texts : List[str]
            List of text strings to embed
        show_progress : bool
            Whether to show progress bar
            
        Returns:
        -------
        np.ndarray
            Array of embeddings with shape (n_texts, embedding_dim)
        """
        all_embeddings = []
        
        # Process in batches
        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        iterator = range(0, len(texts), self.batch_size)
        
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting embeddings", total=n_batches)
        
        for i in iterator:
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize
            batch_dict = self.tokenizer(
                text=batch_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            
            attention_mask = batch_dict["attention_mask"]
            
            # Forward pass
            with torch.no_grad():
                model_outputs = self.model(**batch_dict)
            
            # Pool to get sentence-level embeddings
            batch_embeddings = average_pool(
                model_outputs.last_hidden_state,
                attention_mask
            )
            
            # Convert to numpy and store
            all_embeddings.append(batch_embeddings.cpu().numpy())
        
        # Concatenate all batches
        embeddings = np.vstack(all_embeddings)
        
        return embeddings
    
    def extract_embeddings_from_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'answer',
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Extract embeddings from a DataFrame column.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing texts
        text_column : str
            Name of the column containing text
        show_progress : bool
            Whether to show progress bar
            
        Returns:
        -------
        np.ndarray
            Array of embeddings with shape (n_rows, embedding_dim)
        """
        texts = df[text_column].tolist()
        return self.extract_embeddings(texts, show_progress=show_progress)


def extract_embeddings_simple(
    texts: List[str],
    model_name_or_path: str = "nvidia/llama-embed-nemotron-8b",
    batch_size: int = 8,
    device: Optional[str] = None
) -> np.ndarray:
    """
    Simple function to extract embeddings (creates extractor internally).
    
    Parameters:
    -----------
    texts : List[str]
        List of text strings to embed
    model_name_or_path : str
        HuggingFace model name or path
    batch_size : int
        Batch size for processing
    device : str, optional
        Device to use. If None, auto-detect.
        
    Returns:
    -------
    np.ndarray
        Array of embeddings
    """
    extractor = EmbeddingExtractor(
        model_name_or_path=model_name_or_path,
        device=device,
        batch_size=batch_size
    )
    return extractor.extract_embeddings(texts)

