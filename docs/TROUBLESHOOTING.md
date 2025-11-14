# Troubleshooting Guide

## Common Issues and Solutions

### Issue: Code Hangs During Embedding Extraction

**Symptoms:**
- Cell runs indefinitely when calling `EmbeddingExtractor()`
- No error messages, just hangs
- Happens in HC3 benchmarking or dataset creation

**Root Cause:**
The `nvidia/llama-embed-nemotron-8b` model is very large (~16GB) and:
1. Takes 5-15 minutes to download (if not cached)
2. Takes 5-10 minutes to load into memory
3. On Mac MPS, can be slower than CUDA

**Solutions:**

1. **Wait it out** (Recommended if you have time)
   - Model loading can take 10-15 minutes
   - Look for progress messages: "Loading tokenizer...", "Loading model..."
   - HuggingFace shows download progress if downloading

2. **Use a smaller/faster embedding model**
   ```python
   # Instead of llama-embed-nemotron-8b, use:
   embedding_extractor = EmbeddingExtractor(
       model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",  # Much smaller, faster
       device=device
   )
   ```

3. **Skip embedding extraction for benchmarking**
   - Use meta features only for HC3 benchmarking
   - Embeddings are only needed if your model was trained with embeddings

4. **Check if model is already cached**
   ```python
   from huggingface_hub import snapshot_download
   # Check cache location
   import os
   cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
   print(f"Model cache: {cache_dir}")
   ```

### Issue: Flash Attention Installation Fails

**Symptoms:**
- Error: "CUDA_HOME environment variable is not set"
- Happens when trying to install `flash-attn`

**Solution:**
- **On Mac**: Flash attention is not needed (and won't work without CUDA)
- The code automatically falls back to "eager" attention
- You can ignore flash-attn installation errors

### Issue: Out of Memory Errors

**Symptoms:**
- `RuntimeError: CUDA out of memory` or similar
- Model fails to load

**Solutions:**
1. **Reduce batch size**
   ```python
   embedding_extractor = EmbeddingExtractor(
       model_name_or_path="...",
       batch_size=4  # Reduce from default 8
   )
   ```

2. **Use CPU instead of GPU**
   ```python
   embedding_extractor = EmbeddingExtractor(
       model_name_or_path="...",
       device="cpu"  # Slower but uses less memory
   )
   ```

3. **Use smaller model** (see above)

### Issue: Model Results Not Logging

**Symptoms:**
- `log_model_experiment()` doesn't create CSV
- No confirmation message

**Solutions:**
1. **Check imports**
   ```python
   from data_processing import log_model_experiment
   ```

2. **Check file permissions**
   - Ensure you have write permissions in the repo directory

3. **Check for errors**
   - Look for error messages in cell output
   - Common: missing variables (e.g., `rf_metrics` not defined)

### Issue: Perfect Validation Score (ROC-AUC = 1.0)

**Symptoms:**
- Validation ROC-AUC is exactly 1.0000
- Only 59 validation samples

**This is suspicious and suggests:**
- Overfitting to small validation set
- Possible data leakage (unlikely based on code review)
- Need stronger regularization

**Solutions:**
1. Increase regularization (dropout, weight_decay)
2. Add cross-validation instead of single split
3. Increase validation set size

## Performance Tips

### Faster Embedding Extraction
- Use smaller models: `sentence-transformers/all-MiniLM-L6-v2` (~80MB vs 16GB)
- Reduce batch size if memory constrained
- Use GPU (CUDA/MPS) instead of CPU

### Faster Benchmarking
- Sample smaller fraction of HC3 (e.g., 0.5% instead of full dataset)
- Use meta features only (skip embeddings)
- Cache embeddings if benchmarking multiple models

## Getting Help

If issues persist:
1. Check cell output for specific error messages
2. Verify all dependencies are installed
3. Check disk space (model needs ~16GB)
4. Check internet connection (for model download)

