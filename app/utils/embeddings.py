from fastembed import TextEmbedding
import numpy as np

# Lazy-loaded — model is NOT loaded at import time, only on first use
_model = None

def _get_model() -> TextEmbedding:
    """
    Lazy loader: initialises the ONNX embedding model on first call only.
    Uses fastembed (ONNX runtime) instead of sentence-transformers (PyTorch).
    RAM usage: ~80MB vs ~1.5GB for torch — safe for Render free tier (512MB).
    """
    global _model
    if _model is None:
        _model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _model

def get_embeddings(text):
    """
    Guaranteed string-safe embedding generation.
    Returns a numpy array — single input returns 1D, list input returns 2D.
    """
    model = _get_model()

    if isinstance(text, dict):
        text = text.get('text', text.get('queries', str(text)))

    if isinstance(text, list):
        text = [str(t) for t in text]
        embeddings = list(model.embed(text))
        return np.array(embeddings)
    else:
        text = str(text)
        embeddings = list(model.embed([text]))
        return np.array(embeddings[0])

def get_embedding_dimension() -> int:
    """Returns the embedding dimension for FAISS index creation."""
    return 384  # all-MiniLM-L6-v2 fixed dimension
