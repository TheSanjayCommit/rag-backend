from fastembed import TextEmbedding
from typing import List
import numpy as np

# Use the smallest possible model for Render Free Tier (only ~100MB)
MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of strings using FastEmbed.
    This model is highly optimized for low-memory environments.
    """
    try:
        # Lazy initialization to save startup memory
        model = TextEmbedding(model_name=MODEL_NAME)
        embeddings = list(model.embed(texts))
        return [e.tolist() for e in embeddings]
    except Exception as e:
        print(f"Embedding Error: {e}")
        # Fallback to zero-vectors if model fails to load (prevents 502 crash)
        return [[0.0] * 768 for _ in texts]

def get_embedding_dimension() -> int:
    """Returns the dimension of the embeddings (768 for Nomic)."""
    return 768
