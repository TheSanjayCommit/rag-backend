import os
import faiss
import pickle
import pandas as pd
from app.config.settings import settings
from app.utils.embeddings import get_embeddings

# Global cache to prevent re-loading on every request (Saves RAM/CPU)
_INDEX_CACHE = None
_DOCS_CACHE = None

def load_resources():
    """Load the FAISS index and documents into memory only once."""
    global _INDEX_CACHE, _DOCS_CACHE
    
    if _INDEX_CACHE is not None:
        return _INDEX_CACHE, _DOCS_CACHE

    index_file = os.path.join(settings.FAISS_INDEX_PATH, "index.faiss")
    docs_file = os.path.join(settings.FAISS_INDEX_PATH, "docs.pkl")

    if not os.path.exists(index_file) or not os.path.exists(docs_file):
        print("Index files not found. RAG will be disabled until indexed.")
        return None, None

    try:
        _INDEX_CACHE = faiss.read_index(index_file)
        with open(docs_file, "rb") as f:
            _DOCS_CACHE = pickle.load(f)
        print("RAG resources loaded successfully into cache.")
        return _INDEX_CACHE, _DOCS_CACHE
    except Exception as e:
        print(f"Error loading RAG resources: {e}")
        return None, None

def is_valid_rag():
    idx, _ = load_resources()
    return idx is not None

async def search_rag(query: str, k: int = 5):
    """Search the cached FAISS index."""
    index, docs = load_resources()
    if not index or not docs:
        return []

    try:
        # Generate embedding for the query
        query_vector = get_embeddings([query])[0]
        
        # Search index
        distances, indices = index.search(pd.NA if query_vector is None else [query_vector], k)
        
        results = []
        for i in indices[0]:
            if i != -1 and i < len(docs):
                results.append({
                    "text": docs[i],
                    "name": "Internal Database"
                })
        return results
    except Exception as e:
        print(f"Search error: {e}")
        return []
