import faiss
import numpy as np
import os
import pickle
from app.config.settings import settings

def create_index(dimension):
    """
    Create a new FAISS index.
    """
    return faiss.IndexFlatL2(dimension)

def save_index(index, metadata, name):
    """
    Save the index and metadata to disk.
    """
    if not os.path.exists(settings.FAISS_INDEX_PATH):
        os.makedirs(settings.FAISS_INDEX_PATH)
    
    index_path = os.path.join(settings.FAISS_INDEX_PATH, f"{name}.index")
    meta_path = os.path.join(settings.FAISS_INDEX_PATH, f"{name}_meta.pkl")
    
    faiss.write_index(index, index_path)
    with open(meta_path, 'wb') as f:
        pickle.dump(metadata, f)

def load_index(name):
    """
    Load the index and metadata from disk.
    """
    index_path = os.path.join(settings.FAISS_INDEX_PATH, f"{name}.index")
    meta_path = os.path.join(settings.FAISS_INDEX_PATH, f"{name}_meta.pkl")
    
    if os.path.exists(index_path) and os.path.exists(meta_path):
        index = faiss.read_index(index_path)
        with open(meta_path, 'rb') as f:
            metadata = pickle.load(f)
        return index, metadata
    return None, None
