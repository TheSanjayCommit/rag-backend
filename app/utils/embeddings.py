from sentence_transformers import SentenceTransformer
import os

# Initialize model once at startup
# Using a small, fast model for production speed
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def get_embeddings(text):
    """
    Guaranteed string-safe embedding generation.
    Prevents Multimodal ValueError by force-casting input.
    """
    if isinstance(text, dict):
        # If a dict leaked through, extract the most likely text field
        text = text.get('text', text.get('queries', str(text)))
    
    if isinstance(text, list):
        # Ensure all items in list are strings
        text = [str(t) for t in text]
    else:
        # Ensure single input is a string
        text = str(text)

    return model.encode(text)

def get_embedding_dimension():
    return model.get_sentence_embedding_dimension()
