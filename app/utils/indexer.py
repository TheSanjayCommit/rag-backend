import os
import pandas as pd
import faiss
import pickle
import numpy as np
from app.config.settings import settings
from app.utils.embeddings import get_embeddings, get_embedding_dimension

def build_index_if_missing():
    """Builds the FAISS index from CSVs using memory-efficient batching."""
    index_file = os.path.join(settings.FAISS_INDEX_PATH, "index.faiss")
    docs_file = os.path.join(settings.FAISS_INDEX_PATH, "docs.pkl")

    if os.path.exists(index_file) and os.path.exists(docs_file):
        print("✅ Database found. Skipping indexing.")
        return

    print("🏗️ Building index in batches to save memory...")
    os.makedirs(settings.FAISS_INDEX_PATH, exist_ok=True)

    all_docs = []
    for file_name in os.listdir(settings.DATA_PATH):
        if file_name.endswith(".csv"):
            file_path = os.path.join(settings.DATA_PATH, file_name)
            try:
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    doc = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                    all_docs.append(doc)
            except Exception as e:
                print(f"Error reading {file_name}: {e}")

    if not all_docs:
        print("⚠️ No data found. Indexing skipped.")
        return

    try:
        dimension = get_embedding_dimension()
        index = faiss.IndexFlatL2(dimension)
        
        # BATCH PROCESSING (100 docs at a time)
        batch_size = 100
        total = len(all_docs)
        
        for i in range(0, total, batch_size):
            batch = all_docs[i:i + batch_size]
            print(f"Vectorizing batch {i//batch_size + 1}/{(total//batch_size)+1}...")
            embeddings = np.array(get_embeddings(batch)).astype('float32')
            index.add(embeddings)
        
        faiss.write_index(index, index_file)
        with open(docs_file, "wb") as f:
            pickle.dump(all_docs, f)
            
        print(f"✅ Successfully indexed {len(all_docs)} documents.")
    except Exception as e:
        print(f"❌ Indexing failed: {e}")
