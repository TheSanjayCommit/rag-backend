import os
import pandas as pd
import faiss
import pickle
from app.config.settings import settings
from app.utils.embeddings import get_embeddings, get_embedding_dimension

def build_index_if_missing():
    """Builds the FAISS index from CSVs if it doesn't exist."""
    index_file = os.path.join(settings.FAISS_INDEX_PATH, "index.faiss")
    docs_file = os.path.join(settings.FAISS_INDEX_PATH, "docs.pkl")

    if os.path.exists(index_file) and os.path.exists(docs_file):
        print("✅ Database found. Skipping indexing.")
        return

    print("🏗️ Database not found. Building index from CSVs...")
    
    # Ensure directory exists
    os.makedirs(settings.FAISS_INDEX_PATH, exist_ok=True)

    all_docs = []
    
    # Process each CSV in the data folder
    for file_name in os.listdir(settings.DATA_PATH):
        if file_name.endswith(".csv"):
            file_path = os.path.join(settings.DATA_PATH, file_name)
            try:
                df = pd.read_csv(file_path)
                # Convert rows to descriptive strings
                for _, row in df.iterrows():
                    doc = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                    all_docs.append(doc)
            except Exception as e:
                print(f"Error reading {file_name}: {e}")

    if not all_docs:
        print("⚠️ No data found in CSVs. Indexing skipped.")
        return

    try:
        # Generate embeddings
        print(f"Vectorizing {len(all_docs)} documents...")
        embeddings = get_embeddings(all_docs)
        
        # Create and save index
        dimension = get_embedding_dimension()
        index = faiss.IndexFlatL2(dimension)
        index.add(pd.NA if embeddings is None else embeddings)
        
        faiss.write_index(index, index_file)
        with open(docs_file, "wb") as f:
            pickle.dump(all_docs, f)
            
        print(f"✅ Successfully indexed {len(all_docs)} documents.")
    except Exception as e:
        print(f"❌ Indexing failed: {e}")
