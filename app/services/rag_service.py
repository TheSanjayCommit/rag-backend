import pandas as pd
import numpy as np
import os
import re
from app.utils.embeddings import get_embeddings, get_embedding_dimension
from app.utils.vector_store import create_index, save_index, load_index
from app.config.settings import settings

indices = {"india": None, "outbound": None, "inbound": None}
metadatas = {"india": None, "outbound": None, "inbound": None}

def is_valid_rag(rag_results, query):
    """
    Checks if RAG results are meaningful. Force-casts query to string to avoid dict errors.
    """
    if not rag_results: return False
    
    # Force query to string in case a dict leaked through
    q_str = str(query).lower()
    combined_text = " ".join(rag_results).lower()
    
    # Block Mixed Domains
    medical_keywords = ["mbbs", "bds", "medical", "hospital", "doctor", "dental"]
    if any(k in combined_text for k in medical_keywords) and "medical" not in q_str:
        return False

    q_words = [w for w in q_str.split() if len(w) > 3]
    match_count = sum(1 for word in q_words if word in combined_text)
    return match_count >= 1

def clean_numeric(value):
    if pd.isna(value) or value == "--": return 0
    num_str = re.sub(r'[^\d.]', '', str(value))
    return float(num_str) if num_str else 0

def apply_filters(results, filters):
    filtered = []
    for res in results:
        if filters.get("state") and filters["state"].lower() not in res.get("state", "").lower():
            continue
        try:
            fee = float(res.get("fee_val", 0))
            max_fee = float(filters.get("max_fee", float('inf')))
            if fee > max_fee: continue
            rating = float(res.get("rating_val", 0))
            min_rating = float(filters.get("min_rating", 0))
            if rating < min_rating: continue
        except Exception: continue
        filtered.append(res)
    return filtered

def rank_results(results):
    for res in results:
        rating = float(res.get("rating_val", 0))
        placement = float(res.get("placement_val", 0))
        fee = float(res.get("fee_val", 1))
        affordability = max(0, 10 - (fee / 50000))
        res["score"] = (rating * 0.4) + (placement * 0.4) + (affordability * 0.2)
    return sorted(results, key=lambda x: x["score"], reverse=True)

def process_csv_to_docs(file_path, category):
    if not os.path.exists(file_path): return [], []
    try:
        df = pd.read_csv(file_path)
    except Exception: return [], []
    texts, metadata = [], []
    for _, row in df.iterrows():
        if category == "india":
            name = str(row.get("College_Name", "")).strip()
            state = str(row.get("State", "")).strip()
            fee_val = str(row.get("UG_fee", "0")).replace(",", "")
            rating_val = row.get("Rating", 0)
            placement_val = row.get("Placement", 0)
            enriched_text = f"COLLEGE: {name} ({state}). Fees={fee_val}, Rating={rating_val}, Placement={placement_val}."
            meta = {"text": enriched_text, "name": name, "state": state, "fee_val": float(fee_val) if fee_val.isdigit() else 0, "rating_val": float(rating_val), "placement_val": float(placement_val)}
        else:
            region = row.get("Region of Country", "Unknown")
            enriched_text = f"STATISTIC: {region}."
            meta = {"text": enriched_text}
        texts.append(enriched_text)
        metadata.append(meta)
    return texts, metadata

def initialize_rag():
    global indices, metadatas
    datasets = {"india": "Indian_Engineering_Colleges_Dataset.csv", "outbound": "Share of Students Studying Abroad.csv", "inbound": "Share of Students from Abroad.csv"}
    for cat, filename in datasets.items():
        idx, meta = load_index(cat)
        if idx and meta:
            indices[cat], metadatas[cat] = idx, meta
            continue
        file_path = os.path.join(settings.DATA_PATH, filename)
        texts, meta = process_csv_to_docs(file_path, cat)
        if texts:
            embeddings = get_embeddings(texts)
            idx = create_index(get_embedding_dimension())
            idx.add(np.array(embeddings).astype('float32'))
            save_index(idx, meta, cat)
            indices[cat], metadatas[cat] = idx, meta

def search_rag(query, category, filters=None, top_k=10):
    if category not in indices or indices[category] is None: return []
    # Ensure query is string before embedding
    q_str = str(query)
    query_vector = get_embeddings(q_str)
    _, ann_indices = indices[category].search(np.array([query_vector]).astype('float32'), top_k)
    raw_results = [metadatas[category][i] for i in ann_indices[0] if i != -1 and i < len(metadatas[category])]
    if filters: raw_results = apply_filters(raw_results, filters)
    return rank_results(raw_results)
