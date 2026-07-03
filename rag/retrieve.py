from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os


# ==========================
# Paths
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INDEX_PATH = os.path.join(BASE_DIR, "index.faiss")
CHUNKS_PATH = os.path.join(BASE_DIR, "chunks.pkl")


# ==========================
# Load Embedding Model
# ==========================

model = SentenceTransformer("all-MiniLM-L6-v2")


# ==========================
# Load FAISS Index
# ==========================

index = faiss.read_index(INDEX_PATH)


# ==========================
# Load Text Chunks
# ==========================

with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)


# ==========================
# Retrieve Function
# ==========================

def retrieve(query, k=1):
    """
    Returns the top-k most relevant chunks
    from the FAISS index.

    Parameters:
        query (str): User query
        k (int): Number of chunks to retrieve

    Returns:
        str: Retrieved text chunks
    """

    query_embedding = model.encode([query])

    distances, indices = index.search(
        np.array(query_embedding, dtype=np.float32),
        k
    )

    results = []

    for idx in indices[0]:
        if idx != -1:
            results.append(chunks[idx])

    return "\n\n".join(results)
