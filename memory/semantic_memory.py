import json
import os
import re
import faiss
import numpy as np

# ====================================
# LOAD MODEL
# ====================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.path.join(BASE_DIR, "faiss.index")
TEXT_FILE = os.path.join(BASE_DIR, "texts.json")

dimension = 384
MODEL = None
EMBEDDINGS_ENABLED = False

if os.environ.get("ENABLE_SEMANTIC_EMBEDDINGS") == "1":
    try:
        from sentence_transformers import SentenceTransformer

        MODEL = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            local_files_only=True
        )
        EMBEDDINGS_ENABLED = True
    except Exception:
        MODEL = None
        EMBEDDINGS_ENABLED = False

# ====================================
# LOAD EXISTING MEMORY
# ====================================
if EMBEDDINGS_ENABLED and os.path.exists(INDEX_FILE):

    index = faiss.read_index(INDEX_FILE)

    with open(TEXT_FILE, "r") as f:
        texts = json.load(f)

else:

    # Cosine similarity index
    index = faiss.IndexFlatIP(dimension)

    texts = []

if not os.path.exists(TEXT_FILE):
    with open(TEXT_FILE, "w") as f:
        json.dump(texts, f, indent=2)
elif not texts:
    with open(TEXT_FILE, "r") as f:
        texts = json.load(f)


def tokenize(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def lexical_search(query, k=3):
    query_terms = tokenize(query)
    if not query_terms:
        return []

    results = []
    for text in texts:
        text_terms = tokenize(text)
        if not text_terms:
            continue
        overlap = len(query_terms & text_terms)
        score = overlap / max(len(query_terms), 1)
        if score > 0:
            results.append({
                "text": text,
                "score": float(score)
            })

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:k]

# ====================================
# ADD MEMORY
# ====================================
def add_memory(text):

    text = text.strip()

    # Prevent duplicates
    if text in texts:
        return

    if not EMBEDDINGS_ENABLED:
        texts.append(text)
        with open(TEXT_FILE, "w") as f:
            json.dump(texts, f, indent=2)
        return

    embedding = MODEL.encode([text], normalize_embeddings=True)

    embedding = np.array(
        embedding
    ).astype("float32")

    index.add(embedding)

    texts.append(text)

    faiss.write_index(
        index,
        INDEX_FILE
    )

    with open(TEXT_FILE, "w") as f:
        json.dump(
            texts,
            f,
            indent=2
        )

# ====================================
# SEARCH MEMORY
# ====================================
def search_memory(query, k=3):

    if len(texts) == 0:
        return []

    if not EMBEDDINGS_ENABLED:
        return lexical_search(query, k)

    embedding = MODEL.encode(
        [query],
        normalize_embeddings=True
    )

    embedding = np.array(
        embedding
    ).astype("float32")

    # retrieve extra candidates
    scores, indices = index.search(
        embedding,
        k * 2
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx >= len(texts):
            continue

        # similarity threshold
        if score > 0.40:

            results.append({
                "text": texts[idx],
                "score": float(score)
            })

    return results[:k]


# ====================================
# DELETE MEMORY
# ====================================
def clear_memory():

    global index
    global texts

    index = faiss.IndexFlatIP(dimension)

    texts = []

    if EMBEDDINGS_ENABLED:
        faiss.write_index(
            index,
            INDEX_FILE
        )

    with open(TEXT_FILE, "w") as f:
        json.dump([], f)


# ====================================
# TEST
# ====================================
if __name__ == "__main__":

    add_memory("ROS2 is a robotics middleware.")
    add_memory("SLAM helps robots localize.")
    add_memory("PID control stabilizes systems.")
    add_memory("Uber is a ride app.")
    add_memory("Zomato is a food delivery app.")

    print()

    print(search_memory("What is ROS2"))

    print()

    print(search_memory("Robot localization"))

    print()

    print(search_memory("Food apps"))
