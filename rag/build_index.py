import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

DOCUMENTS_PATH = "rag/documents"
CHUNKS_PATH = "rag/chunks.pkl"
INDEX_PATH = "rag/index.faiss"


def load_documents():
    documents = []
    for file in os.listdir(DOCUMENTS_PATH):
        if file.endswith(".txt"):
            with open(os.path.join(DOCUMENTS_PATH, file), "r", encoding="utf-8") as f:
                documents.append(f.read())
    return documents


def chunk_text(text, chunk_size=200):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


def create_chunks(documents):
    all_chunks = []

    for doc in documents:
        chunks = chunk_text(doc)
        all_chunks.extend(chunks)

    return all_chunks


def build_faiss_index(chunks):
    print("Generating embeddings...")

    embeddings = model.encode(chunks)

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, embeddings


def save_index(index, chunks):
    faiss.write_index(index, INDEX_PATH)

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)


def main():
    print("Loading documents...")
    documents = load_documents()

    print(f"Loaded {len(documents)} documents")

    print("Creating chunks...")
    chunks = create_chunks(documents)

    print(f"Created {len(chunks)} chunks")

    index, _ = build_faiss_index(chunks)

    print("Saving index...")
    save_index(index, chunks)

    print("✅ Index built successfully!")


if __name__ == "__main__":
    main()
