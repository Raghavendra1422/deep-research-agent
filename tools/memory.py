import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ── HuggingFace Embedding Model (runs locally, 100% free) ──
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ── ChromaDB local client ───────────────────────────────────
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="research_reports")


def chunk_text(text: str, chunk_size: int = 150) -> list:
    """Split report into chunks for storage"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def store_report(query: str, report: str):
    """Store report chunks in ChromaDB with HuggingFace embeddings"""
    print("\n[MEMORY] Storing report in ChromaDB...")

    chunks = chunk_text(report)

    for i, chunk in enumerate(chunks):
        chunk_id = f"{query[:30].replace(' ', '_')}_{i}"

        # Generate embedding using HuggingFace locally
        embedding = embedder.encode(chunk).tolist()

        collection.upsert(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"query": query, "chunk_index": i}]
        )

    print(f"[MEMORY] ✅ Stored {len(chunks)} chunks in ChromaDB")


def search_report(question: str, top_k: int = 3) -> list:
    """Search ChromaDB for relevant chunks using semantic search"""

    # Embed the question using HuggingFace
    question_embedding = embedder.encode(question).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )

    chunks = results["documents"][0] if results["documents"] else []
    return chunks


def report_exists(query: str) -> bool:
    """Check if we already have research for this query"""
    results = collection.get(
        where={"query": query[:30] if len(query) > 30 else query}
    )
    return len(results["ids"]) > 0