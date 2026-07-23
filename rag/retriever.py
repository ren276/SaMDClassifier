"""
Semantic retrieval over the NLEM 2022 ChromaDB vector store built by
rag/ingest.py. No LLM involved: a local sentence-transformer embeds the
query and Chroma returns nearest-neighbour chunks.
"""

from pathlib import Path
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from ingest import (
    VECTOR_STORE_DIR,
    MEDICINES_COLLECTION,
    ALPHA_INDEX_COLLECTION,
    EMBEDDING_MODEL_NAME,
)

QUERY_TEMPLATE = "essential medicines for {disease_name}"


@lru_cache(maxsize=1)
def _get_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def _get_client():
    if not Path(VECTOR_STORE_DIR).exists():
        raise FileNotFoundError(
            f"Vector store not found at {VECTOR_STORE_DIR}. Run `python rag/ingest.py` first."
        )
    return chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))


@lru_cache(maxsize=1)
def _get_medicines_collection():
    return _get_client().get_collection(MEDICINES_COLLECTION)


@lru_cache(maxsize=1)
def _get_alpha_collection():
    return _get_client().get_collection(ALPHA_INDEX_COLLECTION)


def retrieve(disease_name: str, top_k: int = 5) -> list[dict]:
    """Return up to `top_k` NLEM chunks semantically closest to
    `disease_name`, each as {"text", "page", "distance", **metadata}."""
    query = QUERY_TEMPLATE.format(disease_name=disease_name)
    embedding = _get_model().encode([query], convert_to_numpy=True).tolist()

    collection = _get_medicines_collection()
    results = collection.query(query_embeddings=embedding, n_results=top_k)

    chunks = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    for doc_text, metadata, distance in zip(documents, metadatas, distances):
        chunk = {"text": doc_text, "page": metadata.get("page"), "distance": distance}
        chunk.update(metadata)
        chunks.append(chunk)
    return chunks


def get_alpha_index_entries() -> list[dict]:
    """Return every entry of the alphabetical medicine name index."""
    collection = _get_alpha_collection()
    raw = collection.get()
    entries = []
    for doc_text, metadata in zip(raw.get("documents", []), raw.get("metadatas", [])):
        entry = {"name": doc_text}
        entry.update(metadata)
        entries.append(entry)
    return entries
