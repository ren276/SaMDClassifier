"""
The one place the embedding model and its revision are named. Read by ingest.py, retriever.py and
the Dockerfile's build-time download, so the model that is baked into the image, the one queried
at runtime and the one that built the vector store are the same by construction.

REVISION is the commit the built image (samd-classifier:latest, created 2026-09-08) had cached,
read from its huggingface cache (MEASURED: refs/main and the snapshot directory name agree).

The revision that INGEST used to build the Chroma store is INFERRED to be the same: ingest ran
against an unpinned model, and nothing recorded which revision it fetched. If the vectors were
built with an older revision, query-time and ingest-time embeddings differ silently. Re-running
ingest.py under this pin is the only way to make that provable.
"""

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
