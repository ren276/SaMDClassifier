# SaMD Classifier Service Dockerfile
FROM python:3.10-slim

# Avoid prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Install required system packages:
# - libgomp1: required for OpenMP in XGBoost & PyTorch
# - curl: for health check probes
# - build-essential: for any binary wheel compatibility
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-cache the sentence transformer embedding model during build
# so the container is fully self-contained and functions offline
# The embedding model and revision are named once, in src/rag/embedding_pin.py, and read by this
# download, ingest.py and retriever.py. HF_HUB_OFFLINE stops the runtime from fetching a
# different revision than the one baked in here.
COPY src/rag/embedding_pin.py /tmp/embedding_pin.py
RUN PYTHONPATH=/tmp python -c "from embedding_pin import EMBEDDING_MODEL_NAME as n, EMBEDDING_MODEL_REVISION as r; from sentence_transformers import SentenceTransformer; SentenceTransformer(n, revision=r)"
ENV HF_HUB_OFFLINE=1

# Copy application data, models, and source code
COPY dataset/ ./dataset/
COPY models/ ./models/
COPY src/ ./src/
COPY ARTIFACT_MANIFEST.json ./

# Reported on /health. Late in the file so a new commit does not invalidate the layers above.
ARG BUILD_COMMIT=unknown
ENV BUILD_COMMIT=$BUILD_COMMIT

# Health check to ensure service readiness
HEALTHCHECK --interval=20s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Start FastAPI application
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
