FROM python:3.11-slim

WORKDIR /app

# Install Python deps first for better layer caching
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Pre-download the ONNX embedding model so the image is self-contained
# (no cold-start fetch from HuggingFace on the first request).
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

# Copy application code + the corpus (PDFs) and extraction cache
COPY backend/ ./backend/
COPY scripts/ ./scripts/
COPY data/ ./data/

# Build a NATIVE graph + vector store inside the image. The committed Kùzu DB
# is platform-specific and won't open on Linux, so we regenerate it here from
# the corpus PDFs + the committed extraction cache. No API key / LLM calls are
# needed (every chunk is already cached); the embedding model is pre-baked above.
RUN python scripts/ingest_corpus.py --glob "*.pdf"

EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
