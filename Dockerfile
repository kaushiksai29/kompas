FROM python:3.11-slim

WORKDIR /app

# Install Python deps first for better layer caching
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Pre-download the ONNX embedding model so the image is self-contained
# (no cold-start fetch from HuggingFace on the first request).
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

# Copy application code
COPY backend/ ./backend/

# NOTE: data/ (Kùzu graph + sqlite-vec store) is .gitignored and built offline.
# The image therefore needs a prebuilt data/ directory present at build time.
# Either build data/ locally before `docker build`, or run the ingest step here, e.g.:
#   COPY scripts/ ./scripts/
#   RUN python scripts/ingest_corpus.py
COPY data/ ./data/

EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
