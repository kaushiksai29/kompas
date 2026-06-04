"""
FastAPI Application — GraphRAG API Server

Exposes endpoints for:
- /query — Run a GraphRAG query (or naive RAG with ?mode=naive)
- /ingest — Trigger document ingestion
- /graph — Get the full knowledge graph for visualization
- /stats — Get system statistics
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import get_settings
from backend.llm_client import create_fallback_client
from backend.database.vector_db import VectorDB
from backend.database.graph_db import GraphDB
from backend.pipeline.ingest import ingest_directory, ingest_file
from backend.pipeline.extract import EntityExtractor
from backend.pipeline.build_graph import GraphBuilder
from backend.search.retriever import HybridRetriever
from backend.search.generator import generate_answer

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Global State ────────────────────────────────────────────────────────────

class AppState:
    """Holds initialized database connections and clients."""
    vector_db: Optional[VectorDB] = None
    graph_db: Optional[GraphDB] = None
    llm_client = None
    embedding_fn = None
    retriever: Optional[HybridRetriever] = None
    settings = None


state = AppState()


def _load_embedding_model():
    """Load the local fastembed (ONNX) embedding model — no PyTorch."""
    from backend.embeddings import embed_one, load

    logger.info(f"Loading embedding model: {state.settings.embedding_model}")
    load()
    return embed_one


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize databases and models on startup, cleanup on shutdown."""
    logger.info("Initializing GraphRAG server...")
    
    state.settings = get_settings()
    
    # Initialize databases
    state.vector_db = VectorDB(
        db_path=str(state.settings.resolved_sqlite_path),
        embedding_dim=384,  # bge-small dimension
    )
    state.graph_db = GraphDB(
        db_path=str(state.settings.resolved_kuzu_path),
    )
    
    # Initialize LLM client
    try:
        state.llm_client = create_fallback_client()
        logger.info("LLM client initialized with fallback chain")
    except Exception as e:
        logger.warning(f"LLM client init failed (will work without generation): {e}")
    
    # Load embedding model
    try:
        state.embedding_fn = _load_embedding_model()
        logger.info("Embedding model loaded")
    except Exception as e:
        logger.warning(f"Embedding model load failed: {e}")
    
    # Initialize retriever
    if state.embedding_fn:
        state.retriever = HybridRetriever(
            vector_db=state.vector_db,
            graph_db=state.graph_db,
            embedding_fn=state.embedding_fn,
            llm_client=state.llm_client,
        )
    
    logger.info("GraphRAG server ready!")
    
    yield
    
    # Cleanup
    if state.vector_db:
        state.vector_db.close()
    if state.graph_db:
        state.graph_db.close()
    
    logger.info("GraphRAG server shut down")


# ─── FastAPI App ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="GraphRAG — US Immigration Forms Navigator",
    description="Citation-grounded immigration Q&A with dual-channel retrieval (vector + typed knowledge graph)",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response Models ───────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(description="The user's question about US immigration forms")
    mode: str = Field(
        default="hybrid",
        description="Retrieval mode: 'hybrid' (GraphRAG) or 'naive' (vector-only RAG)"
    )
    top_k: int = Field(default=10, description="Number of chunks to retrieve")
    graph_hops: int = Field(default=2, description="Graph traversal depth")


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict]
    subgraph: dict
    query_entities: list[dict]
    matched_entities: list[dict]
    chunks_used: int
    retrieval_mode: str


class IngestRequest(BaseModel):
    path: str = Field(description="Path to a file or directory to ingest")


class IngestResponse(BaseModel):
    chunks_created: int
    entities_extracted: int
    graph_stats: dict


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Health check and system info."""
    stats = {}
    if state.vector_db:
        stats["chunks"] = state.vector_db.count()
    if state.graph_db:
        try:
            stats["graph"] = state.graph_db.stats()
        except Exception:
            stats["graph"] = {}
    
    return {
        "service": "GraphRAG — US Immigration Forms Navigator",
        "status": "running",
        "stats": stats,
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Run a GraphRAG query and return a citation-grounded answer.
    
    Supports two modes:
    - 'hybrid': Full dual-channel retrieval (vector + graph) — this is GraphRAG
    - 'naive': Vector-only retrieval — the baseline for comparison
    """
    if not state.retriever:
        raise HTTPException(
            status_code=503,
            detail="Retriever not initialized. Ensure embedding model is loaded."
        )
    
    if not state.llm_client:
        raise HTTPException(
            status_code=503,
            detail="LLM client not initialized. Check your API keys in .env"
        )
    
    use_graph = request.mode == "hybrid"
    
    # Retrieve
    retrieval_result = state.retriever.retrieve(
        query=request.query,
        top_k_vector=request.top_k,
        top_k_graph=request.top_k,
        graph_hops=request.graph_hops,
        use_graph=use_graph,
    )
    
    # Generate
    gen_result = generate_answer(
        query=request.query,
        retrieved_chunks=retrieval_result.merged_chunks,
        llm_client=state.llm_client,
        subgraph=retrieval_result.subgraph if use_graph else None,
    )
    
    return QueryResponse(
        answer=gen_result["answer"],
        citations=gen_result["citations"],
        subgraph=retrieval_result.subgraph,
        query_entities=retrieval_result.query_entities,
        matched_entities=retrieval_result.matched_entities,
        chunks_used=gen_result["chunks_used"],
        retrieval_mode=request.mode,
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """Ingest documents from a file or directory.
    
    Runs the full pipeline: parse → chunk → embed → extract entities → build graph.
    """
    if not state.embedding_fn:
        raise HTTPException(status_code=503, detail="Embedding model not loaded")
    
    if not state.llm_client:
        raise HTTPException(status_code=503, detail="LLM client not initialized")
    
    path = Path(request.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {request.path}")
    
    # Step 1: Parse and chunk
    logger.info(f"Ingesting from: {path}")
    if path.is_dir():
        chunks = ingest_directory(str(path))
    else:
        chunks = ingest_file(str(path))
    
    if not chunks:
        return IngestResponse(chunks_created=0, entities_extracted=0, graph_stats={})
    
    logger.info(f"Parsed {len(chunks)} chunks")
    
    # Step 2: Embed and store in vector DB
    for chunk in chunks:
        embedding = state.embedding_fn(chunk.content)
        state.vector_db.insert_chunk(
            chunk_id=chunk.chunk_id,
            filepath=chunk.filepath,
            heading_path=chunk.heading_path,
            content=chunk.content,
            embedding=embedding,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            metadata=chunk.metadata,
        )
    
    logger.info(f"Embedded and stored {len(chunks)} chunks in vector DB")
    
    # Step 3: Extract entities
    extractor = EntityExtractor(state.llm_client)
    extraction_results = extractor.extract_from_chunks(chunks, batch_size=5)
    
    total_entities = sum(len(r.entities) for r in extraction_results.values())
    logger.info(f"Extracted {total_entities} entity mentions")
    
    # Step 4: Build knowledge graph
    builder = GraphBuilder(state.graph_db)
    chunk_metadata = {
        chunk.chunk_id: {
            "heading_path": chunk.heading_path,
            "filepath": chunk.filepath,
        }
        for chunk in chunks
    }
    graph_stats = builder.build_from_extractions(
        extraction_results, chunk_metadata, adjacency_window=1
    )
    
    return IngestResponse(
        chunks_created=len(chunks),
        entities_extracted=total_entities,
        graph_stats=graph_stats,
    )


@app.get("/graph")
async def get_graph():
    """Get the full knowledge graph for frontend visualization."""
    if not state.graph_db:
        raise HTTPException(status_code=503, detail="Graph DB not initialized")
    
    return state.graph_db.get_full_graph()


@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    stats = {"vector_db": {}, "graph_db": {}}
    
    if state.vector_db:
        stats["vector_db"]["total_chunks"] = state.vector_db.count()
    
    if state.graph_db:
        try:
            stats["graph_db"] = state.graph_db.stats()
        except Exception:
            pass
    
    return stats


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=settings.backend_port,
        reload=True,
    )
