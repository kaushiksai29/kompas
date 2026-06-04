"""
Dual-Channel Retriever — Hybrid Vector + Graph Search

This is the core retrieval engine that combines two channels:

1. SEMANTIC CHANNEL: Vector similarity search via sqlite-vec
   - Embeds the query using bge-small
   - Retrieves top-k most similar text chunks
   
2. STRUCTURAL CHANNEL: Graph traversal via Kùzu
   - Extracts entities from the query and from top-k chunks
   - Matches entities to graph nodes
   - Traverses 1-2 hop typed RELATES edges to find related entities
   - Pulls the text chunks that mention those related entities

3. CONTEXT FUSION: Merges both channels into a unified context
   - Deduplicates chunks
   - Ranks by a combined score (vector similarity + graph centrality)
   - Formats the context with clear source attribution for citation
"""

import logging
from typing import Optional

from backend.database.vector_db import VectorDB
from backend.database.graph_db import GraphDB
from backend.pipeline.extract import (
    extract_query_entities,
    generate_entity_id,
    ExtractedEntity,
)

logger = logging.getLogger(__name__)


class RetrievalResult:
    """Container for retrieval results from both channels."""
    
    def __init__(self):
        self.vector_chunks: list[dict] = []     # From semantic search
        self.graph_chunks: list[dict] = []      # From graph traversal
        self.merged_chunks: list[dict] = []     # Deduplicated & ranked
        self.subgraph: dict = {"nodes": [], "edges": []}  # Reasoning path
        self.query_entities: list[dict] = []    # Entities extracted from query
        self.matched_entities: list[dict] = []  # Entities found in graph
    
    def to_dict(self) -> dict:
        return {
            "vector_chunks": self.vector_chunks,
            "graph_chunks": self.graph_chunks,
            "merged_chunks": self.merged_chunks,
            "subgraph": self.subgraph,
            "query_entities": self.query_entities,
            "matched_entities": self.matched_entities,
        }


class HybridRetriever:
    """Dual-channel retriever combining vector search and graph traversal.
    
    The key insight: vector search finds relevant text chunks by meaning,
    while graph traversal finds structurally connected entities that
    may not share surface-level similarity with the query but are
    critical for multi-hop reasoning.
    """
    
    def __init__(
        self,
        vector_db: VectorDB,
        graph_db: GraphDB,
        embedding_fn,
        llm_client=None,
    ):
        """Initialize the hybrid retriever.
        
        Args:
            vector_db: sqlite-vec vector store instance.
            graph_db: Kùzu graph database instance.
            embedding_fn: Function that takes a string and returns a list[float].
            llm_client: LLM client for query entity extraction (optional,
                        enables the structural channel).
        """
        self.vector_db = vector_db
        self.graph_db = graph_db
        self.embed = embedding_fn
        self.llm_client = llm_client
    
    def retrieve(
        self,
        query: str,
        top_k_vector: int = 10,
        top_k_graph: int = 10,
        graph_hops: int = 2,
        use_graph: bool = True,
    ) -> RetrievalResult:
        """Execute dual-channel retrieval.
        
        Args:
            query: The user's question.
            top_k_vector: Number of chunks from vector search.
            top_k_graph: Number of additional chunks from graph traversal.
            graph_hops: Number of hops for graph traversal.
            use_graph: Whether to use the structural channel (set False for naive RAG baseline).
            
        Returns:
            RetrievalResult with chunks, subgraph, and metadata.
        """
        result = RetrievalResult()
        
        # ── Channel 1: Semantic Vector Search ────────────────────────────
        logger.info(f"Vector search for: {query[:80]}...")
        query_embedding = self.embed(query)
        vector_results = self.vector_db.search(query_embedding, top_k=top_k_vector)
        result.vector_chunks = vector_results
        
        logger.info(f"Vector channel returned {len(vector_results)} chunks")
        
        if not use_graph or self.llm_client is None:
            # Naive RAG mode — vector search only
            result.merged_chunks = self._format_chunks(vector_results)
            return result
        
        # ── Channel 2: Structural Graph Traversal ────────────────────────
        logger.info("Starting structural graph traversal...")
        
        # Step 2a: Extract entities from the query
        query_entities = extract_query_entities(query, self.llm_client)
        result.query_entities = [
            {"name": e.name, "type": e.entity_type.value}
            for e in query_entities
        ]
        
        # Step 2b: Also pick up FORMS mentioned in the top vector chunks.
        # Forms (e.g. "Form I-864") are precise, unambiguous anchors; scanning
        # for all entity names would inject noise like "son" or "Form W-2".
        chunk_entity_names = set()
        all_entities = self.graph_db.get_all_entities()
        form_names = [
            e["name"] for e in all_entities
            if e.get("type") == "form" and len(e["name"]) >= 5
        ]
        for chunk in vector_results[:5]:  # Only scan top-5 for efficiency
            content_lower = chunk["content"].lower()
            for name in form_names:
                if name.lower() in content_lower:
                    chunk_entity_names.add(name)
        
        # Step 2c: Match query entities + chunk entities to graph nodes
        matched_entity_ids = set()
        
        for entity in query_entities:
            # Try exact match first
            found = self.graph_db.get_entity_by_name(entity.name)
            if found:
                matched_entity_ids.add(found["id"])
                continue

            # Substring fallback only for precise identifiers (forms, statutes);
            # for free-text concepts a substring match is too noisy.
            etype = entity.entity_type.value
            if etype in ("form", "statute") and len(entity.name) >= 4:
                for sr in self.graph_db.search_entities(entity.name, limit=2):
                    matched_entity_ids.add(sr["id"])
        
        for name in chunk_entity_names:
            found = self.graph_db.get_entity_by_name(name)
            if found:
                matched_entity_ids.add(found["id"])
        
        result.matched_entities = [
            self.graph_db.get_entity_by_id(eid)
            for eid in matched_entity_ids
            if self.graph_db.get_entity_by_id(eid) is not None
        ]
        
        logger.info(
            f"Matched {len(matched_entity_ids)} entities in graph "
            f"(from {len(query_entities)} query + {len(chunk_entity_names)} chunk entities)"
        )
        
        if matched_entity_ids:
            # Step 2d: Traverse the graph to find connected entities
            subgraph = self.graph_db.get_k_hop_subgraph(
                list(matched_entity_ids), hops=graph_hops
            )
            result.subgraph = subgraph
            
            # Step 2e: Get chunks that mention the discovered entities.
            # Build entity_id -> set(chunk_ids) once, reused for scoring below.
            entity_to_chunks: dict[str, set[str]] = {}
            graph_chunk_ids: set[str] = set()
            for node in subgraph["nodes"]:
                cids = set(self.graph_db.get_chunks_for_entity(node["id"]))
                entity_to_chunks[node["id"]] = cids
                graph_chunk_ids.update(cids)

            # Remove chunks already found by vector search
            vector_chunk_ids = {c["chunk_id"] for c in vector_results}
            new_chunk_ids = graph_chunk_ids - vector_chunk_ids

            if new_chunk_ids:
                graph_chunks = self.vector_db.get_chunks_by_ids(list(new_chunk_ids))
                # Score each chunk by how many *seed-matched* entities it mentions:
                # chunks tying together more of the query's entities rank higher.
                for chunk in graph_chunks:
                    cid = chunk["chunk_id"]
                    chunk["graph_score"] = sum(
                        1 for eid in matched_entity_ids
                        if cid in entity_to_chunks.get(eid, ())
                    )
                graph_chunks.sort(key=lambda x: x.get("graph_score", 0), reverse=True)
                result.graph_chunks = graph_chunks[:top_k_graph]
            
            logger.info(
                f"Graph channel found {len(result.graph_chunks)} additional chunks "
                f"({len(subgraph['nodes'])} nodes, {len(subgraph['edges'])} edges in subgraph)"
            )
        
        # ── Channel 3: Context Fusion ────────────────────────────────────
        result.merged_chunks = self._merge_channels(
            result.vector_chunks, result.graph_chunks
        )
        
        logger.info(f"Merged context: {len(result.merged_chunks)} total chunks")
        
        return result
    
    def _merge_channels(
        self,
        vector_chunks: list[dict],
        graph_chunks: list[dict],
    ) -> list[dict]:
        """Merge and deduplicate chunks from both channels.
        
        Vector chunks come first (they're directly relevant),
        followed by graph-discovered chunks (structurally connected).
        
        Args:
            vector_chunks: Chunks from semantic search.
            graph_chunks: Chunks from graph traversal.
            
        Returns:
            Deduplicated, ranked list of chunks.
        """
        seen = set()
        merged = []
        
        # Vector chunks first, marked with their source
        for chunk in vector_chunks:
            cid = chunk["chunk_id"]
            if cid not in seen:
                seen.add(cid)
                chunk["retrieval_source"] = "vector"
                merged.append(chunk)
        
        # Graph chunks second
        for chunk in graph_chunks:
            cid = chunk["chunk_id"]
            if cid not in seen:
                seen.add(cid)
                chunk["retrieval_source"] = "graph"
                merged.append(chunk)
        
        return self._format_chunks(merged)
    
    def _format_chunks(self, chunks: list[dict]) -> list[dict]:
        """Format chunks with source attribution for citation.
        
        Adds a 'citation_id' field that the generator can reference.
        
        Args:
            chunks: List of chunk dicts.
            
        Returns:
            Formatted chunks with citation IDs.
        """
        for i, chunk in enumerate(chunks):
            chunk["citation_id"] = f"[{i + 1}]"
        return chunks


def build_context_string(chunks: list[dict]) -> str:
    """Build a formatted context string from retrieved chunks.
    
    This is what gets injected into the generation prompt.
    Each chunk is labeled with a citation ID and source info.
    
    Args:
        chunks: List of formatted chunk dicts from the retriever.
        
    Returns:
        Formatted context string.
    """
    parts = []
    for chunk in chunks:
        citation = chunk.get("citation_id", "")
        source = chunk.get("retrieval_source", "unknown")
        heading = chunk.get("heading_path", "")
        filepath = chunk.get("filepath", "")
        content = chunk.get("content", "")
        
        part = f"""--- Source {citation} [{source}] ---
Document: {filepath}
Section: {heading}
{content}
"""
        parts.append(part)
    
    return "\n".join(parts)
