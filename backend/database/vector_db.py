"""
Vector Database — sqlite-vec powered embedding store.

Stores document chunk embeddings in SQLite using the sqlite-vec extension
for fast approximate nearest neighbor (ANN) vector search.
All data lives in a single local .db file — zero external dependencies.
"""

import json
import logging
import sqlite3
import struct
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _serialize_f32(vector: list[float]) -> bytes:
    """Serialize a list of floats into a compact binary format for sqlite-vec.
    
    sqlite-vec expects vectors as packed little-endian float32 bytes.
    
    Args:
        vector: List of float values (the embedding).
        
    Returns:
        Packed bytes representation.
    """
    return struct.pack(f"<{len(vector)}f", *vector)


class VectorDB:
    """SQLite + sqlite-vec vector store for document chunk embeddings.
    
    Provides:
    - Storage of text chunks with metadata
    - Embedding storage and ANN search via sqlite-vec
    - Full-text metadata queries
    """
    
    def __init__(self, db_path: str, embedding_dim: int = 384):
        """Initialize the vector database.
        
        Args:
            db_path: Path to the SQLite database file.
            embedding_dim: Dimension of embedding vectors (384 for bge-small).
        """
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        
        # Ensure parent directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._load_sqlite_vec()
        self._create_tables()
    
    def _load_sqlite_vec(self):
        """Load the sqlite-vec extension."""
        try:
            import sqlite_vec
            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)
            logger.info("sqlite-vec extension loaded successfully")
        except ImportError:
            raise ImportError(
                "sqlite-vec is not installed. Run: pip install sqlite-vec"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load sqlite-vec extension: {e}")
    
    def _create_tables(self):
        """Create the chunks table and the virtual vec0 table for embeddings."""
        cursor = self.conn.cursor()
        
        # Main chunks table (metadata + content)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                filepath TEXT NOT NULL,
                heading_path TEXT DEFAULT '',
                content TEXT NOT NULL,
                char_start INTEGER DEFAULT 0,
                char_end INTEGER DEFAULT 0,
                metadata TEXT DEFAULT '{}'
            )
        """)
        
        # sqlite-vec virtual table for vector search
        cursor.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_embeddings USING vec0(
                chunk_id TEXT PRIMARY KEY,
                embedding FLOAT[{self.embedding_dim}]
            )
        """)
        
        self.conn.commit()
        logger.info("Vector DB tables initialized")
    
    def insert_chunk(
        self,
        chunk_id: str,
        filepath: str,
        heading_path: str,
        content: str,
        embedding: list[float],
        char_start: int = 0,
        char_end: int = 0,
        metadata: Optional[dict] = None,
    ):
        """Insert a single chunk with its embedding.
        
        Args:
            chunk_id: Unique identifier for this chunk.
            filepath: Source file path.
            heading_path: Heading hierarchy (e.g., 'Chapter 3 > Section 1941.1').
            content: The text content of the chunk.
            embedding: Vector embedding of the content.
            char_start: Start character offset in the source document.
            char_end: End character offset in the source document.
            metadata: Additional metadata dict.
        """
        cursor = self.conn.cursor()
        
        try:
            cursor.execute(
                """INSERT OR REPLACE INTO chunks 
                   (chunk_id, filepath, heading_path, content, char_start, char_end, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    chunk_id,
                    filepath,
                    heading_path,
                    content,
                    char_start,
                    char_end,
                    json.dumps(metadata or {}),
                )
            )
            
            cursor.execute(
                """INSERT OR REPLACE INTO chunk_embeddings (chunk_id, embedding)
                   VALUES (?, ?)""",
                (chunk_id, _serialize_f32(embedding))
            )
            
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to insert chunk {chunk_id}: {e}")
            raise
    
    def insert_chunks_batch(
        self,
        chunks: list[dict],
    ):
        """Insert multiple chunks with embeddings in a single transaction.
        
        Args:
            chunks: List of dicts, each containing:
                chunk_id, filepath, heading_path, content, embedding,
                and optionally char_start, char_end, metadata.
        """
        cursor = self.conn.cursor()
        
        try:
            for chunk in chunks:
                cursor.execute(
                    """INSERT OR REPLACE INTO chunks 
                       (chunk_id, filepath, heading_path, content, char_start, char_end, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        chunk["chunk_id"],
                        chunk["filepath"],
                        chunk.get("heading_path", ""),
                        chunk["content"],
                        chunk.get("char_start", 0),
                        chunk.get("char_end", 0),
                        json.dumps(chunk.get("metadata", {})),
                    )
                )
                
                cursor.execute(
                    """INSERT OR REPLACE INTO chunk_embeddings (chunk_id, embedding)
                       VALUES (?, ?)""",
                    (chunk["chunk_id"], _serialize_f32(chunk["embedding"]))
                )
            
            self.conn.commit()
            logger.info(f"Inserted {len(chunks)} chunks into vector DB")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Batch insert failed: {e}")
            raise
    
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[dict]:
        """Search for the most similar chunks to a query embedding.
        
        Args:
            query_embedding: The query vector.
            top_k: Number of results to return.
            
        Returns:
            List of dicts with chunk data and similarity distance.
        """
        cursor = self.conn.cursor()
        
        results = cursor.execute(
            """
            SELECT
                ce.chunk_id,
                ce.distance,
                c.filepath,
                c.heading_path,
                c.content,
                c.char_start,
                c.char_end,
                c.metadata
            FROM chunk_embeddings ce
            JOIN chunks c ON ce.chunk_id = c.chunk_id
            WHERE ce.embedding MATCH ?
                AND k = ?
            ORDER BY ce.distance
            """,
            (_serialize_f32(query_embedding), top_k)
        ).fetchall()
        
        return [
            {
                "chunk_id": row["chunk_id"],
                "distance": row["distance"],
                "filepath": row["filepath"],
                "heading_path": row["heading_path"],
                "content": row["content"],
                "char_start": row["char_start"],
                "char_end": row["char_end"],
                "metadata": json.loads(row["metadata"]),
            }
            for row in results
        ]
    
    def get_chunk(self, chunk_id: str) -> Optional[dict]:
        """Retrieve a specific chunk by ID.
        
        Args:
            chunk_id: The chunk identifier.
            
        Returns:
            Chunk data dict or None if not found.
        """
        cursor = self.conn.cursor()
        row = cursor.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        
        if row is None:
            return None
        
        return {
            "chunk_id": row["chunk_id"],
            "filepath": row["filepath"],
            "heading_path": row["heading_path"],
            "content": row["content"],
            "char_start": row["char_start"],
            "char_end": row["char_end"],
            "metadata": json.loads(row["metadata"]),
        }
    
    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict]:
        """Retrieve multiple chunks by their IDs.
        
        Args:
            chunk_ids: List of chunk identifiers.
            
        Returns:
            List of chunk data dicts.
        """
        if not chunk_ids:
            return []
        
        placeholders = ",".join(["?"] * len(chunk_ids))
        cursor = self.conn.cursor()
        rows = cursor.execute(
            f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})",
            chunk_ids
        ).fetchall()
        
        return [
            {
                "chunk_id": row["chunk_id"],
                "filepath": row["filepath"],
                "heading_path": row["heading_path"],
                "content": row["content"],
                "char_start": row["char_start"],
                "char_end": row["char_end"],
                "metadata": json.loads(row["metadata"]),
            }
            for row in rows
        ]
    
    def count(self) -> int:
        """Return the total number of chunks stored."""
        cursor = self.conn.cursor()
        result = cursor.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return result[0]
    
    def close(self):
        """Close the database connection."""
        self.conn.close()
        logger.info("Vector DB connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
