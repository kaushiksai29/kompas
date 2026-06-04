"""
Application configuration loaded from environment variables via Pydantic Settings.

Reads from a .env file at the project root (one level above backend/).
"""

from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings
from pydantic import Field

# ---------------------------------------------------------------------------
# Resolve the project root so relative DB paths work regardless of cwd.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Central configuration for the GraphRAG backend."""

    # ── LLM provider API keys ─────────────────────────────────────────────
    groq_api_key: str = Field(default="", description="Groq API key")
    cerebras_api_key: str = Field(default="", description="Cerebras API key")
    gemini_api_key: str = Field(default="", description="Google Gemini API key")

    # ── Model identifiers ─────────────────────────────────────────────────
    groq_model: str = Field(
        default="meta-llama/llama-4-scout-17b-16e-instruct",
        description="Model name for Groq inference",
    )
    cerebras_model: str = Field(
        default="llama-3.3-70b",
        description="Model name for Cerebras inference",
    )
    gemini_model: str = Field(
        default="gemini-2.5-flash",
        description="Model name for Gemini inference",
    )

    # ── Embedding ─────────────────────────────────────────────────────────
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="HuggingFace sentence-transformer model for local embeddings",
    )

    # ── Database paths ────────────────────────────────────────────────────
    sqlite_db_path: str = Field(
        default="./data/vectorstore.db",
        description="Path to the SQLite + sqlite-vec vector store",
    )
    kuzu_db_path: str = Field(
        default="./data/graphdb",
        description="Directory for the Kùzu graph database",
    )

    # ── Server ────────────────────────────────────────────────────────────
    backend_port: int = Field(default=8000, description="Port for the FastAPI server")

    # ── Derived helpers (not loaded from env) ─────────────────────────────
    @property
    def project_root(self) -> Path:
        """Return the resolved project root directory."""
        return _PROJECT_ROOT

    @property
    def data_dir(self) -> Path:
        """Return the project-level data directory, creating it if needed."""
        d = _PROJECT_ROOT / "data"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def sources_dir(self) -> Path:
        """Return the directory where seed/source documents are stored."""
        d = self.data_dir / "sources"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def resolved_sqlite_path(self) -> Path:
        """Return the absolute path for the SQLite DB."""
        p = Path(self.sqlite_db_path)
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def resolved_kuzu_path(self) -> Path:
        """Return the absolute path for the Kùzu graph DB directory."""
        p = Path(self.kuzu_db_path)
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        # Kùzu 0.11+ creates its own directory — only ensure parent exists
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    model_config = {
        "env_file": str(_PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of the application settings."""
    return Settings()
