"""Shared pytest fixtures for the GraphRAG test suite.

These tests cover pure/unit behaviors only — no network, no LLM calls.
Any test that constructs a real on-disk DB (GraphDB) MUST use a fresh
temporary directory via the ``tmp_graph_dir`` fixture; the real database
under ``data/`` is owned exclusively by the ingestion/extraction process
and must never be touched from the test suite.
"""

from __future__ import annotations

import sys
import tempfile
import shutil
from pathlib import Path

import pytest

# Ensure the project root is importable as the ``backend`` package root,
# regardless of the directory pytest is invoked from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@pytest.fixture
def tmp_graph_dir():
    """Yield a fresh temporary directory for a Kùzu GraphDB.

    NEVER points at ``data/graphdb`` — the real DB is owned by the
    extractor. The directory (and the Kùzu files Kùzu creates beside it)
    is removed on teardown.
    """
    base = tempfile.mkdtemp(prefix="graphrag_test_graphdb_")
    db_path = str(Path(base) / "graph")
    try:
        yield db_path
    finally:
        shutil.rmtree(base, ignore_errors=True)
