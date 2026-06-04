"""
One-command corpus ingestion: PDFs -> clean chunks -> vectors + typed graph.

Pipeline:
  1. Parse + chunk each PDF (PyMuPDF), drop boilerplate/fragment chunks.
  2. Embed chunks (bge-small, local) and store in the sqlite-vec vector DB.
  3. Extract entities + typed relations per chunk (LLM, concurrent), with an
     on-disk cache so re-runs and rate-limit interruptions are cheap.
  4. Build the typed knowledge graph (deduped entities, RELATES edges).
  5. Print stats and a few sample cross-document edges.

Usage:
    .venv/Scripts/python.exe scripts/ingest_corpus.py                 # instruction booklets
    .venv/Scripts/python.exe scripts/ingest_corpus.py --glob "*.pdf"  # everything
    .venv/Scripts/python.exe scripts/ingest_corpus.py --limit 50      # quick smoke run
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import get_settings
from backend.llm_client import create_fallback_client
from backend.database.vector_db import VectorDB
from backend.database.graph_db import GraphDB
from backend.pipeline.ingest import ingest_file, clean_chunks
from backend.pipeline.extract import (
    EntityExtractor,
    ExtractionResult,
    _form_label_from_filepath,
)
from backend.pipeline.build_graph import GraphBuilder

CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "extractions.json"


def _cache_key(doc_ctx: str, content: str) -> str:
    return hashlib.md5(f"{doc_ctx}::{content}".encode()).hexdigest()


def load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="*instr.pdf", help="glob within data/sources/immigration")
    ap.add_argument("--limit", type=int, default=0, help="cap total chunks (smoke test)")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    settings = get_settings()
    src_dir = settings.sources_dir / "immigration"
    files = sorted(src_dir.glob(args.glob))
    if not files:
        print(f"No files matched {args.glob} in {src_dir}")
        return 1

    # ── 1. Parse + chunk + clean ──────────────────────────────────────────────
    print(f"Parsing {len(files)} PDFs from {src_dir} ...")
    chunks = []
    for f in files:
        fchunks = clean_chunks(ingest_file(str(f)))
        print(f"  {f.name:18} -> {len(fchunks):4d} clean chunks")
        chunks.extend(fchunks)
    if args.limit:
        chunks = chunks[: args.limit]
    print(f"Total clean chunks: {len(chunks)}\n")

    # ── 2. Embed + store vectors ──────────────────────────────────────────────
    print("Loading embedding model (bge-small, fastembed/ONNX) ...")
    from backend.embeddings import embed_many, load

    load()
    vdb = VectorDB(db_path=str(settings.resolved_sqlite_path), embedding_dim=384)

    print(f"Embedding {len(chunks)} chunks ...")
    embeddings = embed_many([c.content for c in chunks])
    for c, emb in zip(chunks, embeddings):
        vdb.insert_chunk(
            chunk_id=c.chunk_id,
            filepath=c.filepath,
            heading_path=c.heading_path,
            content=c.content,
            embedding=emb,
            char_start=c.char_start,
            char_end=c.char_end,
            metadata=c.metadata,
        )
    print(f"Vector store now holds {vdb.count()} chunks.\n")

    # ── 3. Extract entities + relations (cached + concurrent) ─────────────────
    cache = load_cache()

    results: dict[str, ExtractionResult] = {}
    to_extract = []
    for c in chunks:
        doc_ctx = _form_label_from_filepath(c.filepath) or ""
        key = _cache_key(doc_ctx, c.content)
        if key in cache:
            results[c.chunk_id] = ExtractionResult(**cache[key])
        else:
            to_extract.append(c)

    print(f"Extraction: {len(results)} cached, {len(to_extract)} to extract "
          f"(concurrency={args.concurrency}) ...")

    if to_extract:
        # Only *uncached* chunks need the LLM. A from-cache rebuild (e.g. the
        # container image build, which has no API key) skips this gracefully.
        try:
            client = create_fallback_client(settings)
            extractor = EntityExtractor(client)
            t0 = time.time()
            new = asyncio.run(
                extractor.extract_from_chunks_async(
                    to_extract,
                    concurrency=args.concurrency,
                    progress_cb=lambda d, t: print(f"  ... {d}/{t} extracted", flush=True),
                )
            )
            for c in to_extract:
                r = new.get(c.chunk_id, ExtractionResult())
                results[c.chunk_id] = r
                doc_ctx = _form_label_from_filepath(c.filepath) or ""
                cache[_cache_key(doc_ctx, c.content)] = r.model_dump(mode="json")
            save_cache(cache)
            print(f"Extraction done in {time.time() - t0:.0f}s "
                  f"({len(to_extract)} new calls).\n")
        except Exception as e:  # noqa: BLE001
            print(f"  ! extraction unavailable ({str(e)[:80]}); using cache only.")
            for c in to_extract:
                results.setdefault(c.chunk_id, ExtractionResult())

    ent_total = sum(len(r.entities) for r in results.values())
    rel_total = sum(len(r.relations) for r in results.values())
    print(f"Extracted {ent_total} entity mentions, {rel_total} relations.\n")

    # ── 4. Build typed graph ──────────────────────────────────────────────────
    gdb = GraphDB(db_path=str(settings.resolved_kuzu_path))
    chunk_meta = {
        c.chunk_id: {"heading_path": c.heading_path, "filepath": c.filepath}
        for c in chunks
    }
    builder = GraphBuilder(gdb)
    stats = builder.build_from_extractions(results, chunk_meta)
    print("\n=== GRAPH STATS ===")
    print(json.dumps(stats, indent=2))

    # ── 5. Sample cross-document edges ────────────────────────────────────────
    print("\n=== SAMPLE 'requires' EDGES (Form -> Form) ===")
    try:
        df = gdb.conn.execute(
            """
            MATCH (s:Entity)-[r:RELATES]->(o:Entity)
            WHERE s.type = 'form' AND o.type = 'form' AND r.predicate = 'requires'
            RETURN s.name AS s, o.name AS o, r.weight AS w
            ORDER BY w DESC LIMIT 15
            """
        ).get_as_df()
        for _, row in df.iterrows():
            print(f"  {row['s']} --requires--> {row['o']}  (x{int(row['w'])})")
    except Exception as e:  # noqa: BLE001
        print("  (query failed:", e, ")")

    gdb.close()
    vdb.close()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
