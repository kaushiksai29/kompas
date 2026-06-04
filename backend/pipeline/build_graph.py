"""
Graph Construction — Typed, Schema-Guided Relations

Takes per-chunk extraction results (entities + typed relations from extract.py)
and builds the knowledge graph:

1. Register entity nodes, deduplicated by canonical (type, name) so the same
   form/benefit referenced in different documents collapses to one node. This
   deduplication is what lets relations span documents — "Form I-693" named in
   the I-485 instructions is the *same* node as the I-693 document's own entity.
2. Create Chunk nodes and MENTIONS edges (Chunk -> Entity) for retrieval.
3. Create typed RELATES edges (Entity -> Entity) from the extracted relations,
   resolving each relation's subject/object name to a registered entity id.

Unlike the previous co-occurrence approach, edges here are directed and carry an
explicit predicate (requires / filed_with / ...) plus the evidence sentence.
"""

import logging

from backend.pipeline.extract import (
    ExtractionResult,
    EntityType,
    generate_entity_id,
    canonicalize_name,
    infer_entity_type,
    _is_gov_form_token,
)
from backend.database.graph_db import GraphDB

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Builds the typed knowledge graph from extraction results."""

    def __init__(self, graph_db: GraphDB):
        self.graph_db = graph_db
        # canonical lower-cased name -> entity_id, for resolving relation endpoints
        self._name_index: dict[str, str] = {}
        # entity_id -> type, used when synthesizing missing endpoints
        self._id_type: dict[str, str] = {}

    def build_from_extractions(
        self,
        extraction_results: dict[str, ExtractionResult],
        chunk_metadata: dict[str, dict] | None = None,
        adjacency_window: int = 1,  # kept for call-site compatibility; unused
    ):
        """Build the full typed knowledge graph from extraction results."""
        logger.info(f"Building typed graph from {len(extraction_results)} chunks")

        self._correct_entity_types(extraction_results)
        self._register_entities(extraction_results)
        self._build_mentions(extraction_results, chunk_metadata or {})
        self._build_relations(extraction_results)

        stats = self.graph_db.stats()
        logger.info(
            "Graph construction complete: "
            f"{stats['entities']} entities, {stats['chunks']} chunks, "
            f"{stats['mentions']} mentions, {stats['relations']} typed relations "
            f"({stats.get('relations_by_predicate', {})})"
        )
        return stats

    # ── Type correction ──────────────────────────────────────────────────────

    def _correct_entity_types(
        self, extraction_results: dict[str, ExtractionResult]
    ) -> None:
        """Re-route entities mistyped as `form` to their true type.

        The LLM (and therefore the on-disk extraction cache) frequently labels
        visa classifications (H-1B, J-1, L-1) and tax forms (W-2, 941) as USCIS
        `form`s. The parse-time guard in extract.py only fires on fresh LLM
        output; rebuilding from cache bypasses it. Applying the correction here,
        at graph-build time, fixes both paths and keeps entity IDs consistent
        across registration, mentions, and relation resolution.
        """
        fixed = 0
        for result in extraction_results.values():
            for e in result.entities:
                if e.entity_type.value == EntityType.FORM.value and not _is_gov_form_token(e.name):
                    e.entity_type = EntityType(infer_entity_type(e.name))
                    fixed += 1
        if fixed:
            logger.info(f"Corrected {fixed} entities mistyped as 'form'")

    # ── Entities ─────────────────────────────────────────────────────────────

    def _register_entities(
        self, extraction_results: dict[str, ExtractionResult]
    ) -> dict[str, dict]:
        """Deduplicate and register all entity nodes; build the name index."""
        registry: dict[str, dict] = {}

        for result in extraction_results.values():
            for entity in result.entities:
                etype = entity.entity_type.value
                canon = canonicalize_name(entity.name, etype)
                eid = generate_entity_id(entity.name, etype)

                if eid not in registry:
                    registry[eid] = {
                        "name": canon,
                        "type": etype,
                        "description": entity.description,
                        "aliases": list(entity.aliases),
                        "mention_count": 0,
                    }
                info = registry[eid]
                if len(entity.description) > len(info["description"]):
                    info["description"] = entity.description
                for alias in entity.aliases:
                    if alias not in info["aliases"]:
                        info["aliases"].append(alias)
                info["mention_count"] += 1

                self._name_index[canon.lower()] = eid
                self._name_index[entity.name.strip().lower()] = eid
                self._id_type[eid] = etype

        for eid, info in registry.items():
            self.graph_db.upsert_entity(
                entity_id=eid,
                name=info["name"],
                entity_type=info["type"],
                description=info["description"],
                aliases=info["aliases"],
            )
        logger.info(f"Registered {len(registry)} unique entities")
        return registry

    # ── Mentions ───────────────────────────────────────────────────────────────

    def _build_mentions(
        self,
        extraction_results: dict[str, ExtractionResult],
        chunk_metadata: dict[str, dict],
    ) -> None:
        """Create Chunk nodes and MENTIONS edges."""
        total = 0
        for chunk_id, result in extraction_results.items():
            meta = chunk_metadata.get(chunk_id, {})
            self.graph_db.upsert_chunk_node(
                chunk_id=chunk_id,
                heading_path=meta.get("heading_path", ""),
                filepath=meta.get("filepath", ""),
            )
            seen: set[str] = set()
            for entity in result.entities:
                eid = generate_entity_id(entity.name, entity.entity_type.value)
                if eid in seen:
                    continue
                seen.add(eid)
                self.graph_db.add_mention(chunk_id, eid)
                total += 1
        logger.info(f"Created {total} MENTIONS edges")

    # ── Typed relations ─────────────────────────────────────────────────────────

    def _resolve_endpoint(self, name: str) -> str | None:
        """Resolve a relation endpoint name to a registered entity id.

        Tries the name index (exact, then type-canonicalized). If the endpoint
        was never emitted in any entity list, synthesize a node with an
        inferred type so the typed edge is never lost — endpoints (especially
        forms) are how the graph connects across documents.
        """
        name = name.strip()
        if not name:
            return None
        key = name.lower()
        if key in self._name_index:
            return self._name_index[key]

        etype = infer_entity_type(name)
        canon = canonicalize_name(name, etype)
        ckey = canon.lower()
        if ckey in self._name_index:
            return self._name_index[ckey]

        eid = generate_entity_id(canon, etype)
        self.graph_db.upsert_entity(
            entity_id=eid, name=canon, entity_type=etype, description="", aliases=[]
        )
        self._name_index[key] = eid
        self._name_index[ckey] = eid
        self._id_type[eid] = etype
        return eid

    def _build_relations(
        self, extraction_results: dict[str, ExtractionResult]
    ) -> None:
        """Create typed RELATES edges from extracted relations."""
        created = 0
        skipped = 0
        for chunk_id, result in extraction_results.items():
            for rel in result.relations:
                sid = self._resolve_endpoint(rel.subject)
                oid = self._resolve_endpoint(rel.object)
                if not sid or not oid or sid == oid:
                    skipped += 1
                    continue
                self.graph_db.add_relation(
                    subject_id=sid,
                    object_id=oid,
                    predicate=rel.predicate.value,
                    evidence=rel.evidence,
                    source_chunk=chunk_id,
                )
                created += 1
        logger.info(f"Created {created} typed RELATES edges ({skipped} unresolved/skipped)")


def build_graph_pipeline(
    chunks: list,
    extraction_results: dict[str, ExtractionResult],
    graph_db: GraphDB,
    adjacency_window: int = 1,
) -> dict:
    """Convenience wrapper: build chunk metadata then build the typed graph."""
    chunk_metadata = {}
    for chunk in chunks:
        cid = getattr(chunk, "chunk_id", None)
        if cid:
            chunk_metadata[cid] = {
                "heading_path": getattr(chunk, "heading_path", ""),
                "filepath": getattr(chunk, "filepath", ""),
            }
    builder = GraphBuilder(graph_db)
    return builder.build_from_extractions(extraction_results, chunk_metadata)
