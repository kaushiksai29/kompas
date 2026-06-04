"""
Graph Database — Kùzu-powered knowledge graph store.

Stores entities and their **typed, schema-guided relationships** in an
embedded Kùzu graph database. Supports Cypher queries for multi-hop
graph traversal.

Unlike a co-occurrence graph (where any two entities sharing a chunk get
linked), every edge here carries an explicit predicate extracted from the
source text — e.g. ``I-485 --requires--> I-693`` — together with the
evidence sentence that justifies it. This is what lets the structural
channel connect entities *across documents* (the I-485 instructions name
Form I-693, which is its own document) and what makes the hybrid retriever
beat naive RAG on procedural, multi-hop questions.

Schema:
- Node table: Entity (id, name, type, description, aliases)
- Node table: Chunk (id, heading_path, filepath)
- Rel table: MENTIONS (Chunk → Entity)
- Rel table: RELATES (Entity → Entity, predicate, evidence, source_chunk, weight)
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class GraphDB:
    """Kùzu embedded graph database for the knowledge graph.
    
    Stores a bipartite Chunk↔Entity graph (MENTIONS) and a directed,
    typed Entity→Entity relation graph (RELATES) for multi-hop traversal.
    """
    
    def __init__(self, db_path: str):
        """Initialize the Kùzu graph database.
        
        Args:
            db_path: Directory path for the Kùzu database files.
        """
        try:
            import kuzu
        except ImportError:
            raise ImportError("kuzu is not installed. Run: pip install kuzu")
        
        self.db_path = db_path
        # Kùzu 0.11+ manages its own directory — only ensure the parent exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.db = kuzu.Database(db_path)
        self.conn = kuzu.Connection(self.db)
        self._create_schema()
    
    def _create_schema(self):
        """Create the graph schema (node and relationship tables)."""
        # ── Node Tables ──
        self._execute_safe("""
            CREATE NODE TABLE IF NOT EXISTS Entity (
                id STRING,
                name STRING,
                type STRING,
                description STRING,
                aliases STRING,
                PRIMARY KEY (id)
            )
        """)
        
        self._execute_safe("""
            CREATE NODE TABLE IF NOT EXISTS Chunk (
                id STRING,
                heading_path STRING,
                filepath STRING,
                PRIMARY KEY (id)
            )
        """)
        
        # ── Relationship Tables ──
        # Chunk mentions an Entity
        self._execute_safe("""
            CREATE REL TABLE IF NOT EXISTS MENTIONS (
                FROM Chunk TO Entity
            )
        """)
        
        # Entity is related to another Entity via a typed predicate.
        # `predicate` is one of the schema relation types (requires,
        # filed_with, evidence_for, eligibility_for, references,
        # supersedes, applies_to). `evidence` is the supporting sentence;
        # `source_chunk` is the chunk the relation was extracted from;
        # `weight` accumulates how many times the relation was asserted.
        self._execute_safe("""
            CREATE REL TABLE IF NOT EXISTS RELATES (
                FROM Entity TO Entity,
                predicate STRING,
                evidence STRING,
                source_chunk STRING,
                weight DOUBLE
            )
        """)

        logger.info("Kùzu graph schema initialized")
    
    def _execute_safe(self, query: str, params: Optional[dict] = None):
        """Execute a Cypher query, catching and logging errors.
        
        Args:
            query: Cypher query string.
            params: Optional parameter dict.
        """
        try:
            if params:
                self.conn.execute(query, params)
            else:
                self.conn.execute(query)
        except Exception as e:
            # Ignore "already exists" errors during schema creation
            err_msg = str(e).lower()
            if "already exists" in err_msg or "duplicate" in err_msg:
                pass
            else:
                logger.error(f"Kùzu query failed: {query[:100]}... Error: {e}")
                raise
    
    # ── Insert Operations ────────────────────────────────────────────────────
    
    def upsert_entity(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        description: str = "",
        aliases: Optional[list[str]] = None,
    ):
        """Insert or update an entity node.
        
        Args:
            entity_id: Deterministic hash ID for the entity.
            name: Canonical entity name.
            entity_type: Entity type (form, agency, benefit_or_status, etc.)
            description: Brief description.
            aliases: List of alternative names.
        """
        aliases_str = json.dumps(aliases or [])
        
        try:
            # Try to merge (upsert)
            self.conn.execute(
                """
                MERGE (e:Entity {id: $id})
                SET e.name = $name, e.type = $type, 
                    e.description = $description, e.aliases = $aliases
                """,
                {
                    "id": entity_id,
                    "name": name,
                    "type": entity_type,
                    "description": description,
                    "aliases": aliases_str,
                }
            )
        except Exception as e:
            logger.warning(f"Entity upsert failed for {name}: {e}")
    
    def upsert_chunk_node(
        self,
        chunk_id: str,
        heading_path: str = "",
        filepath: str = "",
    ):
        """Insert or update a chunk node in the graph.
        
        Args:
            chunk_id: The chunk identifier.
            heading_path: Document heading hierarchy.
            filepath: Source file path.
        """
        try:
            self.conn.execute(
                """
                MERGE (c:Chunk {id: $id})
                SET c.heading_path = $heading_path, c.filepath = $filepath
                """,
                {
                    "id": chunk_id,
                    "heading_path": heading_path,
                    "filepath": filepath,
                }
            )
        except Exception as e:
            logger.warning(f"Chunk node upsert failed for {chunk_id}: {e}")
    
    def add_mention(self, chunk_id: str, entity_id: str):
        """Add a MENTIONS edge from a Chunk to an Entity.
        
        Args:
            chunk_id: Source chunk ID.
            entity_id: Target entity ID.
        """
        try:
            self.conn.execute(
                """
                MATCH (c:Chunk {id: $chunk_id}), (e:Entity {id: $entity_id})
                MERGE (c)-[:MENTIONS]->(e)
                """,
                {"chunk_id": chunk_id, "entity_id": entity_id}
            )
        except Exception as e:
            logger.warning(f"MENTIONS edge failed {chunk_id} -> {entity_id}: {e}")
    
    def add_relation(
        self,
        subject_id: str,
        object_id: str,
        predicate: str,
        evidence: str = "",
        source_chunk: str = "",
    ):
        """Add or reinforce a typed RELATES edge between two entities.

        The edge is keyed on (subject, predicate, object): asserting the
        same relation again from another chunk increments its weight rather
        than creating a duplicate. The first-seen evidence sentence is kept.

        Args:
            subject_id: Source entity ID (the relation's subject).
            object_id: Target entity ID (the relation's object).
            predicate: Relation type (e.g. "requires", "filed_with").
            evidence: Supporting sentence from the source text.
            source_chunk: Chunk ID the relation was extracted from.
        """
        if subject_id == object_id:
            return  # no self-loops
        try:
            self.conn.execute(
                """
                MATCH (s:Entity {id: $sid}), (o:Entity {id: $oid})
                MERGE (s)-[r:RELATES {predicate: $predicate}]->(o)
                ON CREATE SET r.weight = 1.0,
                              r.evidence = $evidence,
                              r.source_chunk = $source_chunk
                ON MATCH SET r.weight = r.weight + 1.0
                """,
                {
                    "sid": subject_id,
                    "oid": object_id,
                    "predicate": predicate,
                    "evidence": evidence[:500],
                    "source_chunk": source_chunk,
                },
            )
        except Exception as e:
            logger.warning(
                f"RELATES edge failed {subject_id} -[{predicate}]-> {object_id}: {e}"
            )
    
    # ── Query Operations ─────────────────────────────────────────────────────
    
    def get_entity_by_name(self, name: str) -> Optional[dict]:
        """Find an entity by its canonical name (case-insensitive).
        
        Args:
            name: Entity name to search for.
            
        Returns:
            Entity dict or None.
        """
        result = self.conn.execute(
            """
            MATCH (e:Entity)
            WHERE lower(e.name) = lower($name)
            RETURN e.id, e.name, e.type, e.description, e.aliases
            LIMIT 1
            """,
            {"name": name}
        )
        
        rows = result.get_as_df()
        if rows.empty:
            return None
        
        row = rows.iloc[0]
        return {
            "id": row["e.id"],
            "name": row["e.name"],
            "type": row["e.type"],
            "description": row["e.description"],
            "aliases": json.loads(row["e.aliases"]) if row["e.aliases"] else [],
        }
    
    def search_entities(self, query: str, limit: int = 10) -> list[dict]:
        """Search for entities by name substring match.
        
        Args:
            query: Search string.
            limit: Max results.
            
        Returns:
            List of matching entity dicts.
        """
        result = self.conn.execute(
            """
            MATCH (e:Entity)
            WHERE contains(lower(e.name), lower($query))
               OR contains(lower(e.aliases), lower($query))
            RETURN e.id, e.name, e.type, e.description, e.aliases
            LIMIT $limit
            """,
            {"query": query, "limit": limit}
        )
        
        rows = result.get_as_df()
        return [
            {
                "id": row["e.id"],
                "name": row["e.name"],
                "type": row["e.type"],
                "description": row["e.description"],
                "aliases": json.loads(row["e.aliases"]) if row["e.aliases"] else [],
            }
            for _, row in rows.iterrows()
        ]
    
    def _incident_relations(self, ids: list[str]) -> list[dict]:
        """Return every typed RELATES edge with an endpoint in `ids`.

        Direction is preserved (subject → object) regardless of which end
        was in the seed set, so the caller always knows the true predicate
        orientation (e.g. "I-485 requires I-693", never the reverse).
        """
        if not ids:
            return []
        try:
            result = self.conn.execute(
                """
                MATCH (s:Entity)-[r:RELATES]->(o:Entity)
                WHERE s.id IN $ids OR o.id IN $ids
                RETURN s.id AS src_id, s.name AS src_name, s.type AS src_type,
                       o.id AS tgt_id, o.name AS tgt_name, o.type AS tgt_type,
                       r.predicate AS predicate, r.weight AS weight,
                       r.evidence AS evidence
                """,
                {"ids": ids},
            )
            df = result.get_as_df()
        except Exception as e:  # noqa: BLE001
            logger.error(f"Incident-relations query failed: {e}")
            return []
        return df.to_dict("records")

    def get_k_hop_subgraph(
        self,
        entity_ids: list[str],
        hops: int = 2,
        max_nodes: int = 60,
        per_node_fanout: int = 10,
        hub_degree: int = 40,
    ) -> dict:
        """Retrieve a focused k-hop subgraph around given entities.

        This is the core of the structural retrieval channel. A naive k-hop
        sweep explodes through hub nodes (e.g. "USCIS" touches hundreds of
        entities), drowning the relevant reasoning path in noise. To stay
        precise we run a *budgeted* breadth-first expansion:

        - From each frontier node we follow only its top `per_node_fanout`
          highest-weight incident edges.
        - "Hub" nodes (incident degree > `hub_degree`) are included but NOT
          expanded through — they are not informative bridges.
        - Total nodes are capped at `max_nodes`.

        Args:
            entity_ids: Seed entity IDs (the query's matched entities).
            hops: Number of hops to traverse (1 or 2 recommended).
            max_nodes: Global cap on subgraph size.
            per_node_fanout: Max edges expanded per node (highest weight first).
            hub_degree: Nodes above this incident degree are not expanded.

        Returns:
            Dict with 'nodes' and typed 'edges' (source, target, predicate,
            weight, evidence).
        """
        if not entity_ids:
            return {"nodes": [], "edges": []}

        nodes: dict[str, dict] = {}
        edges: dict[tuple[str, str, str], dict] = {}

        for eid in entity_ids:
            entity = self.get_entity_by_id(eid)
            if entity:
                nodes[eid] = {"id": eid, "name": entity["name"], "type": entity["type"]}

        frontier = list(dict.fromkeys(entity_ids))  # de-dup, keep order
        visited: set[str] = set(entity_ids)

        for _ in range(max(1, hops)):
            if not frontier or len(nodes) >= max_nodes:
                break
            next_frontier: list[str] = []

            for node_id in frontier:
                rows = self._incident_relations([node_id])
                # A hub: keep the node, record edges to already-known nodes,
                # but don't expand outward through it.
                is_hub = len(rows) > hub_degree
                # Highest-weight edges first.
                rows.sort(key=lambda r: float(r.get("weight") or 0.0), reverse=True)
                taken = 0
                for row in rows:
                    src_id, tgt_id = row["src_id"], row["tgt_id"]
                    other = tgt_id if src_id == node_id else src_id
                    if is_hub and other not in nodes:
                        continue  # don't grow the graph through a hub
                    if taken >= per_node_fanout and other not in nodes:
                        continue
                    for nid, nm, ty in (
                        (src_id, row["src_name"], row["src_type"]),
                        (tgt_id, row["tgt_name"], row["tgt_type"]),
                    ):
                        if nid not in nodes and len(nodes) < max_nodes:
                            nodes[nid] = {"id": nid, "name": nm, "type": ty}
                        if nid not in visited and nid in nodes:
                            next_frontier.append(nid)
                            visited.add(nid)
                    if other in nodes:
                        key = (src_id, row["predicate"], tgt_id)
                        if key not in edges:
                            edges[key] = {
                                "source": src_id,
                                "target": tgt_id,
                                "predicate": row["predicate"],
                                "weight": float(row["weight"]) if row["weight"] else 1.0,
                                "evidence": row.get("evidence", "") or "",
                            }
                        if not is_hub:
                            taken += 1
            frontier = next_frontier

        return {"nodes": list(nodes.values()), "edges": list(edges.values())}
    
    def get_entity_by_id(self, entity_id: str) -> Optional[dict]:
        """Retrieve an entity by its ID.
        
        Args:
            entity_id: Entity hash ID.
            
        Returns:
            Entity dict or None.
        """
        result = self.conn.execute(
            """
            MATCH (e:Entity {id: $id})
            RETURN e.id, e.name, e.type, e.description, e.aliases
            """,
            {"id": entity_id}
        )
        
        rows = result.get_as_df()
        if rows.empty:
            return None
        
        row = rows.iloc[0]
        return {
            "id": row["e.id"],
            "name": row["e.name"],
            "type": row["e.type"],
            "description": row.get("e.description", ""),
            "aliases": json.loads(row["e.aliases"]) if row.get("e.aliases") else [],
        }
    
    def get_chunks_for_entity(self, entity_id: str) -> list[str]:
        """Get all chunk IDs that mention a given entity.
        
        Args:
            entity_id: The entity ID.
            
        Returns:
            List of chunk IDs.
        """
        result = self.conn.execute(
            """
            MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {id: $id})
            RETURN c.id
            """,
            {"id": entity_id}
        )
        
        df = result.get_as_df()
        return df["c.id"].tolist() if not df.empty else []
    
    def get_all_entities(self) -> list[dict]:
        """Retrieve all entities in the graph.
        
        Returns:
            List of entity dicts.
        """
        result = self.conn.execute(
            """
            MATCH (e:Entity)
            RETURN e.id, e.name, e.type, e.description, e.aliases
            """
        )
        
        df = result.get_as_df()
        return [
            {
                "id": row["e.id"],
                "name": row["e.name"],
                "type": row["e.type"],
                "description": row.get("e.description", ""),
                "aliases": json.loads(row["e.aliases"]) if row.get("e.aliases") else [],
            }
            for _, row in df.iterrows()
        ]
    
    def get_full_graph(self) -> dict:
        """Export the entire typed entity→entity relation graph.

        Useful for frontend visualization.

        Returns:
            Dict with 'nodes' and 'edges' lists. Each edge carries its
            predicate so the UI can label and colour edges by relation type.
        """
        nodes = self.get_all_entities()

        result = self.conn.execute(
            """
            MATCH (e1:Entity)-[r:RELATES]->(e2:Entity)
            RETURN e1.id AS source, e2.id AS target,
                   r.predicate AS predicate, r.weight AS weight
            """
        )

        df = result.get_as_df()
        edges = [
            {
                "source": row["source"],
                "target": row["target"],
                "predicate": row["predicate"],
                "weight": float(row["weight"]) if row["weight"] else 1.0,
            }
            for _, row in df.iterrows()
        ]

        return {"nodes": nodes, "edges": edges}
    
    def stats(self) -> dict:
        """Get graph statistics."""
        entity_count = self.conn.execute(
            "MATCH (e:Entity) RETURN count(e) AS cnt"
        ).get_as_df()["cnt"].iloc[0]
        
        chunk_count = self.conn.execute(
            "MATCH (c:Chunk) RETURN count(c) AS cnt"
        ).get_as_df()["cnt"].iloc[0]
        
        mention_count = self.conn.execute(
            "MATCH ()-[r:MENTIONS]->() RETURN count(r) AS cnt"
        ).get_as_df()["cnt"].iloc[0]
        
        relation_count = self.conn.execute(
            "MATCH ()-[r:RELATES]->() RETURN count(r) AS cnt"
        ).get_as_df()["cnt"].iloc[0]

        # Breakdown by predicate (handy for eval / debugging).
        pred_df = self.conn.execute(
            "MATCH ()-[r:RELATES]->() RETURN r.predicate AS p, count(r) AS c ORDER BY c DESC"
        ).get_as_df()
        by_predicate = {row["p"]: int(row["c"]) for _, row in pred_df.iterrows()}

        return {
            "entities": int(entity_count),
            "chunks": int(chunk_count),
            "mentions": int(mention_count),
            "relations": int(relation_count),
            "relations_by_predicate": by_predicate,
        }
    
    def close(self):
        """Close the database connection."""
        # Kùzu connections don't need explicit close, but we log it
        logger.info("Graph DB connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
