"""Unit tests for backend.database.graph_db.GraphDB.

Covers the typed RELATES edge behavior:
  * ``add_relation`` increments weight on a duplicate (subject, predicate,
    object) instead of creating a parallel edge, and keeps first-seen
    evidence.
  * ``get_k_hop_subgraph`` returns directed, typed edges (orientation
    preserved) and respects the ``max_nodes`` cap and the hub-stop rule
    (hub nodes are included but not expanded through).

RESOURCE RULE: every GraphDB here is built in a fresh tempfile dir via the
``tmp_graph_dir`` fixture — never the real ``data/graphdb`` owned by the
extractor.
"""

from __future__ import annotations

import pytest

from backend.database.graph_db import GraphDB


# ─── Fixtures / helpers ─────────────────────────────────────────────────────


@pytest.fixture
def db(tmp_graph_dir):
    """A fresh, empty GraphDB on a temporary path."""
    graph = GraphDB(tmp_graph_dir)
    try:
        yield graph
    finally:
        graph.close()


def _add_entity(db: GraphDB, eid: str, name: str, etype: str = "form") -> None:
    db.upsert_entity(eid, name, etype, description="", aliases=[])


def _relation_rows(db: GraphDB, subject_id: str, object_id: str, predicate: str):
    """Return raw RELATES rows for one (subject, predicate, object) triple."""
    result = db.conn.execute(
        """
        MATCH (s:Entity {id: $sid})-[r:RELATES {predicate: $p}]->(o:Entity {id: $oid})
        RETURN r.weight AS weight, r.evidence AS evidence
        """,
        {"sid": subject_id, "p": predicate, "oid": object_id},
    )
    return result.get_as_df().to_dict("records")


# ─── add_relation: weight increment on duplicate ────────────────────────────


def test_add_relation_creates_edge_weight_one(db):
    _add_entity(db, "e1", "Form I-485")
    _add_entity(db, "e2", "Form I-693")
    db.add_relation("e1", "e2", "requires", evidence="first evidence", source_chunk="c1")

    rows = _relation_rows(db, "e1", "e2", "requires")
    assert len(rows) == 1
    assert rows[0]["weight"] == 1.0
    assert rows[0]["evidence"] == "first evidence"


def test_add_relation_duplicate_increments_weight(db):
    _add_entity(db, "e1", "Form I-485")
    _add_entity(db, "e2", "Form I-693")
    db.add_relation("e1", "e2", "requires", evidence="first", source_chunk="c1")
    db.add_relation("e1", "e2", "requires", evidence="second", source_chunk="c2")
    db.add_relation("e1", "e2", "requires", evidence="third", source_chunk="c3")

    rows = _relation_rows(db, "e1", "e2", "requires")
    assert len(rows) == 1, "duplicate triple must not create a parallel edge"
    assert rows[0]["weight"] == 3.0
    # First-seen evidence is retained (ON CREATE only).
    assert rows[0]["evidence"] == "first"


def test_add_relation_distinct_predicate_separate_edge(db):
    _add_entity(db, "e1", "Form I-485")
    _add_entity(db, "e2", "Form I-693")
    db.add_relation("e1", "e2", "requires", source_chunk="c1")
    db.add_relation("e1", "e2", "references", source_chunk="c2")

    assert len(_relation_rows(db, "e1", "e2", "requires")) == 1
    assert len(_relation_rows(db, "e1", "e2", "references")) == 1
    # Each starts at weight 1.0 — they are independent edges.
    assert _relation_rows(db, "e1", "e2", "requires")[0]["weight"] == 1.0
    assert _relation_rows(db, "e1", "e2", "references")[0]["weight"] == 1.0


def test_add_relation_ignores_self_loop(db):
    _add_entity(db, "e1", "Form I-485")
    db.add_relation("e1", "e1", "requires", source_chunk="c1")
    rows = _relation_rows(db, "e1", "e1", "requires")
    assert rows == []


# ─── get_k_hop_subgraph: directed typed edges ──────────────────────────────


def test_k_hop_returns_directed_typed_edge(db):
    _add_entity(db, "e1", "Form I-485")
    _add_entity(db, "e2", "Form I-693")
    db.add_relation("e1", "e2", "requires", evidence="needs medical exam", source_chunk="c1")

    sub = db.get_k_hop_subgraph(["e1"], hops=1)
    node_ids = {n["id"] for n in sub["nodes"]}
    assert {"e1", "e2"} <= node_ids

    assert len(sub["edges"]) == 1
    edge = sub["edges"][0]
    # Direction must be subject -> object regardless of which end seeded.
    assert edge["source"] == "e1"
    assert edge["target"] == "e2"
    assert edge["predicate"] == "requires"
    assert edge["weight"] == 1.0
    assert edge["evidence"] == "needs medical exam"


def test_k_hop_direction_preserved_when_seed_is_object(db):
    _add_entity(db, "e1", "Form I-485")
    _add_entity(db, "e2", "Form I-693")
    db.add_relation("e1", "e2", "requires", source_chunk="c1")

    # Seed from the OBJECT end; edge must still read e1 -> e2.
    sub = db.get_k_hop_subgraph(["e2"], hops=1)
    assert len(sub["edges"]) == 1
    edge = sub["edges"][0]
    assert edge["source"] == "e1"
    assert edge["target"] == "e2"
    assert edge["predicate"] == "requires"


def test_k_hop_empty_seed_returns_empty(db):
    assert db.get_k_hop_subgraph([], hops=2) == {"nodes": [], "edges": []}


def test_k_hop_two_hop_traversal(db):
    # Chain: e1 -requires-> e2 -references-> e3. Two hops from e1 reaches e3.
    for i in (1, 2, 3):
        _add_entity(db, f"e{i}", f"Entity {i}")
    db.add_relation("e1", "e2", "requires", source_chunk="c1")
    db.add_relation("e2", "e3", "references", source_chunk="c2")

    sub = db.get_k_hop_subgraph(["e1"], hops=2)
    node_ids = {n["id"] for n in sub["nodes"]}
    assert {"e1", "e2", "e3"} <= node_ids
    preds = {(e["source"], e["predicate"], e["target"]) for e in sub["edges"]}
    assert ("e1", "requires", "e2") in preds
    assert ("e2", "references", "e3") in preds


# ─── get_k_hop_subgraph: budget / hub-stop ──────────────────────────────────


def test_k_hop_respects_max_nodes(db):
    # A star: hub "h" connected to many leaves. Cap the subgraph size.
    _add_entity(db, "h", "USCIS", etype="agency")
    n_leaves = 20
    for i in range(n_leaves):
        lid = f"leaf{i}"
        _add_entity(db, lid, f"Form X-{i}")
        db.add_relation("h", lid, "filed_with", source_chunk=f"c{i}")

    sub = db.get_k_hop_subgraph(["h"], hops=2, max_nodes=5)
    assert len(sub["nodes"]) <= 5
    # Every edge's endpoints must be present among the returned nodes.
    node_ids = {n["id"] for n in sub["nodes"]}
    for e in sub["edges"]:
        assert e["source"] in node_ids
        assert e["target"] in node_ids


def test_k_hop_hub_not_expanded_through(db):
    # Seed -> hub, and hub -> many other leaves. The hub has high incident
    # degree, so with a low hub_degree it should be included but NOT expanded
    # through: the far leaves stay out of the subgraph.
    _add_entity(db, "seed", "Form I-130")
    _add_entity(db, "hub", "USCIS", etype="agency")
    db.add_relation("seed", "hub", "filed_with", source_chunk="c0")

    for i in range(10):
        lid = f"far{i}"
        _add_entity(db, lid, f"Form F-{i}")
        db.add_relation("hub", lid, "filed_with", source_chunk=f"cf{i}")

    sub = db.get_k_hop_subgraph(
        ["seed"], hops=2, max_nodes=60, per_node_fanout=10, hub_degree=3
    )
    node_ids = {n["id"] for n in sub["nodes"]}
    assert "seed" in node_ids
    assert "hub" in node_ids
    # The hub (degree 11 > hub_degree 3) must not have been expanded through,
    # so its far leaves are absent.
    assert not any(nid.startswith("far") for nid in node_ids)
