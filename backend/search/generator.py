"""
Citation-Grounded Answer Generator

Takes the merged context from the dual-channel retriever and generates
a comprehensive, citation-grounded answer using the LLM.

Every claim in the answer must reference a specific source using
citation tags like [1], [2], etc. The system prompt strictly enforces
this constraint to prevent hallucination.
"""

import json
import logging
import os
from typing import Optional

from backend.search.retriever import build_context_string

logger = logging.getLogger(__name__)


# ─── System Prompt ───────────────────────────────────────────────────────────

GENERATION_SYSTEM_PROMPT = """You are an expert US immigration research assistant. You answer questions about USCIS forms and procedures accurately and comprehensively, grounded in official form instructions, with EXACT citations to the source documents.

RULES — follow these strictly:
1. ONLY use information from the provided source documents. Do NOT use your general knowledge or assume facts not present.
2. EVERY factual claim MUST include a citation tag (e.g., [1], [2]) referencing the specific source.
3. If the sources do not contain enough information to answer, say so explicitly. Do NOT guess or fabricate filing requirements, fees, or deadlines.
4. Cite the most specific source: prefer a specific form-instruction section or statute over a general overview.
5. When multiple sources cover the same point, cite ALL relevant sources.
6. Pay attention to the "Knowledge Graph" section: it lists typed relationships (e.g. "Form I-485 requires Form I-693") discovered across documents. Use these to give complete, multi-step procedural answers — e.g. which prerequisite forms, supporting evidence, and agencies are involved — but still cite the underlying source text for each claim.
7. Structure your answer clearly with:
   - A direct answer to the question
   - Required forms (use the exact form number, e.g. "Form I-864")
   - Supporting evidence or documents to submit
   - Eligibility conditions that must be met
   - Where/with whom it is filed, and any fees or deadlines mentioned
8. Add a brief disclaimer that this is general information, not legal advice, when the question seeks guidance on an individual's case.

FORMAT:
- Use markdown for structure (headers, bullets, bold for key terms and form numbers)
- Place citation tags immediately after the claim they support
- End with a "Sources" section listing each citation with its document and section"""


GENERATION_USER_TEMPLATE = """Answer this US immigration question using ONLY the provided sources.

QUESTION: {query}

SOURCES:
{context}

Remember: cite every claim with [N] tags, name exact form numbers, and if the sources don't cover the question, say so."""


def generate_answer(
    query: str,
    retrieved_chunks: list[dict],
    llm_client,
    subgraph: Optional[dict] = None,
) -> dict:
    """Generate a citation-grounded answer from retrieved context.
    
    Args:
        query: The user's question.
        retrieved_chunks: Chunks from the retriever (with citation_ids).
        llm_client: LLM client instance.
        subgraph: Optional graph subgraph for additional context.
        
    Returns:
        Dict with:
            - answer: The generated markdown answer text
            - citations: List of citation dicts mapping [N] to sources
            - model_used: Which model generated the answer
    """
    if not retrieved_chunks:
        return {
            "answer": "I couldn't find any relevant information in the loaded documents to answer this question. Please make sure the relevant regulatory documents have been ingested.",
            "citations": [],
            "model_used": "none",
        }
    
    # Build context string from retrieved chunks
    context = build_context_string(retrieved_chunks)
    
    # Optionally add graph context
    if subgraph and subgraph.get("nodes"):
        graph_context = _format_graph_context(subgraph)
        context += f"\n\n--- Graph Context (Related Entities) ---\n{graph_context}"
    
    # Build messages
    user_message = GENERATION_USER_TEMPLATE.format(
        query=query,
        context=context,
    )
    
    messages = [
        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    
    try:
        answer_text = llm_client.generate_sync(messages=messages)
        
        # Build citation mapping
        citations = _build_citation_map(retrieved_chunks)
        
        logger.info(f"Generated answer ({len(answer_text)} chars) with {len(citations)} citations")
        
        return {
            "answer": answer_text,
            "citations": citations,
            "chunks_used": len(retrieved_chunks),
        }
        
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        return {
            "answer": f"An error occurred while generating the answer: {str(e)}",
            "citations": [],
            "chunks_used": 0,
            "model_used": "error",
        }


def _format_graph_context(subgraph: dict) -> str:
    """Format the graph subgraph as additional context for the LLM.
    
    This gives the model awareness of entity relationships discovered
    through graph traversal.
    
    Args:
        subgraph: Dict with 'nodes' and 'edges'.
        
    Returns:
        Formatted string describing the entity relationships.
    """
    lines = ["The following entities and relationships were found in the knowledge graph:\n"]
    
    # Group nodes by type
    nodes_by_type: dict[str, list] = {}
    for node in subgraph["nodes"]:
        ntype = node.get("type", "unknown")
        if ntype not in nodes_by_type:
            nodes_by_type[ntype] = []
        nodes_by_type[ntype].append(node)
    
    for ntype, nodes in nodes_by_type.items():
        lines.append(f"\n{ntype.upper()}:")
        for node in nodes:
            desc = node.get("description", "")
            name = node.get("name", "Unknown")
            if desc:
                lines.append(f"  - {name}: {desc}")
            else:
                lines.append(f"  - {name}")
    
    # Describe typed relationships (the structural reasoning path).
    if subgraph["edges"]:
        lines.append("\nTYPED RELATIONSHIPS (subject — predicate → object):")
        # Strongest (most-asserted) relationships first.
        sorted_edges = sorted(
            subgraph["edges"],
            key=lambda e: e.get("weight", 0),
            reverse=True,
        )
        node_map = {n["id"]: n.get("name", n["id"]) for n in subgraph["nodes"]}

        for edge in sorted_edges[:20]:  # Top 20 relationships
            src = node_map.get(edge["source"], edge["source"])
            tgt = node_map.get(edge["target"], edge["target"])
            pred = edge.get("predicate", "related to").replace("_", " ")
            line = f"  - {src} — {pred} → {tgt}"
            evidence = (edge.get("evidence") or "").strip()
            if evidence:
                line += f'   (\"{evidence[:160]}\")'
            lines.append(line)

    return "\n".join(lines)


def _uscis_url(filepath: str) -> str:
    """Derive the online USCIS source URL from a chunk's local filepath.

    Maps the filepath basename's stem (e.g. "i-485instr") to the official
    USCIS forms document URL so citations point to an authoritative,
    clickable online source rather than a local C:\\ path.

    Args:
        filepath: Local filepath of the source document.

    Returns:
        The USCIS document URL, or "" if no filepath was given.
    """
    if not filepath:
        return ""
    base = os.path.basename(filepath.replace("\\", "/"))
    stem = os.path.splitext(base)[0]
    if not stem:
        return ""
    return f"https://www.uscis.gov/sites/default/files/document/forms/{stem}.pdf"


def _build_citation_map(chunks: list[dict]) -> list[dict]:
    """Build a citation mapping from citation IDs to source info.

    Args:
        chunks: Retrieved chunks with citation_id fields.

    Returns:
        List of citation dicts.
    """
    citations = []
    for chunk in chunks:
        filepath = chunk.get("filepath", "")
        source_url = _uscis_url(filepath)

        citation = {
            "citation_id": chunk.get("citation_id", ""),
            "chunk_id": chunk.get("chunk_id", ""),
            "filepath": filepath,
            "heading_path": chunk.get("heading_path", ""),
            "retrieval_source": chunk.get("retrieval_source", "unknown"),
            "preview": chunk.get("content", "")[:200] + "...",
            "source_url": source_url,
        }

        # PyMuPDF stores the source page as metadata.page_number.
        page = chunk.get("metadata", {}).get("page_number")
        if page is not None:
            citation["page"] = page
            if source_url:
                citation["source_url_with_page"] = f"{source_url}#page={page}"

        citations.append(citation)
    return citations
