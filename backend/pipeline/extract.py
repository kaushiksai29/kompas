"""
Schema-Guided Extraction for US Immigration Forms — Entities + Typed Relations

For each text chunk we ask the LLM to return TWO things:

1. ENTITIES — typed nodes (forms, agencies, benefits, eligibility conditions,
   evidence, statutes, applicant categories, fees).
2. RELATIONS — typed, directed edges between those entities, each drawn from a
   fixed predicate vocabulary (requires / filed_with / evidence_for /
   eligibility_for / references / supersedes / applies_to) and justified by an
   evidence sentence from the text.

This is the project's differentiator. Co-occurrence graphs link any two
entities that share a chunk, which largely recovers what vector search already
finds. Typed relations instead capture the *procedural logic* of immigration —
"Form I-485 **requires** Form I-693", "an **approved Form I-130** confers
**eligibility for** adjustment of status" — and those edges connect entities
**across documents**, which is exactly what powers multi-hop retrieval that
naive RAG cannot do.

To raise relation quality we tell the model which form's document the chunk
came from, so an instruction like "submit Form I-693 with your application"
resolves to a relation whose subject is the host form.
"""

import json
import logging
import hashlib
import re
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ─── Entity Schema ───────────────────────────────────────────────────────────

class EntityType(str, Enum):
    """Types of entities we extract from USCIS immigration documents."""
    FORM = "form"                              # e.g., "Form I-485", "Form I-130", "Form I-693"
    AGENCY = "agency"                          # e.g., "USCIS", "National Visa Center", "Department of State"
    BENEFIT_OR_STATUS = "benefit_or_status"    # e.g., "Adjustment of Status", "Lawful Permanent Resident", "Employment Authorization (EAD)", "Advance Parole"
    ELIGIBILITY_CONDITION = "eligibility_condition"  # e.g., "approved immigrant petition", "visa immediately available", "physically present in the U.S.", "admissible"
    EVIDENCE_OR_DOCUMENT = "evidence_or_document"    # e.g., "birth certificate", "passport-style photographs", "Affidavit of Support", "marriage certificate"
    STATUTE = "statute"                        # e.g., "INA section 245", "8 CFR 245.2", "section 213A of the INA"
    APPLICANT_CATEGORY = "applicant_category"  # e.g., "immediate relative", "spouse of a U.S. citizen", "employment-based first preference (EB-1)", "derivative applicant"
    FEE = "fee"                                # e.g., "filing fee", "biometric services fee"


class RelationType(str, Enum):
    """Typed, directed predicates connecting two entities."""
    REQUIRES = "requires"            # FORM/BENEFIT requires FORM/EVIDENCE/CONDITION  ("I-485 requires I-693")
    FILED_WITH = "filed_with"        # FORM filed with AGENCY, or FORM filed together with FORM  ("I-765 filed_with I-485")
    EVIDENCE_FOR = "evidence_for"    # EVIDENCE supports FORM/BENEFIT  ("birth certificate evidence_for I-485")
    ELIGIBILITY_FOR = "eligibility_for"  # CONDITION/CATEGORY confers eligibility for BENEFIT/FORM  ("immediate relative eligibility_for adjustment of status")
    REFERENCES = "references"        # FORM cites STATUTE or another FORM  ("I-485 references INA 245")
    SUPERSEDES = "supersedes"        # newer FORM/edition replaces older one
    APPLIES_TO = "applies_to"        # FORM/BENEFIT applies to APPLICANT_CATEGORY  ("I-130 applies_to spouse of U.S. citizen")


_ALLOWED_PREDICATES = {r.value for r in RelationType}
_ALLOWED_ENTITY_TYPES = {e.value for e in EntityType}


class ExtractedEntity(BaseModel):
    """A single entity extracted from a text chunk."""
    name: str = Field(description="Canonical name of the entity (e.g., 'Form I-485')")
    entity_type: EntityType = Field(description="Type of entity")
    description: str = Field(default="", description="Brief description of what this entity is or does, in context")
    aliases: list[str] = Field(
        default_factory=list,
        description="Alternative names or abbreviations (e.g., ['I-485', 'adjustment application'])",
    )


class ExtractedRelation(BaseModel):
    """A single typed, directed relation between two entities."""
    subject: str = Field(description="Name of the subject entity (must match an extracted entity name)")
    predicate: RelationType = Field(description="Relation type from the fixed vocabulary")
    object: str = Field(description="Name of the object entity (must match an extracted entity name)")
    evidence: str = Field(default="", description="The sentence/phrase from the text that supports this relation")


class ExtractionResult(BaseModel):
    """Result of extraction from a single chunk: entities + typed relations."""
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)


# ─── Extraction Prompt ──────────────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """You are an expert paralegal extracting a knowledge graph from US Citizenship and Immigration Services (USCIS) forms and their instruction booklets.

From a chunk of text you extract TWO things: ENTITIES and typed RELATIONS.

ENTITY TYPES:
1. form — A USCIS or government form, by number. Canonical name format: "Form I-485". Always include the number.
2. agency — A government body (USCIS, National Visa Center, Department of State, Department of Labor, CBP, SSA), a field office, service center, or lockbox.
3. benefit_or_status — An immigration benefit, status, or document a person can obtain (e.g., "Adjustment of Status", "Lawful Permanent Resident", "Employment Authorization Document (EAD)", "Advance Parole").
4. eligibility_condition — A condition that must be true to qualify or proceed (e.g., "approved immigrant petition", "an immigrant visa is immediately available", "physically present in the United States", "admissible to the United States").
5. evidence_or_document — A supporting document or piece of evidence to submit (e.g., "birth certificate", "two passport-style photographs", "Affidavit of Support", "medical examination results").
6. statute — A law or regulation citation (e.g., "INA section 245", "section 213A of the INA", "8 CFR 245").
7. applicant_category — A class of applicant (e.g., "immediate relative", "spouse of a U.S. citizen", "employment-based first preference (EB-1)", "derivative applicant").
8. fee — A fee (e.g., "filing fee", "biometric services fee").

RELATION TYPES (directed: subject -> object). Use ONLY these predicates:
- requires — subject needs object to be filed/satisfied. (form requires form | form requires evidence_or_document | form/benefit requires eligibility_condition)
- filed_with — subject is submitted to / together with object. (form filed_with agency | form filed_with form, e.g. concurrent filing)
- evidence_for — subject is supporting evidence for object. (evidence_or_document evidence_for form/benefit)
- eligibility_for — subject makes someone eligible for object. (eligibility_condition/applicant_category eligibility_for benefit_or_status/form)
- references — subject cites object. (form references statute | form references form)
- supersedes — subject replaces an older object. (form supersedes form)
- applies_to — subject is for object. (form/benefit applies_to applicant_category)

RULES:
- The chunk comes from a known host form (given to you as DOCUMENT). When the text says "you must submit Form X" or "file Form X", the subject is usually that host form. Resolve such implicit subjects to the host form.
- Every relation's subject and object MUST also appear in your entities list (add them if needed).
- Use the most specific canonical name. For forms, always "Form I-NNN".
- Provide the evidence sentence for each relation. Do NOT invent relations the text does not support.
- If the chunk is boilerplate (page numbers, "Form I-485 Instructions 04/01/24"), return empty arrays.
- Return ONLY valid JSON: {"entities": [...], "relations": [...]}."""


def _form_label_from_filepath(filepath: str) -> Optional[str]:
    """Derive a human form label like 'Form I-485' from a source filename.

    'i-485instr.pdf' -> 'Form I-485 (instructions)', 'i-130a.pdf' -> 'Form I-130A'.
    Returns None if the filename doesn't look like a USCIS form slug.
    """
    if not filepath:
        return None
    stem = re.split(r"[\\/]", filepath)[-1].lower()
    stem = stem.rsplit(".", 1)[0]
    is_instr = stem.endswith("instr")
    if is_instr:
        stem = stem[: -len("instr")]
    m = re.match(r"^([a-z]+)-?(\d+)([a-z]?)$", stem)
    if not m:
        return None
    prefix, num, suffix = m.groups()
    label = f"Form {prefix.upper()}-{num}{suffix.upper()}"
    return f"{label} (instructions)" if is_instr else label


def build_extraction_prompt(
    chunk_text: str,
    heading_path: str = "",
    document_context: str = "",
) -> list[dict]:
    """Build the messages list for entity + relation extraction."""
    doc_line = document_context or "Unknown immigration document"
    user_content = f"""Extract entities and typed relations from this text and return JSON.

DOCUMENT: {doc_line}
SECTION (heading path): {heading_path if heading_path else 'Unknown'}

---TEXT START---
{chunk_text}
---TEXT END---

Return a JSON object with an "entities" array and a "relations" array."""

    return [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ─── JSON Schema for Structured Output ──────────────────────────────────────

EXTRACTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "entity_type": {"type": "string", "enum": [e.value for e in EntityType]},
                    "description": {"type": "string"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "entity_type"],
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string", "enum": list(_ALLOWED_PREDICATES)},
                    "object": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["subject", "predicate", "object"],
            },
        },
    },
    "required": ["entities", "relations"],
}


# ─── Entity ID Generation ───────────────────────────────────────────────────

# Light canonicalization so the same form referred to as "I-485", "Form I-485",
# or "Form I 485" collapses to one node.
#
# CRITICAL: a USCIS/government immigration form is "<gov-prefix>-<digits>" where
# the prefix is one of a small, KNOWN set (I, N, G, DS, ETA, AR). A naive
# "<letters>-<digits>" pattern wrongly swallows two other families that look the
# same but are NOT USCIS forms:
#   - visa CLASSIFICATIONS (H-1B, J-1, L-1, O-1, E-2, E-3, TN, F-1, B-2, K-1) —
#     these are applicant categories / statuses, not forms.
#   - tax / IRS forms (W-2, W-4, 941, 943, 1040) — these are supporting
#     documents (evidence), not USCIS forms.
# So _FORM_RE is restricted to the real government-form prefixes, and the other
# two families are detected separately and routed to the correct entity types.

# Government immigration form prefixes that _FORM_RE will accept as type `form`.
_GOV_FORM_PREFIXES = ("I", "N", "G", "DS", "ETA", "AR")

# Only "Form <gov-prefix>-<digits>[suffix]" is a real USCIS/government form.
_FORM_RE = re.compile(
    r"\b(I|N|G|DS|ETA|AR)[-\s]?(\d{1,4})([a-z]?)\b",
    re.IGNORECASE,
)

# Visa classifications / nonimmigrant categories — route to applicant_category.
# Letter(s) + hyphen + digit, optionally with a trailing letter (H-1B, E-2, O-1A),
# plus the two bare two-letter classes TN and TD.
_VISA_CLASS_RE = re.compile(
    r"\b("
    r"[A-Z]-\d[A-Z]?\d?"   # H-1B, J-1, L-1, O-1, O-1A, E-2, E-3, F-1, B-2, K-1, P-1, R-1, ...
    r"|TN|TD"              # TN / TD (no hyphen+digit)
    r")\b"
)

# Tax / IRS forms — route to evidence_or_document. Either a W-style hyphenated
# form (W-2, W-4, W-7, W-2G) or a bare IRS numeric form (1040, 941, 943, 1099).
_TAX_FORM_RE = re.compile(
    r"\b("
    r"W-\d[A-Z]?G?"                     # W-2, W-4, W-7, W-2G
    r"|10(?:40|99)[A-Z]*"              # 1040, 1099 (+ variants like 1040EZ, 1099-MISC stem)
    r"|9(?:41|43|40)"                  # 941, 943, 940
    r")\b"
)


def _is_gov_form_token(name: str) -> bool:
    """True if `name` contains a real USCIS/government form number (I-/N-/G-/DS-/ETA-/AR-)."""
    return _FORM_RE.search(name) is not None


def _is_visa_class(name: str) -> bool:
    """True if `name` looks like a visa classification (H-1B, J-1, TN, ...)."""
    return _VISA_CLASS_RE.search(name) is not None


def _is_tax_form(name: str) -> bool:
    """True if `name` looks like a tax/IRS form (W-2, 941, 1040, ...)."""
    return _TAX_FORM_RE.search(name) is not None


def canonicalize_name(name: str, entity_type: str) -> str:
    """Normalize an entity name for stable, cross-document dedup."""
    n = " ".join(name.strip().split())
    if entity_type == EntityType.FORM.value:
        m = _FORM_RE.search(n)
        if m:
            prefix, num, suffix = m.groups()
            return f"Form {prefix.upper()}-{num}{suffix.upper()}"
    return n


# Keyword cascades for inferring an entity type when the model returns a bare
# string or an invalid type. Small free-tier models frequently emit entities as
# plain names; rather than drop them we type them heuristically so the graph
# stays complete. Order matters — more specific buckets first.
_TYPE_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    (EntityType.STATUTE.value, ("ina ", "8 cfr", "cfr ", "u.s.c", "usc ", "§", "section ", "title 8")),
    (EntityType.AGENCY.value, ("uscis", "national visa center", "department of", "service center",
                               "field office", "lockbox", "consulate", "embassy", "cbp", "social security",
                               " nvc", "uscis.gov", "ombudsman")),
    (EntityType.FEE.value, ("fee",)),
    (EntityType.BENEFIT_OR_STATUS.value, ("adjustment of status", "permanent resident", "green card",
                                          "employment authorization", "advance parole", "naturalization",
                                          "citizenship", "asylum", "refugee", " status", "travel document",
                                          "work permit", "immigrant visa", "lawful")),
    (EntityType.APPLICANT_CATEGORY.value, ("immediate relative", "spouse", "preference", "eb-1", "eb-2",
                                           "eb-3", "derivative", "self-petition", "beneficiary", "petitioner",
                                           "applicant", "principal", "fiancé", "widow")),
    (EntityType.ELIGIBILITY_CONDITION.value, ("eligible", "admissible", "inadmissib", "present in",
                                              "approved", "available", "priority date", "physically",
                                              "must be", "bona fide", "continuous")),
    (EntityType.EVIDENCE_OR_DOCUMENT.value, ("certificate", "photograph", "passport", "record", "letter",
                                             "affidavit", "translation", "evidence", "documentation",
                                             "transcript", "examination", "report", "receipt", "i-94",
                                             "identification", "decree", "statement")),
]


def infer_entity_type(name: str) -> str:
    """Best-effort entity type for a bare name (no declared type).

    Token families that look like "<letters>-<digits>" must be disambiguated:
    only real USCIS/government prefixes (I-/N-/G-/DS-/ETA-/AR-) are `form`;
    visa classifications (H-1B, J-1, L-1, ...) are `applicant_category`; and
    tax/IRS forms (W-2, 941, 1040, ...) are `evidence_or_document`.
    """
    has_section = re.search(r"section|§", name, re.IGNORECASE)
    if _is_gov_form_token(name) and not has_section:
        return EntityType.FORM.value
    # Visa classes and tax forms are checked before the keyword cascade so they
    # are never mistyped as forms or swept into the generic document catch-all.
    if _is_visa_class(name) and not has_section:
        return EntityType.APPLICANT_CATEGORY.value
    if _is_tax_form(name) and not has_section:
        return EntityType.EVIDENCE_OR_DOCUMENT.value
    low = name.lower()
    for etype, kws in _TYPE_KEYWORDS:
        if any(kw in low for kw in kws):
            return etype
    # Broadest catch-all: most untyped nouns in these booklets are documents.
    return EntityType.EVIDENCE_OR_DOCUMENT.value


def generate_entity_id(name: str, entity_type: str) -> str:
    """Generate a deterministic ID from canonicalized (type, name).

    Ensures the same entity extracted from different chunks/documents maps to
    the same graph node — the basis for cross-document multi-hop edges.
    """
    canon = canonicalize_name(name, entity_type)
    normalized = f"{entity_type}::{canon.lower().strip()}"
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


# ─── Extraction Engine ──────────────────────────────────────────────────────

class EntityExtractor:
    """Extracts entities and typed relations from text chunks via an LLM."""

    def __init__(self, llm_client):
        self.llm = llm_client
        self._extraction_cache: dict[str, ExtractionResult] = {}

    def extract_entities(
        self,
        chunk_text: str,
        heading_path: str = "",
        document_context: str = "",
        use_cache: bool = True,
    ) -> ExtractionResult:
        """Extract entities + relations from a single text chunk."""
        cache_key = hashlib.md5(
            (document_context + "::" + chunk_text).encode()
        ).hexdigest()
        if use_cache and cache_key in self._extraction_cache:
            return self._extraction_cache[cache_key]

        messages = build_extraction_prompt(chunk_text, heading_path, document_context)

        try:
            response_text = self.llm.generate_sync(
                messages=messages,
                json_schema=EXTRACTION_JSON_SCHEMA,
            )
            result = self._parse_response(response_text)
            if use_cache:
                self._extraction_cache[cache_key] = result
            logger.info(
                f"Extracted {len(result.entities)} entities, "
                f"{len(result.relations)} relations from chunk "
                f"({len(chunk_text)} chars, doc: {document_context[:40]})"
            )
            return result
        except Exception as e:
            logger.error(f"Extraction failed for chunk {cache_key[:8]}: {e}")
            return ExtractionResult()

    def extract_from_chunks(
        self,
        chunks: list,  # list[DocumentChunk] - loose typing to avoid circular import
        batch_size: int = 5,
        show_progress: bool = True,
    ) -> dict[str, ExtractionResult]:
        """Extract entities + relations from multiple chunks."""
        results: dict[str, ExtractionResult] = {}
        total = len(chunks)

        for i, chunk in enumerate(chunks):
            chunk_id = getattr(chunk, "chunk_id", str(i))
            heading_path = getattr(chunk, "heading_path", "")
            content = getattr(chunk, "content", str(chunk))
            filepath = getattr(chunk, "filepath", "")
            doc_ctx = _form_label_from_filepath(filepath) or ""

            results[chunk_id] = self.extract_entities(content, heading_path, doc_ctx)

            if show_progress and (i + 1) % batch_size == 0:
                logger.info(f"Extraction progress: {i + 1}/{total} chunks")

        total_entities = sum(len(r.entities) for r in results.values())
        total_relations = sum(len(r.relations) for r in results.values())
        unique_names = {
            e.name.lower().strip()
            for r in results.values()
            for e in r.entities
        }
        logger.info(
            f"Extraction complete: {total_entities} entity mentions "
            f"({len(unique_names)} unique), {total_relations} relations "
            f"from {total} chunks"
        )
        return results

    async def aextract_entities(
        self,
        chunk_text: str,
        heading_path: str = "",
        document_context: str = "",
    ) -> ExtractionResult:
        """Async single-chunk extraction (uses the client's async API)."""
        messages = build_extraction_prompt(chunk_text, heading_path, document_context)
        try:
            response_text = await self.llm.generate(
                messages, json_schema=EXTRACTION_JSON_SCHEMA
            )
            return self._parse_response(response_text)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Async extraction failed: {e}")
            return ExtractionResult()

    async def extract_from_chunks_async(
        self,
        chunks: list,
        concurrency: int = 8,
        progress_cb=None,
    ) -> dict[str, ExtractionResult]:
        """Concurrently extract from many chunks over a single event loop.

        A bounded semaphore caps in-flight requests; the fallback client
        handles rate limits by switching providers. Far faster than the
        sequential path for hundreds of chunks.
        """
        import asyncio

        sem = asyncio.Semaphore(concurrency)
        done = 0
        total = len(chunks)

        async def work(idx: int, chunk) -> tuple[str, ExtractionResult]:
            nonlocal done
            chunk_id = getattr(chunk, "chunk_id", str(idx))
            content = getattr(chunk, "content", str(chunk))
            heading = getattr(chunk, "heading_path", "")
            doc_ctx = _form_label_from_filepath(getattr(chunk, "filepath", "")) or ""
            async with sem:
                res = await self.aextract_entities(content, heading, doc_ctx)
            done += 1
            if progress_cb and done % 25 == 0:
                progress_cb(done, total)
            return chunk_id, res

        pairs = await asyncio.gather(
            *(work(i, c) for i, c in enumerate(chunks))
        )
        return dict(pairs)

    def _parse_response(self, response_text: str) -> ExtractionResult:
        """Parse the LLM response into entities + relations, robustly."""
        text = response_text.strip()

        if text.startswith("```"):
            lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}") + 1
            if 0 <= start < end:
                try:
                    data = json.loads(text[start:end])
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse JSON: {text[:200]}...")
                    return ExtractionResult()
            else:
                logger.warning(f"No JSON found: {text[:200]}...")
                return ExtractionResult()

        entities = self._parse_entities(data.get("entities", []))
        valid_names = {e.name for e in entities}
        relations = self._parse_relations(data.get("relations", []), valid_names)
        return ExtractionResult(entities=entities, relations=relations)

    def _parse_entities(self, raw_entities: list) -> list[ExtractedEntity]:
        entities: list[ExtractedEntity] = []
        for raw in raw_entities:
            # Models sometimes emit entities as bare strings ("Form I-485").
            if isinstance(raw, str):
                name = raw.strip()
                if name:
                    entities.append(ExtractedEntity(name=name, entity_type=infer_entity_type(name)))
                continue
            if not isinstance(raw, dict):
                continue
            raw_name = raw.get("name") or raw.get("value", "")
            if not raw_name:
                continue
            raw_name = str(raw_name).strip()
            raw_type = str(raw.get("entity_type") or raw.get("category") or raw.get("type", "")).lower().strip().replace(" ", "_")
            if raw_type not in _ALLOWED_ENTITY_TYPES:
                raw_type = infer_entity_type(raw_name)  # invalid/missing -> infer
            elif raw_type == EntityType.FORM.value and not _is_gov_form_token(raw_name):
                # The model loves to label visa classes (H-1B, J-1) and tax
                # forms (W-2, 941) as USCIS `form`s. Only a real government
                # form prefix (I-/N-/G-/DS-/ETA-/AR-) is a form; re-infer the
                # rest so they land in applicant_category / evidence_or_document.
                raw_type = infer_entity_type(raw_name)
            entities.append(
                ExtractedEntity(
                    name=raw_name,
                    entity_type=raw_type,
                    description=raw.get("description", "") or "",
                    aliases=raw.get("aliases", []) or [],
                )
            )
        return entities

    def _parse_relations(
        self, raw_relations: list, valid_names: set[str]
    ) -> list[ExtractedRelation]:
        relations: list[ExtractedRelation] = []
        for raw in raw_relations:
            if not isinstance(raw, dict):
                continue
            try:
                pred = str(raw.get("predicate") or raw.get("relation") or "").lower().strip()
                subj = str(raw.get("subject") or raw.get("source") or "").strip()
                obj = str(raw.get("object") or raw.get("target") or "").strip()
                if pred not in _ALLOWED_PREDICATES or not subj or not obj or subj == obj:
                    continue
                relations.append(
                    ExtractedRelation(
                        subject=subj,
                        predicate=pred,
                        object=obj,
                        evidence=raw.get("evidence", "") or "",
                    )
                )
            except (KeyError, ValueError) as e:
                logger.debug(f"Skipping malformed relation {raw}: {e}")
        return relations


# ─── Query Entity Extraction ────────────────────────────────────────────────

QUERY_EXTRACTION_PROMPT = """You are analyzing a user's question about US immigration forms and processes.
Extract the key entities mentioned or implied in their question. These will be matched against a knowledge graph of USCIS forms, benefits, eligibility conditions, evidence, and agencies.

Focus on:
- Forms mentioned (e.g., Form I-130, Form I-485) — always use "Form I-NNN".
- Immigration benefits or statuses (green card, adjustment of status, EAD, advance parole, naturalization).
- Applicant categories (spouse of U.S. citizen, EB-2, derivative).
- Eligibility conditions or evidence the user describes.
- Agencies (USCIS, NVC).

Return a JSON object with an "entities" array (and an empty "relations" array)."""


def extract_query_entities(query: str, llm_client) -> list[ExtractedEntity]:
    """Extract entities from a user query for graph search."""
    messages = [
        {"role": "system", "content": QUERY_EXTRACTION_PROMPT},
        {"role": "user", "content": f"Extract entities from this question:\n\n{query}"},
    ]
    try:
        response_text = llm_client.generate_sync(
            messages=messages, json_schema=EXTRACTION_JSON_SCHEMA
        )
        extractor = EntityExtractor(llm_client)
        result = extractor._parse_response(response_text)
        logger.info(f"Extracted {len(result.entities)} entities from query")
        return result.entities
    except Exception as e:
        logger.error(f"Query entity extraction failed: {e}")
        return []
