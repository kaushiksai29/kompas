"""Unit tests for backend.pipeline.extract — pure extraction helpers.

Covers entity-type inference, name canonicalization, deterministic ID
generation, and ``EntityExtractor._parse_response`` parsing of raw JSON
(including bare-string entities and relations). No LLM is ever called:
``_parse_response`` works purely on JSON strings, and the extractor is
constructed with a stub client that raises if used.
"""

from __future__ import annotations

import json

import pytest

from backend.pipeline.extract import (
    EntityExtractor,
    EntityType,
    RelationType,
    canonicalize_name,
    generate_entity_id,
    infer_entity_type,
)


# ─── A stub LLM client that must never be invoked ───────────────────────────


class _ExplodingLLM:
    """Stand-in LLM client. Any network/generation call raises immediately."""

    def generate_sync(self, *args, **kwargs):  # noqa: D401
        raise AssertionError("LLM should not be called in these tests")

    async def generate(self, *args, **kwargs):
        raise AssertionError("LLM should not be called in these tests")


@pytest.fixture
def extractor() -> EntityExtractor:
    return EntityExtractor(_ExplodingLLM())


# ─── infer_entity_type ──────────────────────────────────────────────────────


def test_infer_form_from_number():
    assert infer_entity_type("Form I-485") == EntityType.FORM.value
    assert infer_entity_type("I-130") == EntityType.FORM.value
    assert infer_entity_type("N-400") == EntityType.FORM.value


def test_infer_statute_not_misread_as_form():
    # Contains a number but is a statute citation — must NOT be a form.
    assert infer_entity_type("INA section 245") == EntityType.STATUTE.value
    assert infer_entity_type("8 CFR 245.2") == EntityType.STATUTE.value
    assert infer_entity_type("section 213A of the INA") == EntityType.STATUTE.value


def test_infer_agency():
    assert infer_entity_type("USCIS") == EntityType.AGENCY.value
    assert infer_entity_type("National Visa Center") == EntityType.AGENCY.value
    assert infer_entity_type("Department of State") == EntityType.AGENCY.value


def test_infer_benefit_or_status():
    assert infer_entity_type("Adjustment of Status") == EntityType.BENEFIT_OR_STATUS.value
    assert infer_entity_type("Lawful Permanent Resident") == EntityType.BENEFIT_OR_STATUS.value
    assert infer_entity_type("Employment Authorization") == EntityType.BENEFIT_OR_STATUS.value


def test_infer_fee():
    assert infer_entity_type("filing fee") == EntityType.FEE.value
    assert infer_entity_type("biometric services fee") == EntityType.FEE.value


def test_infer_applicant_category():
    assert infer_entity_type("immediate relative") == EntityType.APPLICANT_CATEGORY.value
    assert infer_entity_type("spouse of a U.S. citizen") == EntityType.APPLICANT_CATEGORY.value


def test_infer_evidence_catch_all():
    # Unknown noun in these booklets defaults to evidence_or_document.
    assert infer_entity_type("birth certificate") == EntityType.EVIDENCE_OR_DOCUMENT.value
    assert infer_entity_type("some unclassifiable phrase") == EntityType.EVIDENCE_OR_DOCUMENT.value


# ─── canonicalize_name ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw",
    ["I-485", "Form I-485", "Form I 485", "form i-485", "I 485"],
)
def test_canonicalize_form_variants_collapse(raw):
    assert canonicalize_name(raw, EntityType.FORM.value) == "Form I-485"


def test_canonicalize_form_with_suffix():
    assert canonicalize_name("i-130a", EntityType.FORM.value) == "Form I-130A"


def test_canonicalize_non_form_passthrough_whitespace_normalized():
    # Non-form types are only whitespace-normalized, not reformatted.
    assert (
        canonicalize_name("  USCIS   National   Benefits  ", EntityType.AGENCY.value)
        == "USCIS National Benefits"
    )


# ─── generate_entity_id ─────────────────────────────────────────────────────


def test_entity_id_is_deterministic():
    a = generate_entity_id("Form I-485", EntityType.FORM.value)
    b = generate_entity_id("Form I-485", EntityType.FORM.value)
    assert a == b
    assert len(a) == 16


def test_entity_id_stable_across_form_aliases():
    # The whole point: "I-485" and "Form I-485" must map to one node.
    a = generate_entity_id("I-485", EntityType.FORM.value)
    b = generate_entity_id("Form I-485", EntityType.FORM.value)
    c = generate_entity_id("form i 485", EntityType.FORM.value)
    assert a == b == c


def test_entity_id_differs_by_type():
    form_id = generate_entity_id("Form I-485", EntityType.FORM.value)
    doc_id = generate_entity_id("Form I-485", EntityType.EVIDENCE_OR_DOCUMENT.value)
    assert form_id != doc_id


def test_entity_id_differs_by_name():
    assert generate_entity_id("Form I-485", EntityType.FORM.value) != generate_entity_id(
        "Form I-130", EntityType.FORM.value
    )


# ─── _parse_response: bare-string entities ──────────────────────────────────


def test_parse_response_bare_string_entities(extractor):
    raw = json.dumps(
        {
            "entities": ["Form I-485", "USCIS", "birth certificate"],
            "relations": [],
        }
    )
    result = extractor._parse_response(raw)
    by_name = {e.name: e.entity_type for e in result.entities}
    assert by_name["Form I-485"] == EntityType.FORM
    assert by_name["USCIS"] == EntityType.AGENCY
    assert by_name["birth certificate"] == EntityType.EVIDENCE_OR_DOCUMENT
    assert result.relations == []


def test_parse_response_bare_string_blank_skipped(extractor):
    raw = json.dumps({"entities": ["", "   ", "USCIS"], "relations": []})
    result = extractor._parse_response(raw)
    assert [e.name for e in result.entities] == ["USCIS"]


# ─── _parse_response: dict entities with invalid/missing types ──────────────


def test_parse_response_invalid_entity_type_inferred(extractor):
    raw = json.dumps(
        {
            "entities": [
                {"name": "Form I-693", "entity_type": "nonsense_type"},
                {"name": "USCIS"},  # missing type
            ],
            "relations": [],
        }
    )
    result = extractor._parse_response(raw)
    by_name = {e.name: e.entity_type for e in result.entities}
    assert by_name["Form I-693"] == EntityType.FORM
    assert by_name["USCIS"] == EntityType.AGENCY


def test_parse_response_alternate_field_keys(extractor):
    # Models sometimes use "value"/"category" instead of "name"/"entity_type".
    raw = json.dumps(
        {
            "entities": [{"value": "filing fee", "category": "fee"}],
            "relations": [],
        }
    )
    result = extractor._parse_response(raw)
    assert len(result.entities) == 1
    assert result.entities[0].name == "filing fee"
    assert result.entities[0].entity_type == EntityType.FEE


# ─── _parse_response: relations ─────────────────────────────────────────────


def test_parse_response_valid_relation(extractor):
    raw = json.dumps(
        {
            "entities": [
                {"name": "Form I-485", "entity_type": "form"},
                {"name": "Form I-693", "entity_type": "form"},
            ],
            "relations": [
                {
                    "subject": "Form I-485",
                    "predicate": "requires",
                    "object": "Form I-693",
                    "evidence": "Submit Form I-693 with your Form I-485.",
                }
            ],
        }
    )
    result = extractor._parse_response(raw)
    assert len(result.relations) == 1
    rel = result.relations[0]
    assert rel.subject == "Form I-485"
    assert rel.predicate == RelationType.REQUIRES
    assert rel.object == "Form I-693"
    assert "Form I-693" in rel.evidence


def test_parse_response_alternate_relation_keys(extractor):
    # source/target/relation instead of subject/object/predicate.
    raw = json.dumps(
        {
            "entities": [
                {"name": "Form I-765", "entity_type": "form"},
                {"name": "Form I-485", "entity_type": "form"},
            ],
            "relations": [
                {
                    "source": "Form I-765",
                    "relation": "filed_with",
                    "target": "Form I-485",
                }
            ],
        }
    )
    result = extractor._parse_response(raw)
    assert len(result.relations) == 1
    assert result.relations[0].predicate == RelationType.FILED_WITH


def test_parse_response_drops_invalid_predicate(extractor):
    raw = json.dumps(
        {
            "entities": [
                {"name": "Form I-485", "entity_type": "form"},
                {"name": "Form I-693", "entity_type": "form"},
            ],
            "relations": [
                {"subject": "Form I-485", "predicate": "frobnicates", "object": "Form I-693"}
            ],
        }
    )
    result = extractor._parse_response(raw)
    assert result.relations == []


def test_parse_response_drops_self_loop_relation(extractor):
    raw = json.dumps(
        {
            "entities": [{"name": "Form I-485", "entity_type": "form"}],
            "relations": [
                {"subject": "Form I-485", "predicate": "requires", "object": "Form I-485"}
            ],
        }
    )
    result = extractor._parse_response(raw)
    assert result.relations == []


def test_parse_response_strips_code_fence(extractor):
    raw = (
        "```json\n"
        + json.dumps({"entities": ["USCIS"], "relations": []})
        + "\n```"
    )
    result = extractor._parse_response(raw)
    assert [e.name for e in result.entities] == ["USCIS"]


def test_parse_response_extracts_embedded_json(extractor):
    # Leading prose before the JSON object — parser should find the braces.
    raw = "Here is the result you asked for:\n" + json.dumps(
        {"entities": ["Form I-130"], "relations": []}
    )
    result = extractor._parse_response(raw)
    assert [e.name for e in result.entities] == ["Form I-130"]


def test_parse_response_garbage_returns_empty(extractor):
    result = extractor._parse_response("not json at all, no braces here")
    assert result.entities == []
    assert result.relations == []
