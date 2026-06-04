"""Unit tests for backend.pipeline.ingest chunk-cleaning heuristics.

Covers ``is_substantial`` and ``clean_chunks``: dropping page-footer
boilerplate, short chunks, and low-alpha (form-field / numeric) chunks
while keeping genuine prose. No PDF/HTML parsing or network here.
"""

from __future__ import annotations

from backend.pipeline.ingest import (
    DocumentChunk,
    clean_chunks,
    is_substantial,
)


def _chunk(content: str) -> DocumentChunk:
    """Build a minimal DocumentChunk with the given content."""
    return DocumentChunk(
        filepath="/tmp/i-485instr.pdf",
        heading_path="Part 1 > Eligibility",
        content=content,
        char_start=0,
        char_end=len(content),
    )


# A realistic block of instruction-booklet prose (well over 200 chars,
# mostly alphabetic, not a page footer).
_PROSE = (
    "You may apply to adjust your status to that of a lawful permanent "
    "resident if an immigrant visa is immediately available to you at the "
    "time you file your application. To be eligible, you must generally be "
    "physically present in the United States, have been inspected and "
    "admitted or paroled, and remain admissible to the United States. "
    "Read the instructions carefully before completing this form."
)


def test_prose_is_substantial():
    assert is_substantial(_chunk(_PROSE)) is True


def test_clean_chunks_keeps_prose():
    chunks = [_chunk(_PROSE)]
    kept = clean_chunks(chunks)
    assert kept == chunks


def test_page_footer_boilerplate_dropped():
    # Exact shape called out in the spec: "Form I-485 Instructions 04/01/24
    # Page 3 of 42". Padded with filler so length alone wouldn't reject it,
    # proving the boilerplate regex (not the length check) does the work.
    footer = (
        "Form I-485 Instructions 04/01/24 Page 3 of 42 "
        + "continued from the previous page and onto the next page. " * 4
    )
    assert len(footer.strip()) >= 200
    assert is_substantial(_chunk(footer)) is False


def test_boilerplate_other_form_number_dropped():
    footer = (
        "Form N-400 Instructions 09/01/22 Page 12 of 18 "
        + "lorem ipsum dolor sit amet consectetur adipiscing elit. " * 4
    )
    assert is_substantial(_chunk(footer)) is False


def test_short_chunk_dropped():
    # Below the default 200-char minimum.
    short = "File Form I-485 with USCIS."
    assert len(short) < 200
    assert is_substantial(_chunk(short)) is False


def test_low_alpha_chunk_dropped():
    # A fillable-form field dump: long enough, but mostly digits/punctuation,
    # so the alpha ratio falls below 0.5.
    low_alpha = (
        "1. ____  2. ____  3. ____  4. ____  5. ____  "
        "A-12345678  04/01/2024  $1,225.00  (___) ___-____  "
        "00000 11111 22222 33333 44444 55555 66666 77777 88888 99999 "
        "[ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ]"
    )
    assert len(low_alpha) >= 200
    alpha = sum(c.isalpha() for c in low_alpha)
    assert alpha / len(low_alpha) <= 0.5
    assert is_substantial(_chunk(low_alpha)) is False


def test_clean_chunks_filters_mixed_list():
    prose = _chunk(_PROSE)
    footer = _chunk(
        "Form I-130 Instructions 01/20/25 Page 1 of 14 "
        + "see the table of contents on the following pages for details. " * 4
    )
    short = _chunk("Too short to keep.")
    kept = clean_chunks([prose, footer, short])
    assert kept == [prose]


def test_min_chars_override_respected():
    # A 120-char prose snippet: dropped at the default 200, kept at 100.
    text = (
        "You must submit your application together with the required "
        "supporting evidence and the correct filing fee today now."
    )
    assert 100 <= len(text) < 200
    c = _chunk(text)
    assert is_substantial(c, min_chars=200) is False
    assert is_substantial(c, min_chars=100) is True


def test_empty_chunk_dropped():
    assert is_substantial(_chunk("   ")) is False
