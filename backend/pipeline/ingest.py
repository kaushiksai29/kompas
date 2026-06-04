"""Ingestion and chunking pipeline for GraphRAG.

Parses PDF and HTML documents into semantically chunked text segments,
preserving heading hierarchy and document structure. Designed for
government/legal documents that may have non-standard encodings.

Usage::

    from backend.pipeline.ingest import ingest_file, ingest_directory

    chunks = ingest_file("path/to/document.pdf")
    all_chunks = ingest_directory("path/to/docs/")
"""

from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

MIN_CHUNK_CHARS: int = 500
"""Minimum number of characters per chunk (soft target)."""

MAX_CHUNK_CHARS: int = 1500
"""Maximum number of characters per chunk (hard ceiling)."""

OVERLAP_CHARS: int = 100
"""Approximate character overlap between consecutive chunks for context."""

_HEADING_SEPARATOR: str = " > "
"""Separator used when joining heading hierarchy into a path string."""

# Font-size thresholds used to detect PDF headings.  Spans whose font size
# exceeds the *body median* by these ratios are treated as headings.
_PDF_HEADING_RATIO_H1: float = 1.60
_PDF_HEADING_RATIO_H2: float = 1.30
_PDF_HEADING_RATIO_H3: float = 1.10

# Fallback encodings tried when reading HTML files.
_HTML_ENCODINGS: list[str] = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]

# HTML tags that denote heading boundaries.
_HTML_HEADING_TAGS: set[str] = {"h1", "h2", "h3", "h4", "h5", "h6"}

# Supported file extensions (lowercase, with leading dot).
_SUPPORTED_PDF_EXTS: set[str] = {".pdf"}
_SUPPORTED_HTML_EXTS: set[str] = {".html", ".htm", ".xhtml"}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class DocumentChunk(BaseModel):
    """A single semantically-chunked text segment from a source document.

    Attributes:
        chunk_id:     Unique identifier (UUID4) for this chunk.
        filepath:     Absolute path (or URL) of the source document.
        heading_path: Human-readable heading hierarchy,
                      e.g. ``'Chapter 3 > Section 1941.1'``.
        content:      The plain-text content of the chunk.
        char_start:   Character offset where this chunk begins in the
                      *full extracted text* of the document.
        char_end:     Character offset where this chunk ends (exclusive).
        metadata:     Arbitrary metadata dict (page number, source URL, …).
    """

    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filepath: str
    heading_path: str = ""
    content: str
    char_start: int
    char_end: int
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal: structured section representation
# ---------------------------------------------------------------------------


class _Section:
    """A contiguous block of text under a single heading hierarchy."""

    __slots__ = ("heading_path", "text", "level", "metadata")

    def __init__(
        self,
        heading_path: str,
        text: str,
        level: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.heading_path = heading_path
        self.text = text
        self.level = level
        self.metadata: dict[str, Any] = metadata or {}


# =========================================================================
# PDF parsing
# =========================================================================


def _median(values: list[float]) -> float:
    """Return the median of a non-empty list of floats."""
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2.0
    return s[mid]


def _classify_pdf_heading_level(
    font_size: float,
    is_bold: bool,
    median_size: float,
) -> int | None:
    """Return a heading level (1-3) or ``None`` for body text.

    Classification is based on font size relative to the document's median
    body font size, with boldness as an additional signal.
    """
    if median_size <= 0:
        return None

    ratio = font_size / median_size

    if ratio >= _PDF_HEADING_RATIO_H1:
        return 1
    if ratio >= _PDF_HEADING_RATIO_H2:
        return 2
    if ratio >= _PDF_HEADING_RATIO_H3 and is_bold:
        return 3
    if is_bold and ratio >= 1.0:
        # Bold text at or above median size → treat as H3
        return 3
    return None


def _extract_pdf_sections(filepath: str) -> list[_Section]:
    """Extract heading-delimited sections from a PDF file.

    Uses PyMuPDF (``fitz``) to iterate over text spans, classify them by
    font size / boldness, and split the document at heading boundaries.

    Returns a list of ``_Section`` objects, each carrying its heading
    hierarchy path and body text.
    """
    doc = fitz.open(filepath)
    try:
        # ------------------------------------------------------------------
        # First pass: collect all span font sizes to compute the median body
        # size (needed for heading classification).
        # ------------------------------------------------------------------
        all_sizes: list[float] = []
        for page in doc:
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
            for block in blocks:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text:
                            all_sizes.append(span["size"])

        if not all_sizes:
            logger.warning("PDF has no extractable text: %s", filepath)
            return []

        median_size = _median(all_sizes)
        logger.debug(
            "PDF median font size: %.2f (%d spans) – %s",
            median_size,
            len(all_sizes),
            filepath,
        )

        # ------------------------------------------------------------------
        # Second pass: build sections.
        # ------------------------------------------------------------------
        heading_stack: list[str] = []  # current heading hierarchy
        level_stack: list[int] = []    # corresponding heading levels

        sections: list[_Section] = []
        current_text_parts: list[str] = []
        current_page: int | None = None

        def _flush_section() -> None:
            """Persist the accumulated text as a new section."""
            body = "\n".join(current_text_parts).strip()
            if not body:
                return
            heading_path = _HEADING_SEPARATOR.join(heading_stack) if heading_stack else ""
            meta: dict[str, Any] = {}
            if current_page is not None:
                meta["page_number"] = current_page
            sections.append(
                _Section(
                    heading_path=heading_path,
                    text=body,
                    level=level_stack[-1] if level_stack else 0,
                    metadata=meta,
                )
            )

        for page_idx, page in enumerate(doc):
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
            for block in blocks:
                for line in block.get("lines", []):
                    line_text_parts: list[str] = []
                    line_heading_level: int | None = None

                    for span in line.get("spans", []):
                        text = span.get("text", "")
                        if not text.strip():
                            continue
                        size: float = span["size"]
                        flags: int = span.get("flags", 0)
                        is_bold = bool(flags & (1 << 4))  # bit 4 = bold

                        level = _classify_pdf_heading_level(size, is_bold, median_size)
                        if level is not None:
                            line_heading_level = min(level, line_heading_level or 99)
                        line_text_parts.append(text)

                    line_text = " ".join(line_text_parts).strip()
                    if not line_text:
                        continue

                    if line_heading_level is not None:
                        # Flush any accumulated body text before the heading.
                        _flush_section()
                        current_text_parts = []
                        current_page = page_idx + 1

                        # Pop headings at or below the current level.
                        while level_stack and level_stack[-1] >= line_heading_level:
                            level_stack.pop()
                            heading_stack.pop()

                        heading_stack.append(line_text)
                        level_stack.append(line_heading_level)
                    else:
                        if current_page is None:
                            current_page = page_idx + 1
                        current_text_parts.append(line_text)

            # Add a blank line between pages to preserve visual separation.
            if current_text_parts:
                current_text_parts.append("")

        _flush_section()
        return sections

    finally:
        doc.close()


# =========================================================================
# HTML parsing
# =========================================================================


def _read_html_file(filepath: str) -> str:
    """Read an HTML file, trying multiple encodings.

    Government documents frequently use ``cp1252`` or ``latin-1`` even when
    they claim UTF-8 in their ``<meta>`` tag.  We try several common
    encodings in order and return the first one that succeeds.
    """
    raw_bytes = Path(filepath).read_bytes()

    for encoding in _HTML_ENCODINGS:
        try:
            return raw_bytes.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue

    # Last resort: decode with replacement characters.
    logger.warning(
        "All encoding attempts failed for %s – falling back to utf-8 "
        "with replacement characters.",
        filepath,
    )
    return raw_bytes.decode("utf-8", errors="replace")


def _heading_level(tag_name: str) -> int:
    """Return the numeric level for an HTML heading tag (e.g. ``h2`` → 2)."""
    match = re.match(r"h(\d)", tag_name, re.IGNORECASE)
    return int(match.group(1)) if match else 0


def _extract_html_sections(filepath: str) -> list[_Section]:
    """Extract heading-delimited sections from an HTML file.

    Uses BeautifulSoup to walk the DOM tree.  Text between heading tags
    is collected into ``_Section`` objects with their heading hierarchy.
    """
    html_text = _read_html_file(filepath)
    soup = BeautifulSoup(html_text, "lxml")

    # Remove <script>, <style>, and <nav> elements.
    for tag in soup.find_all(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    body = soup.body or soup

    heading_stack: list[str] = []
    level_stack: list[int] = []
    sections: list[_Section] = []
    current_text_parts: list[str] = []

    def _flush_section() -> None:
        body_text = "\n".join(current_text_parts).strip()
        if not body_text:
            return
        heading_path = _HEADING_SEPARATOR.join(heading_stack) if heading_stack else ""
        sections.append(
            _Section(
                heading_path=heading_path,
                text=body_text,
                level=level_stack[-1] if level_stack else 0,
            )
        )

    for element in body.descendants:
        if isinstance(element, Tag):
            tag_name = element.name.lower()
            if tag_name in _HTML_HEADING_TAGS:
                _flush_section()
                current_text_parts = []

                level = _heading_level(tag_name)
                heading_text = element.get_text(separator=" ", strip=True)
                if not heading_text:
                    continue

                while level_stack and level_stack[-1] >= level:
                    level_stack.pop()
                    heading_stack.pop()

                heading_stack.append(heading_text)
                level_stack.append(level)

        elif isinstance(element, str):
            # NavigableString — only collect if parent is a visible tag.
            parent = element.parent
            if parent and isinstance(parent, Tag):
                parent_name = parent.name.lower()
                if parent_name in _HTML_HEADING_TAGS:
                    continue  # heading text is handled above
                text = element.strip()
                if text:
                    current_text_parts.append(text)

    _flush_section()
    return sections


# =========================================================================
# Chunking logic
# =========================================================================


def _split_text_with_overlap(
    text: str,
    max_chars: int = MAX_CHUNK_CHARS,
    min_chars: int = MIN_CHUNK_CHARS,
    overlap: int = OVERLAP_CHARS,
) -> list[tuple[str, int, int]]:
    """Split *text* into chunks respecting paragraph/sentence boundaries.

    Returns a list of ``(chunk_text, start_offset, end_offset)`` tuples
    where offsets are relative to *text*.

    The algorithm:
    1. Split the text into paragraphs (double-newline delimited).
    2. Accumulate paragraphs until ``max_chars`` would be exceeded.
    3. When a chunk is flushed, include ~``overlap`` characters from the
       end of the previous chunk at the start of the next one.
    4. If a single paragraph exceeds ``max_chars``, split it at sentence
       boundaries; if a single sentence is still too long, hard-split it.
    """
    if len(text) <= max_chars:
        return [(text, 0, len(text))]

    # Split into paragraphs (preserving their relative offsets).
    paragraphs: list[tuple[str, int]] = []
    for match in re.finditer(r"(?s).+?(?:\n{2,}|$)", text):
        para = match.group().strip()
        if para:
            paragraphs.append((para, match.start()))

    if not paragraphs:
        return [(text, 0, len(text))]

    chunks: list[tuple[str, int, int]] = []
    buffer: list[str] = []
    buffer_char_count: int = 0
    chunk_start_offset: int = paragraphs[0][1]

    def _flush_buffer() -> None:
        nonlocal buffer, buffer_char_count, chunk_start_offset
        if not buffer:
            return
        chunk_text = "\n\n".join(buffer)
        chunk_end_offset = chunk_start_offset + len(chunk_text)
        chunks.append((chunk_text, chunk_start_offset, chunk_end_offset))

        # Compute the overlap prefix for the next chunk.
        overlap_text = chunk_text[-overlap:] if len(chunk_text) > overlap else chunk_text
        buffer = [overlap_text]
        buffer_char_count = len(overlap_text)
        chunk_start_offset = max(0, chunk_end_offset - len(overlap_text))

    for para_text, para_offset in paragraphs:
        para_len = len(para_text)

        # If adding this paragraph would exceed max, flush first.
        if buffer_char_count + para_len > max_chars and buffer_char_count >= min_chars:
            _flush_buffer()

        # If a single paragraph is itself larger than max_chars, split it
        # by sentences.
        if para_len > max_chars:
            sentences = re.split(r"(?<=[.!?])\s+", para_text)
            for sentence in sentences:
                if buffer_char_count + len(sentence) > max_chars and buffer:
                    _flush_buffer()
                if len(sentence) > max_chars:
                    # Hard split for extremely long sentences.
                    for i in range(0, len(sentence), max_chars - overlap):
                        fragment = sentence[i : i + max_chars]
                        if buffer_char_count + len(fragment) > max_chars and buffer:
                            _flush_buffer()
                        buffer.append(fragment)
                        buffer_char_count += len(fragment)
                else:
                    buffer.append(sentence)
                    buffer_char_count += len(sentence)
        else:
            buffer.append(para_text)
            buffer_char_count += para_len

    # Flush remaining.
    if buffer:
        chunk_text = "\n\n".join(buffer)
        chunk_end_offset = chunk_start_offset + len(chunk_text)
        chunks.append((chunk_text, chunk_start_offset, chunk_end_offset))

    return chunks


def _sections_to_chunks(
    sections: list[_Section],
    filepath: str,
    extra_metadata: dict[str, Any] | None = None,
) -> list[DocumentChunk]:
    """Convert a list of ``_Section`` objects into ``DocumentChunk`` models.

    Long sections are split using ``_split_text_with_overlap``; short
    consecutive sections under the same heading path may be merged.
    """
    base_meta: dict[str, Any] = extra_metadata.copy() if extra_metadata else {}
    chunks: list[DocumentChunk] = []
    global_offset: int = 0  # running character offset in the full document

    for section in sections:
        text = section.text
        sec_meta = {**base_meta, **section.metadata}

        sub_chunks = _split_text_with_overlap(text)

        for chunk_text, local_start, local_end in sub_chunks:
            char_start = global_offset + local_start
            char_end = global_offset + local_end
            chunks.append(
                DocumentChunk(
                    filepath=filepath,
                    heading_path=section.heading_path,
                    content=chunk_text,
                    char_start=char_start,
                    char_end=char_end,
                    metadata=sec_meta,
                )
            )

        global_offset += len(text) + 1  # +1 for implicit separator

    return chunks


# =========================================================================
# Public API
# =========================================================================


def ingest_file(filepath: str) -> list[DocumentChunk]:
    """Parse a single PDF or HTML file and return semantically chunked text.

    Args:
        filepath: Absolute or relative path to the source document.

    Returns:
        A list of :class:`DocumentChunk` objects.

    Raises:
        FileNotFoundError: If *filepath* does not exist.
        ValueError: If the file extension is not supported.
    """
    path = Path(filepath).resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    abs_path = str(path)

    logger.info("Ingesting file: %s", abs_path)

    if ext in _SUPPORTED_PDF_EXTS:
        try:
            sections = _extract_pdf_sections(abs_path)
        except Exception:
            logger.exception("Failed to parse PDF: %s", abs_path)
            raise
    elif ext in _SUPPORTED_HTML_EXTS:
        try:
            sections = _extract_html_sections(abs_path)
        except Exception:
            logger.exception("Failed to parse HTML: %s", abs_path)
            raise
    else:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            f"Supported: {_SUPPORTED_PDF_EXTS | _SUPPORTED_HTML_EXTS}"
        )

    if not sections:
        logger.warning("No sections extracted from: %s", abs_path)
        return []

    chunks = _sections_to_chunks(sections, filepath=abs_path)
    logger.info(
        "Extracted %d chunk(s) from %s (total chars: %d)",
        len(chunks),
        abs_path,
        sum(len(c.content) for c in chunks),
    )
    return chunks


_BOILERPLATE_RE = re.compile(
    r"(?i)^\s*form\s+[a-z]+-\d+[a-z]?\s+.{0,40}?page\s+\d+\s+of\s+\d+"
)


def is_substantial(chunk: DocumentChunk, min_chars: int = 200) -> bool:
    """Heuristic: keep prose chunks, drop footers / fragmentary form fields.

    Government instruction booklets carry the procedural prose we want; the
    fillable form PDFs and repeated page footers ("Form I-485 Instructions
    04/01/24 Page 3 of 42") add noise and waste extraction calls.
    """
    text = chunk.content.strip()
    if len(text) < min_chars:
        return False
    if _BOILERPLATE_RE.match(text):
        return False
    alpha = sum(ch.isalpha() for ch in text)
    return alpha / max(len(text), 1) > 0.5


def clean_chunks(chunks: list[DocumentChunk], min_chars: int = 200) -> list[DocumentChunk]:
    """Filter a chunk list down to substantial, non-boilerplate chunks."""
    return [c for c in chunks if is_substantial(c, min_chars)]


def ingest_directory(path: str) -> list[DocumentChunk]:
    """Recursively process all PDF and HTML files in *path*.

    Args:
        path: Absolute or relative path to the directory to scan.

    Returns:
        A flat list of :class:`DocumentChunk` objects from all files.

    Raises:
        FileNotFoundError: If *path* does not exist.
        NotADirectoryError: If *path* is not a directory.
    """
    dir_path = Path(path).resolve()
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    supported_exts = _SUPPORTED_PDF_EXTS | _SUPPORTED_HTML_EXTS
    files = sorted(
        p
        for p in dir_path.rglob("*")
        if p.is_file() and p.suffix.lower() in supported_exts
    )

    if not files:
        logger.warning("No supported files found in: %s", dir_path)
        return []

    logger.info("Found %d file(s) to ingest in %s", len(files), dir_path)

    all_chunks: list[DocumentChunk] = []
    for file_path in files:
        try:
            chunks = ingest_file(str(file_path))
            all_chunks.extend(chunks)
        except Exception:
            logger.exception(
                "Skipping file due to error: %s",
                file_path,
            )

    logger.info(
        "Directory ingestion complete: %d chunk(s) from %d file(s) in %s",
        len(all_chunks),
        len(files),
        dir_path,
    )
    return all_chunks
