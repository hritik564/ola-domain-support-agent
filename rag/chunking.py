"""Deterministic chunking strategies for policy documents."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

_REQUIRED_DOCUMENT_FIELDS = {"document_id", "title", "topic", "filename", "text"}
_SENTENCE_SEPARATOR = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE = re.compile(r"\s+")


def _validate_documents(
    documents: Iterable[Mapping[str, str]],
) -> list[Mapping[str, str]]:
    validated = list(documents)
    for document in validated:
        missing = _REQUIRED_DOCUMENT_FIELDS - set(document)
        if missing:
            raise ValueError(f"Document is missing required fields: {sorted(missing)}")
        if not document["document_id"] or not document["text"].strip():
            raise ValueError("Documents require a non-empty ID and policy text.")
    return validated


def _make_chunk(
    document: Mapping[str, str],
    text: str,
    index: int,
    strategy: str,
) -> dict[str, str | int]:
    return {
        "chunk_id": f"{document['document_id']}-{strategy}-{index:03d}",
        "document_id": document["document_id"],
        "parent_title": document["title"],
        "topic": document["topic"],
        "filename": document["filename"],
        "chunk_index": index,
        "chunk_strategy": strategy,
        "text": text.strip(),
    }


def _fixed_ranges(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split by character count, preferring nearby whitespace boundaries."""
    normalized = _WHITESPACE.sub(" ", text).strip()
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        tentative_end = min(start + chunk_size, len(normalized))
        end = tentative_end
        if tentative_end < len(normalized):
            lower_bound = start + max(1, chunk_size // 2)
            boundary = normalized.rfind(" ", lower_bound, tentative_end)
            if boundary > start:
                end = boundary

        chunk_text = normalized[start:end].strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end >= len(normalized):
            break

        next_start = max(start + 1, end - overlap)
        if next_start < len(normalized) and normalized[next_start] != " ":
            boundary = normalized.rfind(" ", start, next_start)
            if boundary > start:
                next_start = boundary + 1
        start = next_start

    return chunks


def chunk_fixed_size(
    documents: Iterable[Mapping[str, str]],
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[dict[str, str | int]]:
    """Create stable character-based chunks with a 500/100 default."""
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if not isinstance(overlap, int) or isinstance(overlap, bool) or overlap < 0:
        raise ValueError("overlap must be a non-negative integer.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    chunks: list[dict[str, str | int]] = []
    for document in _validate_documents(documents):
        for index, text in enumerate(
            _fixed_ranges(document["text"], chunk_size, overlap)
        ):
            chunks.append(_make_chunk(document, text, index, "fixed"))
    return chunks


def split_policy_sentences(text: str) -> list[str]:
    """Split policy text at sentence-ending punctuation and preserve each sentence."""
    return [
        sentence.strip()
        for sentence in _SENTENCE_SEPARATOR.split(text.strip())
        if sentence.strip()
    ]


def chunk_by_sentence(
    documents: Iterable[Mapping[str, str]],
    max_sentences_per_chunk: int = 2,
) -> list[dict[str, str | int]]:
    """Group policy sentences without crossing sentence boundaries."""
    if (
        not isinstance(max_sentences_per_chunk, int)
        or isinstance(max_sentences_per_chunk, bool)
        or max_sentences_per_chunk <= 0
    ):
        raise ValueError("max_sentences_per_chunk must be a positive integer.")

    chunks: list[dict[str, str | int]] = []
    for document in _validate_documents(documents):
        sentences = split_policy_sentences(document["text"])
        for index, start in enumerate(range(0, len(sentences), max_sentences_per_chunk)):
            text = " ".join(sentences[start : start + max_sentences_per_chunk])
            chunks.append(_make_chunk(document, text, index, "sentence"))
    return chunks
