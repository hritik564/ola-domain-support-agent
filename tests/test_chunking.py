import re

import pytest

from knowledge_base.loader import load_knowledge_base
from rag.chunking import (
    chunk_by_sentence,
    chunk_fixed_size,
    split_policy_sentences,
)


def _assert_chunk_metadata(chunks, strategy):
    required = {
        "chunk_id",
        "document_id",
        "parent_title",
        "topic",
        "filename",
        "chunk_index",
        "chunk_strategy",
        "text",
    }
    assert all(required.issubset(chunk) for chunk in chunks)
    assert all(chunk["chunk_strategy"] == strategy for chunk in chunks)
    assert all(chunk["text"].strip() for chunk in chunks)
    ids = [chunk["chunk_id"] for chunk in chunks]
    assert len(ids) == len(set(ids))


def test_fixed_chunking_is_deterministic_and_preserves_documents():
    documents = load_knowledge_base()
    first = chunk_fixed_size(documents)
    second = chunk_fixed_size(documents)

    assert first == second
    assert first
    _assert_chunk_metadata(first, "fixed")
    assert {chunk["document_id"] for chunk in first} == {
        document["document_id"] for document in documents
    }
    assert all(len(chunk["text"]) <= 500 for chunk in first)
    assert all(chunk["chunk_id"].endswith("-fixed-000") for chunk in first if chunk["chunk_index"] == 0)


def test_fixed_chunking_rejects_invalid_configuration():
    documents = load_knowledge_base()
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_fixed_size(documents, chunk_size=0)
    with pytest.raises(ValueError, match="overlap"):
        chunk_fixed_size(documents, chunk_size=100, overlap=100)
    with pytest.raises(ValueError, match="overlap"):
        chunk_fixed_size(documents, chunk_size=100, overlap=-1)


def test_fixed_chunking_uses_stable_ids_for_long_text():
    document = {
        "document_id": "KB-900",
        "title": "Long Test",
        "topic": "test",
        "filename": "long.md",
        "text": " ".join(["policyword"] * 200),
    }
    first = chunk_fixed_size([document], chunk_size=80, overlap=20)
    assert first == chunk_fixed_size([document], chunk_size=80, overlap=20)
    assert len(first) > 1
    assert all(len(chunk["text"]) <= 80 for chunk in first)


def test_sentence_chunking_is_deterministic_and_limits_sentence_count():
    documents = load_knowledge_base()
    first = chunk_by_sentence(documents)
    second = chunk_by_sentence(documents)

    assert first == second
    assert first
    _assert_chunk_metadata(first, "sentence")
    assert {chunk["document_id"] for chunk in first} == {
        document["document_id"] for document in documents
    }
    assert all(
        len(split_policy_sentences(chunk["text"])) <= 2 for chunk in first
    )


def test_sentence_splitter_respects_punctuation_and_line_breaks():
    text = "First policy sentence.\nSecond policy sentence! Third policy sentence?"
    assert split_policy_sentences(text) == [
        "First policy sentence.",
        "Second policy sentence!",
        "Third policy sentence?",
    ]


def test_sentence_chunking_rejects_invalid_group_size():
    with pytest.raises(ValueError, match="positive integer"):
        chunk_by_sentence(load_knowledge_base(), max_sentences_per_chunk=0)
