from types import SimpleNamespace

import pytest

import rag.grounded_generation as grounded_generation
import rag.retrieval as retrieval
from evaluation.retrieval_eval import evaluate_query
from knowledge_base.loader import load_knowledge_base
from rag.chunking import chunk_by_sentence
from rag.config import PRODUCTION_THRESHOLD, PRODUCTION_TOP_K, PRODUCTION_STRATEGY
from rag.grounded_generation import FALLBACK_ANSWER, generate_grounded_answer
from rag.retrieval import expand_parent_context
from rag.vector_store import SENTENCE_COLLECTION

P1_ANSWER = (
    "For P1, the initial response is due within 15 minutes and the target "
    "resolution time is within 4 hours."
)


class FakeCollection:
    metadata = {"hnsw:space": "cosine"}

    def __init__(self, records):
        self.records = records
        self.get_calls = []

    def get(self, *, where, include):
        self.get_calls.append({"where": where, "include": include})
        parent_ids = set(where["document_id"]["$in"])
        records = [
            record
            for record in self.records
            if record["metadata"]["document_id"] in parent_ids
        ]
        return {
            "ids": [record["chunk_id"] for record in records],
            "documents": [record["text"] for record in records],
            "metadatas": [record["metadata"] for record in records],
        }


class FakeClient:
    def __init__(self, collection):
        self.collection = collection

    def list_collections(self):
        return [SimpleNamespace(name=SENTENCE_COLLECTION)]

    def get_collection(self, name):
        assert name == SENTENCE_COLLECTION
        return self.collection


def _install_fake_collection(monkeypatch, records):
    collection = FakeCollection(records)
    client = FakeClient(collection)
    monkeypatch.setattr(retrieval, "_persistent_client", lambda _path: client)
    return collection


def _records_from_policy_chunks(chunks):
    return [
        {
            "chunk_id": str(chunk["chunk_id"]),
            "text": str(chunk["text"]),
            "metadata": {
                key: chunk[key]
                for key in (
                    "document_id",
                    "parent_title",
                    "topic",
                    "filename",
                    "chunk_index",
                    "chunk_strategy",
                )
            },
        }
        for chunk in chunks
    ]


def test_expansion_fetches_only_retrieved_parents_and_marks_siblings(monkeypatch):
    records = [
        {
            "chunk_id": f"KB-{document:03d}-sentence-{index:03d}",
            "text": f"Policy sentence {document}-{index}.",
            "metadata": {
                "document_id": f"KB-{document:03d}",
                "chunk_index": index,
                "chunk_strategy": "sentence",
            },
        }
        for document, indexes in ((2, range(3)), (3, range(2)), (999, range(1)))
        for index in indexes
    ]
    collection = _install_fake_collection(monkeypatch, records)
    raw_hits = [
        {
            "rank": 1,
            "chunk_id": "KB-002-sentence-002",
            "document_id": "KB-002",
            "text": "Policy sentence 2-2.",
            "metadata": {"chunk_index": 2, "chunk_strategy": "sentence"},
            "distance": 0.4,
            "similarity": 0.6,
        },
        {
            "rank": 2,
            "chunk_id": "KB-003-sentence-000",
            "document_id": "KB-003",
            "text": "Policy sentence 3-0.",
            "metadata": {"chunk_index": 0, "chunk_strategy": "sentence"},
            "distance": 0.5,
            "similarity": 0.5,
        },
    ]
    original_raw_hits = [dict(hit) for hit in raw_hits]

    expanded = expand_parent_context(raw_hits, strategy="sentence")

    assert collection.get_calls == [
        {
            "where": {"document_id": {"$in": ["KB-002", "KB-003"]}},
            "include": ["documents", "metadatas"],
        }
    ]
    assert [result["chunk_id"] for result in expanded] == [
        "KB-002-sentence-002",
        "KB-003-sentence-000",
        "KB-002-sentence-000",
        "KB-002-sentence-001",
        "KB-003-sentence-001",
    ]
    assert [result["retrieval_origin"] for result in expanded[:2]] == [
        "vector_search",
        "vector_search",
    ]
    siblings = expanded[2:]
    assert all(item["retrieval_origin"] == "parent_expansion" for item in siblings)
    assert all(item["document_id"] in {"KB-002", "KB-003"} for item in siblings)
    assert all("similarity" not in item and "distance" not in item for item in siblings)
    assert all("rank" not in item for item in siblings)
    assert raw_hits == original_raw_hits
    assert all("KB-999" not in item["document_id"] for item in expanded)


def test_real_chroma_top_five_case_uses_parent_expansion_for_exact_p1_answer(
    monkeypatch,
):
    chunks = chunk_by_sentence(load_knowledge_base())
    chunks_by_id = {str(chunk["chunk_id"]): chunk for chunk in chunks}
    records = _records_from_policy_chunks(chunks)
    collection = _install_fake_collection(monkeypatch, records)

    ranking = (
        ("KB-003-sentence-000", 0.576289),
        ("KB-003-sentence-001", 0.537036),
        ("KB-002-sentence-002", 0.534301),
        ("KB-001-sentence-000", 0.526174),
        ("KB-010-sentence-000", 0.486308),
    )
    raw_hits = []
    for rank, (chunk_id, similarity) in enumerate(ranking, start=1):
        chunk = chunks_by_id[chunk_id]
        raw_hits.append(
            {
                "rank": rank,
                "chunk_id": chunk_id,
                "document_id": str(chunk["document_id"]),
                "text": str(chunk["text"]),
                "metadata": {
                    "chunk_strategy": "sentence",
                    "chunk_index": chunk["chunk_index"],
                },
                "distance": 1.0 - similarity,
                "similarity": similarity,
            }
        )
    monkeypatch.setattr(
        grounded_generation,
        "retrieve",
        lambda query, strategy, top_k: raw_hits[:top_k],
    )
    query = "What is the response SLA for a P1 critical ticket?"

    result = generate_grounded_answer(
        query,
        strategy=PRODUCTION_STRATEGY,
        threshold=PRODUCTION_THRESHOLD,
        top_k=PRODUCTION_TOP_K,
        expand_parent_siblings=True,
    )

    assert collection.get_calls
    assert result["answer"] == P1_ANSWER
    assert result["grounded"] is True
    assert result["fallback_triggered"] is False
    assert result["top1_document_id"] == "KB-003"
    assert result["top1_chunk_id"] == "KB-003-sentence-000"
    assert result["top1_similarity"] == pytest.approx(0.576289)
    assert result["threshold"] == PRODUCTION_THRESHOLD
    assert result["sources"] == [
        {
            "document_id": "KB-002",
            "chunk_id": "KB-002-sentence-000",
            "retrieval_origin": "parent_expansion",
        }
    ]
    assert "similarity" not in result["sources"][0]
    assert "1 hour" not in result["answer"]
    assert "12 hours" not in result["answer"]


def test_oos_threshold_returns_fallback_before_parent_expansion(monkeypatch):
    monkeypatch.setattr(
        grounded_generation,
        "retrieve",
        lambda query, strategy, top_k: [
            {
                "rank": 1,
                "chunk_id": "KB-008-sentence-000",
                "document_id": "KB-008",
                "text": "Service credit is considered after a prolonged outage.",
                "metadata": {"chunk_strategy": "sentence", "chunk_index": 0},
                "distance": 1.0 - 0.028549,
            }
        ],
    )

    def expansion_must_not_run(*args, **kwargs):
        pytest.fail("out-of-scope evidence must not trigger parent expansion")

    monkeypatch.setattr(
        grounded_generation, "expand_parent_context", expansion_must_not_run
    )
    result = generate_grounded_answer(
        "How do I bake a chocolate cake?",
        strategy=PRODUCTION_STRATEGY,
        threshold=PRODUCTION_THRESHOLD,
        top_k=PRODUCTION_TOP_K,
        expand_parent_siblings=True,
    )

    assert result["answer"] == FALLBACK_ANSWER
    assert result["fallback_triggered"] is True
    assert result["top1_similarity"] == pytest.approx(0.028549)
    assert result["sources"] == []


def test_task5_evaluation_still_scores_only_raw_top_three():
    query = {
        "query_id": "Q1",
        "query": "What is the response SLA for a P1 critical ticket?",
        "relevant_document_ids": ["KB-002"],
    }
    calls = []
    raw_hits = [
        {"chunk_id": "KB-002-sentence-002", "document_id": "KB-002"},
        {"chunk_id": "KB-003-sentence-000", "document_id": "KB-003"},
        {"chunk_id": "KB-001-sentence-000", "document_id": "KB-001"},
    ]

    def fake_retrieve(query_text, strategy, top_k):
        calls.append((query_text, strategy, top_k))
        return raw_hits[:top_k]

    result = evaluate_query(query, "sentence", fake_retrieve)

    assert calls == [(query["query"], "sentence", 3)]
    assert result["raw_retrieved_chunk_ids"] == [
        "KB-002-sentence-002",
        "KB-003-sentence-000",
        "KB-001-sentence-000",
    ]
    assert result["unique_retrieved_parent_document_ids"] == [
        "KB-002",
        "KB-003",
        "KB-001",
    ]
    assert (result["precision_numerator"], result["precision_denominator"]) == (1, 3)
    assert (result["recall_numerator"], result["recall_denominator"]) == (1, 1)
    assert result["precision"] == pytest.approx(1 / 3)
    assert result["recall"] == 1.0
