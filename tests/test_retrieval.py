import pytest

from rag.embeddings import LocalEmbeddingService
from rag.retrieval import retrieve
from rag.vector_store import build_indexes


@pytest.fixture(scope="module")
def indexed_store(tmp_path_factory):
    path = tmp_path_factory.mktemp("chroma-retrieval")
    service = LocalEmbeddingService()
    build_indexes(persist_directory=path, embedding_service=service)
    return path, service


def test_retrieval_is_limited_to_one_strategy(indexed_store):
    path, service = indexed_store
    query = "What is the response SLA for a P1 critical ticket?"

    for strategy in ("fixed", "sentence"):
        results = retrieve(
            query,
            strategy=strategy,
            top_k=3,
            persist_directory=path,
            embedding_service=service,
        )
        assert 1 <= len(results) <= 3
        assert [result["rank"] for result in results] == list(
            range(1, len(results) + 1)
        )
        for result in results:
            assert result["chunk_id"]
            assert result["document_id"]
            assert result["text"]
            assert result["metadata"]["chunk_strategy"] == strategy
            assert result["distance"] >= 0
            assert result["similarity"] == pytest.approx(1 - result["distance"])
        assert "KB-002" in {result["document_id"] for result in results}


def test_repeat_complaint_smoke_query_returns_relevant_policy(indexed_store):
    path, service = indexed_store
    query = "When should a repeated unresolved complaint be escalated?"

    for strategy in ("fixed", "sentence"):
        results = retrieve(
            query,
            strategy=strategy,
            top_k=3,
            persist_directory=path,
            embedding_service=service,
        )
        assert results
        assert {result["document_id"] for result in results} & {"KB-003", "KB-007"}


def test_retrieval_rejects_invalid_strategy_and_top_k(indexed_store):
    path, service = indexed_store
    with pytest.raises(ValueError, match="strategy"):
        retrieve(
            "refund policy",
            strategy="both",
            persist_directory=path,
            embedding_service=service,
        )
    with pytest.raises(ValueError, match="top_k"):
        retrieve(
            "refund policy",
            top_k=0,
            persist_directory=path,
            embedding_service=service,
        )
