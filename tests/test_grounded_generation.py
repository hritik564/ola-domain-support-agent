import pytest

import rag.grounded_generation as grounded_generation
from knowledge_base.loader import load_knowledge_base
from rag.grounded_generation import (
    DEMONSTRATION_QUERIES,
    FALLBACK_ANSWER,
    generate_grounded_answer,
)

SUPPORTED_RESULT = {
    "rank": 1,
    "chunk_id": "KB-002-sentence-000",
    "document_id": "KB-002",
    "text": (
        "For P1, the initial response is due within 15 minutes and the target "
        "resolution time is within 4 hours. "
        "For P2, the initial response is due within 1 hour and the target "
        "resolution time is within 12 hours."
    ),
    "metadata": {"chunk_strategy": "sentence"},
    "distance": 0.2,
    "similarity": 0.2,  # Deliberately wrong to verify distance conversion.
}


def test_required_grounded_demonstrations_are_defined():
    assert len(DEMONSTRATION_QUERIES) >= 6
    assert DEMONSTRATION_QUERIES[:5] == (
        "What is the response SLA for a P1 critical ticket?",
        "When should a repeated unresolved complaint be escalated?",
        "When can service credit be considered?",
        "What support is available outside standard business hours?",
        "How should customers be informed during a widespread outage?",
    )
    assert DEMONSTRATION_QUERIES[5] == "How do I bake a chocolate cake?"


def test_supported_answer_uses_only_relevant_retrieved_sentences(monkeypatch):
    monkeypatch.setattr(
        grounded_generation,
        "retrieve",
        lambda query, strategy, top_k: [SUPPORTED_RESULT],
    )
    result = generate_grounded_answer(
        "What is the response SLA for a P1 critical ticket?",
        strategy="sentence",
        threshold=0.75,
    )
    assert result["grounded"] is True
    assert result["fallback_triggered"] is False
    assert result["answer"] == (
        "For P1, the initial response is due within 15 minutes and the target "
        "resolution time is within 4 hours."
    )
    assert "12 hours" not in result["answer"]
    assert result["top1_similarity"] == pytest.approx(0.8)
    assert result["threshold"] == 0.75
    assert result["top1_document_id"] == "KB-002"
    assert result["top1_chunk_id"] == "KB-002-sentence-000"
    assert result["sources"] == [
        {
            "document_id": "KB-002",
            "chunk_id": "KB-002-sentence-000",
            "similarity": pytest.approx(0.8),
            "retrieval_origin": "vector_search",
        }
    ]


def test_five_required_in_scope_demonstrations_are_grounded_with_retrieved_policy(
    monkeypatch,
):
    documents = {
        document["document_id"]: document for document in load_knowledge_base()
    }
    expected_documents = {
        DEMONSTRATION_QUERIES[0]: ["KB-002"],
        DEMONSTRATION_QUERIES[1]: ["KB-003", "KB-007"],
        DEMONSTRATION_QUERIES[2]: ["KB-008"],
        DEMONSTRATION_QUERIES[3]: ["KB-006"],
        DEMONSTRATION_QUERIES[4]: ["KB-011"],
    }

    def fake_retrieve(query, strategy, top_k):
        assert top_k == 3
        return [
            {
                "rank": rank,
                "chunk_id": f"{document_id}-{strategy}-000",
                "document_id": document_id,
                "text": documents[document_id]["text"],
                "metadata": {"chunk_strategy": strategy},
                "distance": 0.2,
            }
            for rank, document_id in enumerate(expected_documents[query], start=1)
        ]

    monkeypatch.setattr(grounded_generation, "retrieve", fake_retrieve)
    for query in DEMONSTRATION_QUERIES[:5]:
        result = generate_grounded_answer(
            query, strategy="sentence", threshold=0.5, top_k=3
        )
        assert result["grounded"] is True, query
        assert result["fallback_triggered"] is False, query
        assert result["answer"]
        assert result["sources"]
        assert all(
            source["document_id"] in expected_documents[query]
            for source in result["sources"]
        )


def test_below_threshold_returns_exact_fallback(monkeypatch):
    monkeypatch.setattr(
        grounded_generation,
        "retrieve",
        lambda query, strategy, top_k: [{**SUPPORTED_RESULT, "distance": 0.4}],
    )
    result = generate_grounded_answer(
        "How do I bake a chocolate cake?",
        strategy="fixed",
        threshold=0.7,
    )
    assert result["answer"] == FALLBACK_ANSWER
    assert result["grounded"] is False
    assert result["fallback_triggered"] is True
    assert result["top1_similarity"] == pytest.approx(0.6)


def test_similarity_uses_one_minus_distance_not_raw_distance(monkeypatch):
    monkeypatch.setattr(
        grounded_generation,
        "retrieve",
        lambda query, strategy, top_k: [SUPPORTED_RESULT],
    )
    result = generate_grounded_answer(
        "What is the response SLA for a P1 critical ticket?",
        strategy="fixed",
        threshold=0.75,
    )
    assert result["top1_similarity"] > result["threshold"]
    assert result["grounded"] is True


def test_no_retrieval_results_returns_fallback(monkeypatch):
    monkeypatch.setattr(
        grounded_generation, "retrieve", lambda query, strategy, top_k: []
    )
    result = generate_grounded_answer(
        "unknown", strategy="sentence", threshold=0.5
    )
    assert result["answer"] == FALLBACK_ANSWER
    assert result["top1_similarity"] is None
    assert result["top1_document_id"] is None
    assert result["top1_chunk_id"] is None
    assert result["sources"] == []


def test_high_similarity_without_matching_policy_sentence_still_falls_back(
    monkeypatch,
):
    monkeypatch.setattr(
        grounded_generation,
        "retrieve",
        lambda query, strategy, top_k: [
            {
                **SUPPORTED_RESULT,
                "text": "P1 tickets receive an initial response within 15 minutes.",
                "distance": 0.1,
            }
        ],
    )
    result = generate_grounded_answer(
        "How do I bake a chocolate cake?",
        strategy="sentence",
        threshold=0.5,
    )
    assert result["top1_similarity"] == pytest.approx(0.9)
    assert result["answer"] == FALLBACK_ANSWER
    assert result["fallback_triggered"] is True


def test_invalid_strategy_and_threshold_raise_clear_errors():
    with pytest.raises(ValueError, match="strategy"):
        generate_grounded_answer("query", strategy="all", threshold=0.5)
    with pytest.raises(ValueError, match="threshold"):
        generate_grounded_answer("query", strategy="fixed", threshold=float("nan"))


def test_partial_fragments_are_discarded_not_spliced(monkeypatch):
    retrieved = [
        {
            **SUPPORTED_RESULT,
            "chunk_id": "KB-002-fixed-000",
            "text": (
                "The initial response SLA measures when support first replies, "
                "while the resolution target"
            ),
            "metadata": {"chunk_strategy": "fixed", "chunk_index": 0},
        },
        {
            **SUPPORTED_RESULT,
            "chunk_id": "KB-002-fixed-001",
            "text": (
                "measures the intended time to resolve the issue. "
                "Escalate P1 tickets immediately to the Incident Manager."
            ),
            "metadata": {"chunk_strategy": "fixed", "chunk_index": 1},
        },
    ]
    monkeypatch.setattr(
        grounded_generation,
        "retrieve",
        lambda query, strategy, top_k: retrieved,
    )

    result = generate_grounded_answer(
        "What is the response SLA?",
        strategy="fixed",
        threshold=0.3259601891040802,
    )

    assert result["answer"] == FALLBACK_ANSWER
    assert result["fallback_triggered"] is True
    assert "resolution target measures" not in result["answer"]
    assert "Escalate P1" not in result["answer"]


def test_fixed_p1_sla_prefers_complete_answer_and_separates_top1_from_sources(
    monkeypatch,
):
    documents = {
        document["document_id"]: document for document in load_knowledge_base()
    }
    retrieved = [
        {
            "rank": 1,
            "chunk_id": "KB-003-fixed-000",
            "document_id": "KB-003",
            "text": "Escalate P1 tickets immediately to the Incident Manager.",
            "metadata": {"chunk_strategy": "fixed", "chunk_index": 0},
            "distance": 0.1,
        },
        {
            "rank": 2,
            "chunk_id": "KB-002-fixed-000",
            "document_id": "KB-002",
            "text": documents["KB-002"]["text"],
            "metadata": {"chunk_strategy": "fixed", "chunk_index": 0},
            "distance": 0.2,
        },
    ]
    monkeypatch.setattr(
        grounded_generation,
        "retrieve",
        lambda query, strategy, top_k: retrieved[:top_k],
    )
    query = "What is the response SLA for a P1 critical ticket?"

    result = generate_grounded_answer(
        query,
        strategy="fixed",
        threshold=0.3259601891040802,
        top_k=3,
    )

    assert result["answer"] == (
        "For P1, the initial response is due within 15 minutes and the target "
        "resolution time is within 4 hours."
    )
    assert "Escalate P1" not in result["answer"]
    assert "15 minutes" in result["answer"]
    assert "4 hours" in result["answer"]
    assert "1 hour" not in result["answer"]
    assert "12 hours" not in result["answer"]
    assert result["top1_document_id"] == "KB-003"
    assert result["top1_chunk_id"] == "KB-003-fixed-000"
    assert result["sources"] == [
        {
            "document_id": "KB-002",
            "chunk_id": "KB-002-fixed-000",
            "similarity": pytest.approx(0.8),
            "retrieval_origin": "vector_search",
        }
    ]
    assert result["sources"][0]["chunk_id"] in {
        item["chunk_id"] for item in retrieved
    }


def test_p1_sla_prefers_exact_numeric_sentence_when_it_is_in_retrieved_top_five(
    monkeypatch,
):
    documents = {
        document["document_id"]: document for document in load_knowledge_base()
    }
    sla_sentences = documents["KB-002"]["text"].splitlines()
    numeric_sentence = sla_sentences[0]
    generic_sentence = sla_sentences[4]
    p1_chunk = f"{sla_sentences[0]} {sla_sentences[1]}"
    retrieved = [
        {
            "rank": 1,
            "chunk_id": "KB-002-sentence-002",
            "document_id": "KB-002",
            "text": generic_sentence,
            "metadata": {"chunk_strategy": "sentence", "chunk_index": 2},
            "distance": 0.1,
        },
        {
            "rank": 2,
            "chunk_id": "KB-003-sentence-000",
            "document_id": "KB-003",
            "text": "Escalate P1 tickets immediately to the Incident Manager.",
            "metadata": {"chunk_strategy": "sentence", "chunk_index": 0},
            "distance": 0.2,
        },
        {
            "rank": 3,
            "chunk_id": "KB-002-sentence-000",
            "document_id": "KB-002",
            "text": p1_chunk,
            "metadata": {"chunk_strategy": "sentence", "chunk_index": 0},
            "distance": 0.25,
        },
        {
            "rank": 4,
            "chunk_id": "KB-010-sentence-000",
            "document_id": "KB-010",
            "text": "VIP status changes response prioritisation.",
            "metadata": {"chunk_strategy": "sentence", "chunk_index": 0},
            "distance": 0.3,
        },
        {
            "rank": 5,
            "chunk_id": "KB-006-sentence-000",
            "document_id": "KB-006",
            "text": "Support is available during standard business hours.",
            "metadata": {"chunk_strategy": "sentence", "chunk_index": 0},
            "distance": 0.32,
        },
    ]
    monkeypatch.setattr(
        grounded_generation,
        "retrieve",
        lambda query, strategy, top_k: retrieved[:top_k],
    )
    query = "What is the response SLA for a P1 critical ticket?"

    first = generate_grounded_answer(
        query,
        strategy="sentence",
        threshold=0.33928388357162476,
        top_k=5,
    )
    second = generate_grounded_answer(
        query,
        strategy="sentence",
        threshold=0.33928388357162476,
        top_k=5,
    )

    assert numeric_sentence == (
        "For P1, the initial response is due within 15 minutes and the target "
        "resolution time is within 4 hours."
    )
    assert first == second
    assert first["answer"] == numeric_sentence
    assert "15 minutes" in first["answer"]
    assert "4 hours" in first["answer"]
    assert generic_sentence not in first["answer"]
    assert first["top1_chunk_id"] == "KB-002-sentence-002"
    assert first["sources"] == [
        {
            "document_id": "KB-002",
            "chunk_id": "KB-002-sentence-000",
            "similarity": pytest.approx(0.75),
            "retrieval_origin": "vector_search",
        }
    ]
    assert first["sources"][0]["chunk_id"] in {
        item["chunk_id"] for item in retrieved
    }


def test_p1_sla_does_not_invent_numeric_targets_when_retrieved_context_lacks_them(
    monkeypatch,
):
    generic_sentence = (
        "The initial response SLA measures when support first replies, while the "
        "resolution target measures the intended time to resolve the issue."
    )
    retrieved = [
        {
            "rank": 1,
            "chunk_id": "KB-002-sentence-002",
            "document_id": "KB-002",
            "text": generic_sentence,
            "metadata": {"chunk_strategy": "sentence", "chunk_index": 2},
            "distance": 0.1,
        }
    ]
    monkeypatch.setattr(
        grounded_generation,
        "retrieve",
        lambda query, strategy, top_k: retrieved[:top_k],
    )

    result = generate_grounded_answer(
        "What is the response SLA for a P1 critical ticket?",
        strategy="sentence",
        threshold=0.33928388357162476,
        top_k=5,
    )

    assert result["answer"] == generic_sentence
    assert result["grounded"] is True
    assert "15 minutes" not in result["answer"]
    assert "4 hours" not in result["answer"]
    assert result["sources"][0]["chunk_id"] == "KB-002-sentence-002"


def test_sla_query_falls_back_when_retrieved_context_has_no_sla_sentence(monkeypatch):
    retrieved = [{
        "rank": 1,
        "chunk_id": "KB-003-sentence-000",
        "document_id": "KB-003",
        "text": "Escalate P1 tickets immediately to the Incident Manager.",
        "metadata": {"chunk_strategy": "sentence", "chunk_index": 0},
        "distance": 0.1,
    }]
    monkeypatch.setattr(
        grounded_generation,
        "retrieve",
        lambda query, strategy, top_k: retrieved[:top_k],
    )

    result = generate_grounded_answer(
        "What is the response SLA for a P1 critical ticket?",
        strategy="sentence",
        threshold=0.33928388357162476,
        top_k=3,
    )

    assert result["answer"] == FALLBACK_ANSWER
    assert result["fallback_triggered"] is True
    assert result["sources"] == []
    assert result["top1_document_id"] == "KB-003"


def test_outage_answer_is_extractive_and_deterministic_without_eta(monkeypatch):
    documents = {
        document["document_id"]: document for document in load_knowledge_base()
    }
    outage_text = documents["KB-011"]["text"]
    retrieved = [{
        "rank": 1,
        "chunk_id": "KB-011-fixed-000",
        "document_id": "KB-011",
        "text": outage_text,
        "metadata": {"chunk_strategy": "fixed", "chunk_index": 0},
        "distance": 0.2,
    }]
    monkeypatch.setattr(
        grounded_generation,
        "retrieve",
        lambda query, strategy, top_k: retrieved[:top_k],
    )
    query = "How should customers be informed during a widespread outage?"

    first = generate_grounded_answer(
        query, strategy="fixed", threshold=0.3259601891040802
    )
    second = generate_grounded_answer(
        query, strategy="fixed", threshold=0.3259601891040802
    )

    assert first == second
    assert first["answer"] == (
        "Initial customer communication should acknowledge the issue without "
        "inventing an estimated time of arrival."
    )
    assert first["answer"] in outage_text
    assert first["grounded"] is True
