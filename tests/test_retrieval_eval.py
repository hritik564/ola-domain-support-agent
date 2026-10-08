import json
from pathlib import Path

import pytest

from evaluation.retrieval_eval import (
    STRATEGIES,
    TOP_K,
    deduplicate_preserving_order,
    evaluate_query,
    evaluate_retrieval,
    load_evaluation_queries,
    pending_evaluation,
    select_recommended_strategy,
)
from rag.config import (
    PRODUCTION_SELECTION_STATUS,
    PRODUCTION_STRATEGY,
    PRODUCTION_TOP_K,
    PRODUCTION_THRESHOLD,
)


def test_evaluation_query_file_has_exact_five_queries_and_relevant_parents():
    queries = load_evaluation_queries()
    assert [(query["query_id"], query["query"]) for query in queries] == [
        ("Q1", "What is the response SLA for a P1 critical ticket?"),
        ("Q2", "When should a repeated unresolved complaint be escalated?"),
        ("Q3", "When can service credit be considered?"),
        ("Q4", "What support is available outside standard business hours?"),
        ("Q5", "How should customers be informed during a widespread outage?"),
    ]
    assert [query["relevant_document_ids"] for query in queries] == [
        ["KB-002"],
        ["KB-003", "KB-007"],
        ["KB-008"],
        ["KB-006"],
        ["KB-011"],
    ]


def test_parent_document_deduplication_preserves_first_retrieval_order():
    assert deduplicate_preserving_order(
        ["KB-007", "KB-007", "KB-003", "KB-007", "KB-002"]
    ) == ["KB-007", "KB-003", "KB-002"]


def test_query_precision_and_recall_are_scored_after_document_deduplication():
    query = {
        "query_id": "Q1",
        "query": "What is the response SLA for a P1 critical ticket?",
        "relevant_document_ids": ["KB-002"],
    }

    def fake_retrieve(query_text, strategy, top_k):
        assert strategy == "fixed"
        assert top_k == 3
        return [
            {"chunk_id": "KB-002-fixed-000", "document_id": "KB-002"},
            {"chunk_id": "KB-002-fixed-001", "document_id": "KB-002"},
            {"chunk_id": "KB-003-fixed-000", "document_id": "KB-003"},
        ]

    result = evaluate_query(query, "fixed", fake_retrieve)
    assert result["raw_retrieved_chunk_ids"] == [
        "KB-002-fixed-000",
        "KB-002-fixed-001",
        "KB-003-fixed-000",
    ]
    assert result["retrieved_parent_document_ids_before_dedup"] == [
        "KB-002",
        "KB-002",
        "KB-003",
    ]
    assert result["unique_retrieved_parent_document_ids"] == ["KB-002", "KB-003"]
    assert result["true_positives"] == 1
    assert (result["precision_numerator"], result["precision_denominator"]) == (1, 2)
    assert result["precision"] == 0.5
    assert (result["recall_numerator"], result["recall_denominator"]) == (1, 1)
    assert result["recall"] == 1.0


def test_same_five_queries_run_against_both_strategies_and_select_measured_winner():
    queries = load_evaluation_queries()
    expected_by_query = {
        "Q1": ["KB-002"],
        "Q2": ["KB-003", "KB-007"],
        "Q3": ["KB-008"],
        "Q4": ["KB-006"],
        "Q5": ["KB-011"],
    }
    seen_calls = []

    def fake_retrieve(query_text, strategy, top_k):
        assert top_k == TOP_K
        query_id = next(
            record["query_id"] for record in queries
            if record["query"] == query_text
        )
        seen_calls.append((strategy, query_id))
        if query_id == "Q1":
            parent_ids = (
                ["KB-002", "KB-002", "KB-003"]
                if strategy == "fixed"
                else ["KB-003"]
            )
        elif query_id == "Q2":
            parent_ids = (
                ["KB-007", "KB-007", "KB-003"]
                if strategy == "fixed"
                else ["KB-007", "KB-003", "KB-999"]
            )
        else:
            parent_ids = expected_by_query[query_id]
        return [
            {
                "chunk_id": f"{document_id}-{strategy}-{index:03d}",
                "document_id": document_id,
            }
            for index, document_id in enumerate(parent_ids)
        ]

    results = evaluate_retrieval(retrieve_fn=fake_retrieve)
    assert set(results["strategies"]) == set(STRATEGIES)
    assert seen_calls == [
        (strategy, query["query_id"])
        for strategy in STRATEGIES
        for query in queries
    ]
    assert len(results["strategies"]["fixed"]["per_query"]) == 5
    assert results["strategies"]["fixed"]["aggregate"] == {
        "mean_precision": 0.9,
        "mean_recall": 1.0,
    }
    assert results["strategies"]["sentence"]["aggregate"]["mean_recall"] == 0.8
    recommendation = results["production_strategy_recommendation"]
    assert recommendation["status"] == "recommended"
    assert recommendation["strategy"] == "fixed"
    assert recommendation["threshold"] == 0.3259601891040802


def test_recommendation_tie_does_not_arbitrarily_choose():
    per_query = [
        {
            "query_id": "Q3",
            "relevant_document_ids": ["KB-008"],
            "unique_retrieved_parent_document_ids": ["KB-008"],
        }
    ]
    results = {
        strategy: {
            "per_query": per_query,
            "aggregate": {"mean_precision": 1.0, "mean_recall": 1.0},
        }
        for strategy in STRATEGIES
    }
    recommendation = select_recommended_strategy(results)
    assert recommendation["status"] == "tie_review_required"
    assert recommendation["strategy"] is None
    assert recommendation["threshold"] is None


def test_strategy_selection_prioritizes_recall_over_precision():
    per_query = [
        {
            "query_id": "Q1",
            "relevant_document_ids": ["KB-002"],
            "unique_retrieved_parent_document_ids": ["KB-002"],
        }
    ]
    results = {
        "fixed": {
            "per_query": per_query,
            "aggregate": {"mean_precision": 0.4, "mean_recall": 0.8},
        },
        "sentence": {
            "per_query": per_query,
            "aggregate": {"mean_precision": 0.9, "mean_recall": 0.6},
        },
    }
    recommendation = select_recommended_strategy(results)
    assert recommendation["strategy"] == "fixed"
    assert "mean recall was 0.8000 versus 0.6000" in recommendation["reason"]
    assert "mean precision was 0.4000 versus 0.9000" in recommendation["reason"]


def test_strategy_selection_uses_precision_when_recall_ties():
    per_query = [
        {
            "query_id": "Q3",
            "relevant_document_ids": ["KB-008"],
            "unique_retrieved_parent_document_ids": ["KB-008"],
        }
    ]
    results = {
        "fixed": {
            "per_query": per_query,
            "aggregate": {"mean_precision": 0.7, "mean_recall": 0.8},
        },
        "sentence": {
            "per_query": per_query,
            "aggregate": {"mean_precision": 0.6, "mean_recall": 0.8},
        },
    }
    recommendation = select_recommended_strategy(results)
    assert recommendation["strategy"] == "fixed"
    assert "Mean recall tied at 0.8000" in recommendation["reason"]


def test_missing_chroma_evidence_stays_pending_without_scores():
    pending = pending_evaluation("ChromaDB is required.")
    assert pending["status"] == "pending_chromadb_validation"
    assert len(pending["queries"]) == 5
    assert all(
        pending["strategies"][strategy]["per_query"] == []
        and pending["strategies"][strategy]["aggregate"] is None
        for strategy in STRATEGIES
    )
    assert pending["production_strategy_recommendation"]["strategy"] is None


def test_production_config_uses_locked_external_evaluation():
    assert PRODUCTION_STRATEGY == "sentence"
    assert PRODUCTION_THRESHOLD == 0.33928388357162476
    assert PRODUCTION_TOP_K == 5
    assert TOP_K == 3
    assert PRODUCTION_SELECTION_STATUS == "external_validation_complete"


def test_external_results_preserve_reported_metrics_and_provenance():
    results_path = Path(__file__).resolve().parents[1] / "data" / (
        "retrieval_evaluation_results.json"
    )
    results = json.loads(results_path.read_text(encoding="utf-8"))
    assert results["status"] == "completed_external_validation"
    assert results["measured_inside_replit"] is False
    assert results["strategies"]["fixed"]["aggregate"] == {
        "mean_precision": 0.4,
        "mean_recall": 1.0,
    }
    assert results["strategies"]["sentence"]["aggregate"] == {
        "mean_precision": 0.6,
        "mean_recall": 1.0,
    }
    assert results["production_strategy_recommendation"]["strategy"] == "sentence"
    assert results["production_strategy_recommendation"]["threshold"] == (
        0.33928388357162476
    )
    sentence_q1 = results["strategies"]["sentence"]["per_query"][0]
    assert sentence_q1["precision_numerator"] == 1
    assert sentence_q1["precision_denominator"] == 2
    assert sentence_q1["precision"] == 0.5
    assert sentence_q1["retrieved_parent_document_ids_before_dedup"] is None
    assert sentence_q1["retrieval_identity_detail_status"] == (
        "not_provided_in_external_validation_summary"
    )


def test_query_loader_rejects_a_non_five_query_file(tmp_path):
    path = tmp_path / "queries.json"
    path.write_text(json.dumps([]), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly five"):
        load_evaluation_queries(path)
