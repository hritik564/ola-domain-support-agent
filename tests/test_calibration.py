import json
from pathlib import Path

from rag.calibration import (
    CALIBRATION_QUERIES,
    CALIBRATION_STRATEGIES,
    run_calibration,
    select_empirical_threshold,
)


def _fake_retriever(query, strategy, top_k):
    assert top_k == 1
    query_id = next(
        item["query_id"] for item in CALIBRATION_QUERIES if item["query"] == query
    )
    similarities = {
        "fixed": {"Q1": 0.82, "Q2": 0.76, "Q3": 0.79, "OOS1": 0.28, "OOS2": 0.34},
        "sentence": {
            "Q1": 0.84,
            "Q2": 0.74,
            "Q3": 0.80,
            "OOS1": 0.30,
            "OOS2": 0.36,
        },
    }
    similarity = similarities[strategy][query_id]
    expected_id = next(
        (
            document_id
            for item in CALIBRATION_QUERIES
            if item["query_id"] == query_id
            for document_id in item["expected_document_ids"]
        ),
        None,
    )
    return [
        {
            "document_id": expected_id or "KB-001",
            "chunk_id": f"{expected_id or 'KB-001'}-{strategy}-000",
            "distance": 1.0 - similarity,
            # Calibration must recompute from distance rather than trust this value.
            "similarity": -123.0,
        }
    ]


def test_required_calibration_queries_and_scopes_are_present():
    assert {item["query_id"] for item in CALIBRATION_QUERIES} == {
        "Q1", "Q2", "Q3", "OOS1", "OOS2"
    }
    assert sum(item["expected_scope"] == "in_scope" for item in CALIBRATION_QUERIES) >= 3
    assert sum(item["expected_scope"] == "out_of_scope" for item in CALIBRATION_QUERIES) >= 2
    assert [item["query"] for item in CALIBRATION_QUERIES] == [
        "What is the response SLA for a P1 critical ticket?",
        "When should a repeated unresolved complaint be escalated?",
        "When can service credit be considered?",
        "Who won the FIFA World Cup?",
        "How do I bake a chocolate cake?",
    ]


def test_calibration_measures_both_strategies_and_derives_similarity_from_distance():
    result = run_calibration(retrieve_fn=_fake_retriever)
    assert set(result["strategies"]) == set(CALIBRATION_STRATEGIES)
    for strategy in CALIBRATION_STRATEGIES:
        measurements = result["strategies"][strategy]["measurements"]
        assert len(measurements) == 5
        assert all("top1_similarity" in item for item in measurements)
        assert all(
            abs(item["top1_similarity"] - (1 - item["top1_distance"])) < 1e-12
            for item in measurements
        )
        assert all(item["top1_similarity"] != -123.0 for item in measurements)
        assert result["strategies"][strategy]["threshold"]["status"] == "separated"


def test_threshold_is_midpoint_of_observed_separated_ranges():
    measurements = [
        {"expected_scope": "in_scope", "top1_similarity": 0.81},
        {"expected_scope": "in_scope", "top1_similarity": 0.72},
        {"expected_scope": "out_of_scope", "top1_similarity": 0.31},
        {"expected_scope": "out_of_scope", "top1_similarity": 0.40},
    ]
    result = select_empirical_threshold(measurements)
    assert result["status"] == "separated"
    assert result["min_in_scope_similarity"] == 0.72
    assert result["max_out_of_scope_similarity"] == 0.40
    assert result["selected_threshold"] == (0.40 + 0.72) / 2
    assert 0.40 < result["selected_threshold"] < 0.72


def test_overlapping_clusters_require_review_without_inventing_threshold():
    measurements = [
        {"expected_scope": "in_scope", "top1_similarity": 0.61},
        {"expected_scope": "out_of_scope", "top1_similarity": 0.63},
    ]
    result = select_empirical_threshold(measurements)
    assert result["status"] == "overlap_review_required"
    assert result["selected_threshold"] is None
    assert result["min_in_scope_similarity"] == 0.61
    assert result["max_out_of_scope_similarity"] == 0.63
    assert "overlap" in result["message"]


def test_missing_measurements_produce_insufficient_data_state():
    result = select_empirical_threshold(
        [
            {"expected_scope": "in_scope", "top1_similarity": 0.8},
            {"expected_scope": "in_scope", "top1_similarity": 0.7},
            {"expected_scope": "in_scope", "top1_similarity": None},
            {"expected_scope": "out_of_scope", "top1_similarity": 0.2},
            {"expected_scope": "out_of_scope", "top1_similarity": 0.3},
        ]
    )
    assert result["status"] == "insufficient_data"
    assert result["selected_threshold"] is None
    assert result["max_out_of_scope_similarity"] == 0.3


def test_external_calibration_artifact_preserves_supplied_measurements():
    artifact_path = (
        Path(__file__).resolve().parents[1] / "data" / "calibration_results.json"
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["measured_inside_replit"] is False
    assert artifact["production_strategy_selected"] is False

    expected = {
        "fixed": {
            "min_in_scope_similarity": 0.5551766157150269,
            "max_out_of_scope_similarity": 0.09674376249313354,
            "selected_threshold": 0.3259601891040802,
            "similarities": [
                0.5551766157150269,
                0.7071335315704346,
                0.6972638964653015,
                0.09674376249313354,
                0.02743762731552124,
            ],
        },
        "sentence": {
            "min_in_scope_similarity": 0.5762889385223389,
            "max_out_of_scope_similarity": 0.10227882862091064,
            "selected_threshold": 0.33928388357162476,
            "similarities": [
                0.5762889385223389,
                0.6403507590293884,
                0.7027689814567566,
                0.10227882862091064,
                0.02854984998703003,
            ],
        },
    }
    for strategy, expected_values in expected.items():
        measurements = artifact["strategies"][strategy]["measurements"]
        threshold = artifact["strategies"][strategy]["threshold"]
        assert [item["top1_similarity"] for item in measurements] == (
            expected_values["similarities"]
        )
        assert threshold["min_in_scope_similarity"] == (
            expected_values["min_in_scope_similarity"]
        )
        assert threshold["max_out_of_scope_similarity"] == (
            expected_values["max_out_of_scope_similarity"]
        )
        assert threshold["selected_threshold"] == (
            expected_values["selected_threshold"]
        )
        assert all(
            1.0 - item["top1_distance"] == item["top1_similarity"]
            for item in measurements
        )
