import json

from agents.crew import _selected_route
from agents.schemas import SourceReference
from api.schemas import AskResponse
from api.service import APIServiceError
from evaluation.end_to_end_eval import (
    _aggregate,
    evaluate,
    load_cases,
    score_response,
    write_results,
)


def test_fixed_evaluation_has_15_queries_full_kb_coverage_and_required_special_cases():
    cases = load_cases()
    assert len(cases) == 15
    assert {
        doc_id
        for case in cases
        for doc_id in case["expected_document_ids"]
    } == {f"KB-{number:03d}" for number in range(1, 13)}
    assert cases[12]["expected_answer"] == (
        "I don't know based on the available support policies."
    )
    assert cases[13]["expected_answer"] == cases[12]["expected_answer"]
    assert cases[14]["expected_ticket_fields"]["record_id"] == "TKT-0007"


def test_fixed_evaluation_routes_match_support_scope_and_oos_expectations():
    for case in load_cases():
        actual_route, _record_id = _selected_route(
            case["query"],
            f"eval-route-{case['query_id']}",
        )
        assert actual_route == case["expected_route"], (
            f"{case['query_id']} expected {case['expected_route']}, got {actual_route}"
        )


def test_scoring_separates_accuracy_grounding_completeness_and_safety():
    case = load_cases()[0]
    response = AskResponse(
        answer=(
            "P1 Critical applies to a service-wide outage, an account-security "
            "incident, or a payment charge when the service is completely unavailable."
        ),
        route="policy",
        grounded=True,
        fallback_triggered=False,
        sources=[SourceReference(document_id="KB-001", chunk_id="KB-001-chunk-1")],
        pii_masked=False,
        trace_id="trace-score",
        session_id="session-score",
    )
    scores = score_response(case, response)
    assert scores == {
        "accuracy": 1.0,
        "grounding": 1.0,
        "completeness": 1.0,
        "safety": 1.0,
    }
    assert all(0.0 <= score <= 1.0 for score in scores.values())


def test_incorrect_source_and_missing_phrase_reduce_distinct_dimensions():
    case = load_cases()[0]
    response = AskResponse(
        answer="This is a P2 issue.",
        route="policy",
        grounded=True,
        fallback_triggered=False,
        sources=[SourceReference(document_id="KB-002", chunk_id="KB-002-chunk-1")],
        pii_masked=False,
        trace_id="trace-score",
        session_id="session-score",
    )
    scores = score_response(case, response)
    assert scores["accuracy"] == 0.0
    assert scores["grounding"] == 0.0
    assert scores["completeness"] == 0.0


def test_ticket_evaluation_checks_all_dataset_fields_and_phone_safety():
    case = load_cases()[-1]
    response = AskResponse(
        answer="Ticket TKT-0007 is In Progress. It has been open for 93 hours.",
        route="ticket",
        grounded=True,
        fallback_triggered=False,
        record_id="TKT-0007",
        ticket_status="In Progress",
        resolution_time_hours=93,
        ticket_age_days=13,
        escalation_score=0.8016666666666667,
        escalation_recommended=True,
        pii_masked=True,
        trace_id="trace-ticket",
        session_id="session-ticket",
    )
    assert score_response(case, response) == {
        "accuracy": 1.0,
        "grounding": 1.0,
        "completeness": 1.0,
        "safety": 1.0,
    }
    unsafe_response = response.model_copy(
        update={"answer": response.answer + " Contact 9876543210."}
    )
    assert score_response(case, unsafe_response)["safety"] == 0.0


def test_out_of_scope_case_requires_the_exact_policy_fallback():
    case = load_cases()[12]
    response = AskResponse(
        answer="I don't know based on the available support policies.",
        route="fallback",
        grounded=False,
        fallback_triggered=True,
        trace_id="trace-oos",
        session_id="session-oos",
    )
    scores = score_response(case, response)
    assert scores == {
        "accuracy": 1.0,
        "grounding": 1.0,
        "completeness": 1.0,
        "safety": 1.0,
    }


def test_failed_oos_case_result_includes_actual_answer_and_exact_reasons():
    case = load_cases()[12]
    response = AskResponse(
        answer=case["expected_answer"],
        route="policy",
        grounded=False,
        fallback_triggered=True,
        trace_id="trace-oos-failure",
        session_id="session-oos-failure",
    )
    results = evaluate(
        cases=[case],
        support_fn=lambda _request, **_kwargs: response,
    )
    item = results["per_query_results"][0]

    assert item["query"] == case["query"]
    assert item["required_phrases"] == case["required_phrases"]
    assert item["actual_route"] == "policy"
    assert item["actual_document_ids"] == []
    assert item["actual_sources"] == []
    assert item["actual_answer"] == case["expected_answer"]
    assert item["scores"] == {
        "accuracy": 0.0,
        "grounding": 0.0,
        "completeness": 1.0,
        "safety": 1.0,
    }
    assert item["failure_reasons"] == [
        "route mismatch: expected fallback, got policy",
        "out-of-scope grounding requires route=fallback",
    ]


def test_aggregate_averages_only_evaluated_cases():
    records = [
        {"scores": {"accuracy": 1.0, "grounding": 0.5, "completeness": 1.0, "safety": 1.0}},
        {"scores": {"accuracy": 0.5, "grounding": 1.0, "completeness": 0.0, "safety": 1.0}},
        {"scores": None, "status": "blocked_dependency"},
    ]
    assert _aggregate(records) == {
        "accuracy": 0.75,
        "grounding": 0.75,
        "completeness": 0.5,
        "safety": 1.0,
    }


def test_all_cases_are_marked_blocked_when_the_real_support_dependency_is_absent():
    def dependency_blocked(_request, **_kwargs):
        raise APIServiceError(
            status_code=503,
            code="dependency_unavailable",
            message="The support service is temporarily unavailable.",
            trace_id="trace-blocked",
        )

    results = evaluate(support_fn=dependency_blocked)
    assert results["case_count"] == 15
    assert results["pass_count"] == 0
    assert results["fail_count"] == 0
    assert results["blocked_count"] == 15
    assert results["status"] == "blocked_dependency"
    assert results["averages"] == {
        "accuracy": None,
        "grounding": None,
        "completeness": None,
        "safety": None,
    }
    assert all(
        item["status"] == "blocked_dependency"
        for item in results["per_query_results"]
    )
    assert results == evaluate(support_fn=dependency_blocked)


def test_results_and_transcript_are_written_without_raw_query_text(tmp_path):
    results = {
        "status": "blocked_dependency",
        "total_queries": 15,
        "case_count": 15,
        "pass_count": 0,
        "fail_count": 0,
        "blocked_count": 15,
        "averages": {"accuracy": None, "grounding": None, "completeness": None, "safety": None},
        "average_accuracy": None,
        "average_grounding": None,
        "average_completeness": None,
        "average_safety": None,
        "per_query_results": [
            {
                "query_id": "Q15",
                "status": "blocked_dependency",
                "actual_route": None,
                "scores": None,
            }
        ],
    }
    json_path = tmp_path / "data" / "results.json"
    transcript_path = tmp_path / "transcripts" / "results.txt"
    write_results(results, results_path=json_path, transcript_path=transcript_path)
    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    assert parsed == results
    assert "9876543210" not in transcript_path.read_text(encoding="utf-8")
