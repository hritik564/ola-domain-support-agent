import json
from types import SimpleNamespace

import pytest

from agents.crew import (
    CREWAI_AVAILABLE,
    SESSION_MEMORY,
    _kickoff_payload,
    run_support_crew,
)
from agents.schemas import SupportResponse


def test_kickoff_payload_normalizes_crewai_result_shapes():
    payload = {
        "answer": "The response window is one hour.",
        "route": "policy",
        "grounded": True,
        "fallback_triggered": False,
        "sources": [],
    }
    structured = SupportResponse.model_validate(payload)
    expected = structured.model_dump()
    shapes = [
        payload,
        json.dumps(payload),
        SimpleNamespace(raw=json.dumps(payload)),
        SimpleNamespace(json_dict=payload),
        SimpleNamespace(pydantic=structured),
    ]
    for result in shapes:
        assert SupportResponse.model_validate(_kickoff_payload(result)).model_dump() == expected


def test_telemetry_is_disabled_before_crewai_import():
    import os

    assert os.environ["CREWAI_DISABLE_TELEMETRY"] == "true"
    assert os.environ["OTEL_SDK_DISABLED"] == "true"


def test_missing_crewai_is_reported_without_fake_kickoff_or_tool_counts():
    if CREWAI_AVAILABLE:
        pytest.skip("CrewAI is installed; this checks the Replit blocked-dependency path.")
    outcome = run_support_crew(
        "What is the response SLA for a P1 critical ticket?",
        session_id="crew-blocked-test",
    )
    assert outcome.response is None
    assert outcome.audit.status == "blocked_dependency"
    assert outcome.audit.kickoff_called is False
    assert outcome.audit.tool_invocation_count == 0
    assert "CrewAI" in outcome.audit.error
    assert "stage=crew_construction" in outcome.audit.error


def test_session_routing_resolves_followup_and_isolates_other_session():
    SESSION_MEMORY.clear_session("crew-session-A")
    SESSION_MEMORY.clear_session("crew-session-B")
    first = run_support_crew(
        "Check status of TKT-0007.",
        session_id="crew-session-A",
    )
    followup = run_support_crew(
        "Should that ticket be escalated?",
        session_id="crew-session-A",
    )
    other_session = run_support_crew(
        "Should that ticket be escalated?",
        session_id="crew-session-B",
    )

    assert first.audit.route == "ticket"
    assert followup.audit.route == "ticket"
    if CREWAI_AVAILABLE:
        assert first.audit.kickoff_called is True
        assert first.response is not None, first.audit.error
        assert first.audit.status == "completed", first.audit.error
        assert first.response.record_id == "TKT-0007"
        assert followup.audit.kickoff_called is True
        assert followup.audit.status == "completed"
        assert followup.response is not None, followup.audit.error
        assert followup.response.route == "ticket"
        assert followup.response.record_id == "TKT-0007"
        assert followup.audit.tool_invocation_count == 1, followup.audit.error
        assert [
            item.tool_name for item in followup.audit.tool_invocations
        ] == ["ticket_status_lookup"], followup.audit.error
    else:
        assert first.audit.status == "blocked_dependency"
        assert followup.audit.status == "blocked_dependency"
        assert followup.audit.kickoff_called is False
    assert other_session.audit.route == "fallback"
    assert other_session.audit.kickoff_called is False
    assert other_session.audit.tool_invocation_count == 0
    assert other_session.response is not None
    assert other_session.response.record_id is None
    assert other_session.response.fallback_triggered is True


@pytest.mark.parametrize(
    "query",
    [
        "Can you recommend a recipe for dinner?",
        "What is tomorrow's cricket score?",
    ],
)
def test_out_of_scope_queries_use_the_exact_policy_fallback(query):
    outcome = run_support_crew(query, session_id=f"oos-{query[:8]}")

    assert outcome.audit.route == "fallback"
    assert outcome.audit.status == "out_of_scope"
    assert outcome.audit.kickoff_called is False
    assert outcome.response is not None
    assert outcome.response.answer == (
        "I don't know based on the available support policies."
    )
    assert outcome.response.route == "fallback"
    assert outcome.response.grounded is False
    assert outcome.response.fallback_triggered is True


def test_invalid_kickoff_output_fails_without_a_fallback_response(monkeypatch):
    import agents.crew as crew_module

    class FakeCrew:
        def kickoff(self, inputs):
            return type(
                "CrewOutput",
                (),
                {
                    "raw": (
                        '{"answer":"broken","route":"ticket",'
                        '"grounded":true,"fallback_triggered":false}'
                    )
                },
            )()

    monkeypatch.setattr(crew_module, "CREWAI_AVAILABLE", True)
    monkeypatch.setattr(crew_module, "MOCK_LLM_USES_CREWAI_BASE", True)
    monkeypatch.setattr(
        crew_module,
        "_agent_bundle",
        lambda **kwargs: (FakeCrew(), {}),
    )
    outcome = run_support_crew(
        "Check status of TKT-0007.",
        session_id="invalid-output-test",
    )
    assert outcome.audit.kickoff_called is True
    assert outcome.audit.status == "kickoff_or_validation_failed"
    assert outcome.response is None
    assert "stage=pydantic_validation" in outcome.audit.error
    assert "ValidationError" in outcome.audit.error


def test_invalid_json_output_reports_final_response_parse_stage(monkeypatch):
    import agents.crew as crew_module

    class FakeCrew:
        def kickoff(self, inputs):
            return SimpleNamespace(raw="not JSON")

    monkeypatch.setattr(crew_module, "CREWAI_AVAILABLE", True)
    monkeypatch.setattr(crew_module, "MOCK_LLM_USES_CREWAI_BASE", True)
    monkeypatch.setattr(
        crew_module,
        "_agent_bundle",
        lambda **kwargs: (FakeCrew(), {}),
    )
    outcome = run_support_crew(
        "What is the response SLA?",
        session_id="invalid-json-output-test",
    )
    assert outcome.response is None
    assert "stage=final_response_parse" in outcome.audit.error
    assert "ValueError" in outcome.audit.error
    assert "not valid JSON" in outcome.audit.error


def test_crew_construction_exception_is_reported_with_stage(monkeypatch):
    import agents.crew as crew_module

    def fail_bundle(**kwargs):
        raise ValueError("construction failed")

    monkeypatch.setattr(crew_module, "CREWAI_AVAILABLE", True)
    monkeypatch.setattr(crew_module, "MOCK_LLM_USES_CREWAI_BASE", True)
    monkeypatch.setattr(crew_module, "_agent_bundle", fail_bundle)
    outcome = run_support_crew(
        "What is the response SLA?",
        session_id="construction-error-test",
    )
    assert outcome.response is None
    assert "stage=crew_construction" in outcome.audit.error
    assert "ValueError: construction failed" in outcome.audit.error


def test_kickoff_exception_is_reported_with_stage(monkeypatch):
    import agents.crew as crew_module

    class FakeCrew:
        def kickoff(self, inputs):
            raise RuntimeError("native kickoff failed")

    monkeypatch.setattr(crew_module, "CREWAI_AVAILABLE", True)
    monkeypatch.setattr(crew_module, "MOCK_LLM_USES_CREWAI_BASE", True)
    monkeypatch.setattr(
        crew_module,
        "_agent_bundle",
        lambda **kwargs: (FakeCrew(), {}),
    )
    outcome = run_support_crew(
        "What is the response SLA?",
        session_id="kickoff-error-test",
    )
    assert outcome.response is None
    assert "stage=kickoff" in outcome.audit.error
    assert "RuntimeError: native kickoff failed" in outcome.audit.error


@pytest.mark.skipif(
    not CREWAI_AVAILABLE,
    reason="CrewAI is not installed; real kickoff cannot run.",
)
def test_real_ticket_crew_kickoff_records_ticket_tool_call():
    outcome = run_support_crew(
        "Check status of TKT-0007.",
        session_id="real-ticket-kickoff-test",
    )
    assert outcome.audit.kickoff_called is True
    assert outcome.audit.status == "completed", outcome.audit.error
    assert outcome.response is not None, outcome.audit.error
    assert outcome.response.record_id == "TKT-0007"
    ticket_calls = [
        item
        for item in outcome.audit.tool_invocations
        if item.tool_name == "ticket_status_lookup"
    ]
    rag_calls = [
        item
        for item in outcome.audit.tool_invocations
        if item.tool_name == "rag_lookup"
    ]
    assert outcome.audit.tool_invocation_count == 1, outcome.audit.error
    assert len(ticket_calls) == 1, outcome.audit.error
    assert ticket_calls[0].status == "completed", outcome.audit.error
    assert ticket_calls[0].arguments.get("record_id") == "TKT-0007"
    assert outcome.response.ticket_status == ticket_calls[0].result["status"]
    assert (
        outcome.response.resolution_time_hours
        == ticket_calls[0].result["resolution_time_hours"]
    )
    assert (
        outcome.response.ticket_age_days
        == ticket_calls[0].result["days_since_created"]
    )
    assert rag_calls == [], outcome.audit.error


@pytest.mark.skipif(
    not CREWAI_AVAILABLE,
    reason="CrewAI is not installed; real kickoff cannot run.",
)
def test_real_policy_crew_kickoff_records_rag_tool_call():
    outcome = run_support_crew(
        "What is the response SLA for a P1 critical ticket?",
        session_id="real-policy-kickoff-test",
    )
    assert outcome.audit.kickoff_called is True
    assert outcome.audit.status == "completed", outcome.audit.error
    assert outcome.response is not None, outcome.audit.error
    rag_calls = [
        item
        for item in outcome.audit.tool_invocations
        if item.tool_name == "rag_lookup"
    ]
    ticket_calls = [
        item
        for item in outcome.audit.tool_invocations
        if item.tool_name == "ticket_status_lookup"
    ]
    assert outcome.audit.tool_invocation_count == 1, outcome.audit.error
    assert len(rag_calls) == 1, outcome.audit.error
    assert rag_calls[0].status == "completed", outcome.audit.error
    assert ticket_calls == [], outcome.audit.error
    assert outcome.response.route == "policy"
    assert outcome.response.grounded is True
    assert outcome.response.fallback_triggered is False
    if outcome.response.sources:
        assert "15 minutes" in outcome.response.answer.lower()
        assert "4 hours" in outcome.response.answer.lower()
