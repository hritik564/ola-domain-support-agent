import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from agents.audit import new_audit
from agents.mock_llm import (
    MOCK_LLM_USES_CREWAI_BASE,
    MockLLM,
    MockLLMRuntimeError,
    ToolResultNormalizationError,
    _parse_tool_result_payload,
    _tool_result_messages,
)


def _function_tool(name, argument):
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {argument: {"type": "string"}},
                "required": [argument],
            },
        },
    }


def test_react_observation_example_is_not_a_runtime_tool_result():
    messages = [
        {
            "role": "system",
            "content": "Example only: Observation: the result of the action",
        },
        {"role": "user", "content": "What is the policy?"},
    ]
    assert _tool_result_messages(messages) == []


def test_dispatch_uses_argument_schema_not_tool_name_substrings():
    llm = MockLLM(route="policy", query="What is the response SLA?")
    calls = []

    def rag_function(query):
        calls.append({"query": query})
        return {"answer": "grounded result"}

    # The misleading name contains "ticket", but its declared argument is query.
    tool = _function_tool("rag_ticket_lookup", "query")
    result = llm.call(
        [
            {"role": "system", "content": "Observation: example, not a result."},
            {"role": "user", "content": "Question follows."},
        ],
        tools=[tool],
        available_functions={"rag_ticket_lookup": rag_function},
    )
    assert calls == []
    assert result == [
        {
            "id": "mock-tool-call-1",
            "type": "function",
            "function": {
                "name": "rag_ticket_lookup",
                "arguments": '{"query": "What is the response SLA?"}',
            },
        }
    ]


def test_ticket_route_uses_record_id_schema_and_not_rag_query_schema():
    llm = MockLLM(route="ticket", query="Check TKT-0007")
    calls = []

    def ticket_function(record_id):
        calls.append(record_id)
        return {"record_id": record_id}

    rag_tool = _function_tool("rag_lookup", "query")
    ticket_tool = _function_tool("ticket_status_lookup", "record_id")
    result = llm.call(
        [{"role": "user", "content": "Check the ticket."}],
        tools=[rag_tool, ticket_tool],
        available_functions={"ticket_status_lookup": ticket_function},
    )
    assert calls == []
    assert result[0]["function"]["name"] == "ticket_status_lookup"
    assert json.loads(result[0]["function"]["arguments"]) == {
        "record_id": "TKT-0007"
    }


def test_rag_lookup_cannot_be_selected_for_ticket_route():
    llm = MockLLM(route="ticket", query="Check TKT-0007")
    with pytest.raises(MockLLMRuntimeError) as error:
        llm.call(
            [{"role": "user", "content": "Check the ticket."}],
            tools=[_function_tool("rag_lookup", "query")],
        )
    assert error.value.stage == "mock_llm_tool_call"


def test_call_ignores_current_and_future_crewai_runtime_metadata():
    llm = MockLLM(route="policy", query="What is the response SLA?")
    calls = []

    def rag_function(query):
        calls.append(query)
        return {"answer": "grounded result"}

    tool = _function_tool("policy_search", "query")
    messages = [{"role": "user", "content": "What is the response SLA?"}]
    tools = [tool]
    functions = {"policy_search": rag_function}

    baseline = llm.call(
        messages,
        tools=tools,
        available_functions=functions,
    )
    with_metadata = llm.call(
        messages,
        tools=tools,
        available_functions=functions,
        from_task=object(),
        from_agent=object(),
        response_model=None,
        executor_context=object(),
        verbose=False,
        future_runtime_flag=True,
    )

    assert with_metadata == baseline
    assert baseline[0]["function"]["name"] == "policy_search"
    assert json.loads(baseline[0]["function"]["arguments"]) == {
        "query": "What is the response SLA?"
    }
    assert calls == []


def test_tool_role_is_the_only_source_of_runtime_tool_observations():
    assert _tool_result_messages(
        [
            {"role": "assistant", "content": "Observation: not a tool message"},
            {"role": "tool", "content": "actual tool result"},
        ]
    ) == ["actual tool result"]


def test_tool_result_message_objects_and_payloads_are_normalized():
    message = SimpleNamespace(
        type="tool",
        name="ticket_status_lookup",
        tool_call_id="call-1",
        content={"record_id": "TKT-0007", "status": "In Progress"},
    )
    assert _tool_result_messages(message) == [
        {"record_id": "TKT-0007", "status": "In Progress"}
    ]
    assert _tool_result_messages(
        {"role": "function", "name": "rag_lookup", "content": "function result"}
    ) == ["function result"]


def test_tool_result_message_uses_nonempty_explicit_result_when_content_is_empty():
    result = '{"answer":"P1 response SLA","grounded":true}'
    message = SimpleNamespace(
        role="tool",
        content="",
        result=result,
        tool_call_id="call-1",
    )
    assert _tool_result_messages(message) == [result]


def test_assistant_tool_calls_are_not_mistaken_for_tool_results():
    assert _tool_result_messages(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "rag_lookup", "arguments": "{}"},
                    }
                ],
            }
        ]
    ) == []


def test_tool_result_normalizer_accepts_json_text_and_mapping():
    evidence = {
        "answer": "P1 response SLA is 15 minutes.",
        "grounded": True,
        "fallback_triggered": False,
        "sources": [{"document_id": "KB-001", "chunk_id": "KB-001-S01"}],
    }
    assert _parse_tool_result_payload(json.dumps(evidence)) == evidence
    assert _parse_tool_result_payload(evidence) == evidence


def test_tool_result_normalizer_accepts_pydantic_json_mode():
    class RAGEvidence(BaseModel):
        answer: str
        grounded: bool
        fallback_triggered: bool
        sources: list[dict[str, str]]

    evidence = RAGEvidence(
        answer="P1 response SLA is 15 minutes.",
        grounded=True,
        fallback_triggered=False,
        sources=[{"document_id": "KB-001", "chunk_id": "KB-001-S01"}],
    )
    assert _parse_tool_result_payload(evidence) == evidence.model_dump(mode="json")


def test_tool_result_normalizer_accepts_bytes_json_fence_and_tool_framing():
    object_text = (
        '{"answer":"P1 response SLA is 15 minutes.",'
        '"grounded":true,"fallback_triggered":false,"sources":[]}'
    )
    expected = json.loads(object_text)
    assert _parse_tool_result_payload(object_text.encode("utf-8")) == expected
    assert _parse_tool_result_payload(f"```json\n{object_text}\n```") == expected
    assert (
        _parse_tool_result_payload(f"Tool returned successfully: {object_text}")
        == expected
    )
    assert _parse_tool_result_payload(
        [{"type": "text", "text": object_text}]
    ) == expected


def test_invalid_tool_result_error_is_deterministic_and_includes_safe_preview():
    payload = "plain text, not a RAG result"
    errors = []
    for _ in range(2):
        with pytest.raises(ToolResultNormalizationError) as error:
            _parse_tool_result_payload(payload)
        errors.append(str(error.value))
    assert errors[0] == errors[1]
    assert "tool_result_type=str" in errors[0]
    assert "tool_result_preview='plain text, not a RAG result'" in errors[0]


def test_actual_ticket_tool_result_produces_response_with_resolution_time():
    llm = MockLLM(route="ticket", query="Check TKT-0007", record_id="TKT-0007")
    tool_result = {
        "record_id": "TKT-0007",
        "status": "In Progress",
        "resolution_time_hours": 18,
        "days_since_created": 4,
        "escalation_score": 0.72,
        "escalation_recommended": True,
    }
    message = SimpleNamespace(
        type="tool",
        tool_call_id="call-1",
        content=json.dumps(tool_result),
    )
    response = json.loads(llm.call([message]))
    assert response["record_id"] == tool_result["record_id"]
    assert response["ticket_status"] == tool_result["status"]
    assert response["resolution_time_hours"] == tool_result["resolution_time_hours"]
    assert "18 hours" in response["answer"]


def test_actual_rag_tool_result_is_preserved_as_final_policy_response():
    llm = MockLLM(route="policy", query="What is the response SLA?")
    tool_result = {
        "answer": "P1 response time is one hour.",
        "grounded": True,
        "fallback_triggered": False,
        "sources": [
            {
                "document_id": "KB-001",
                "chunk_id": "KB-001-S01",
                "similarity": 0.8,
            }
        ],
    }
    message = SimpleNamespace(
        role="tool",
        tool_call_id="call-1",
        content=tool_result,
    )
    response = json.loads(llm.call(message))
    assert response["answer"] == tool_result["answer"]
    assert response["sources"] == tool_result["sources"]
    assert response["fallback_triggered"] is False


def test_malformed_tool_result_is_reported_with_stage():
    audit = new_audit()
    llm = MockLLM(
        route="policy",
        query="What is the response SLA?",
        audit=audit,
    )
    payload = "not JSON; no policy evidence"
    with pytest.raises(MockLLMRuntimeError) as error:
        llm.call(
            {"role": "tool", "tool_call_id": "call-1", "content": payload}
        )
    assert error.value.stage == "mock_llm_tool_result"
    assert "tool_result_type=str" in audit.error
    assert repr(payload) in audit.error


def test_mock_llm_is_zero_key_and_uses_installed_base_when_available():
    assert isinstance(MOCK_LLM_USES_CREWAI_BASE, bool)
    llm = MockLLM(route="policy", query="What is the response SLA?")
    assert llm.model == "ola-mock-llm"
    assert llm.supports_function_calling()
