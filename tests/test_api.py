import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agents.schemas import (
    CrewAudit,
    CrewRunResult,
    SourceReference,
    SupportResponse,
    ToolInvocation,
)
from api.app import create_app
from api.logging import mask_log_text
from api.schemas import AskRequest, AskResponse
from api.service import APIServiceError, ask_support as service_ask_support


def _support_response(
    *,
    route="fallback",
    answer="I don't know based on the available support policies.",
    grounded=False,
    fallback_triggered=True,
    pii_masked=False,
    record_id=None,
    ticket_status=None,
    resolution_time_hours=None,
    ticket_age_days=None,
    escalation_score=None,
    escalation_recommended=None,
    sources=None,
):
    return SupportResponse(
        answer=answer,
        route=route,
        grounded=grounded,
        fallback_triggered=fallback_triggered,
        pii_masked=pii_masked,
        record_id=record_id,
        ticket_status=ticket_status,
        resolution_time_hours=resolution_time_hours,
        ticket_age_days=ticket_age_days,
        escalation_score=escalation_score,
        escalation_recommended=escalation_recommended,
        sources=sources or [],
    )


def _records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _diagnostic_text(value: str | None) -> str | None:
    if not value:
        return None
    safe_text, _ = mask_log_text(value)
    if re.search(
        r"(?i)(system\s+(?:prompt|message|instructions)|<\|system\|>|begin system)",
        safe_text or "",
    ):
        return "[REDACTED: diagnostic contained system-prompt content]"
    return " ".join((safe_text or "").split())[:1200]


def _parsed_audit_error(error: str | None) -> tuple[str | None, str | None]:
    if not error:
        return None, None
    detail = error.split("; ", 1)[-1]
    match = re.match(r"([A-Za-z_][A-Za-z0-9_.]*):\s*(.*)", detail, re.DOTALL)
    if not match:
        return None, _diagnostic_text(detail)
    message = match.group(2).split(" -> caused by ", 1)[0]
    return match.group(1), _diagnostic_text(message)


def _capture_real_crew_calls(monkeypatch) -> dict:
    import api.service as service_module

    original_run = service_module.run_support_crew
    capture = {"results": [], "exceptions": []}

    def capture_run(query: str, *, session_id: str):
        try:
            result = original_run(query, session_id=session_id)
        except Exception as exc:
            capture["exceptions"].append(exc)
            raise
        capture["results"].append(result)
        return result

    monkeypatch.setattr(service_module, "run_support_crew", capture_run)
    return capture


def _real_crew_diagnostics(capture: dict) -> dict:
    results = []
    for result in capture["results"]:
        response = getattr(result, "response", None)
        audit = getattr(result, "audit", None)
        audit_error = getattr(audit, "error", None)
        exception_type, exception_message = _parsed_audit_error(audit_error)
        results.append(
            {
                "type(result)": (
                    f"{type(result).__module__}.{type(result).__qualname__}"
                ),
                "type(result.response)": (
                    f"{type(response).__module__}.{type(response).__qualname__}"
                ),
                "result.audit.status": getattr(audit, "status", None),
                "result.audit.error": _diagnostic_text(audit_error),
                "internal_exception_type": exception_type,
                "internal_exception_message": exception_message,
            }
        )

    raised = [
        {
            "type(result)": None,
            "type(result.response)": None,
            "result.audit.status": None,
            "result.audit.error": None,
            "internal_exception_type": type(exc).__name__,
            "internal_exception_message": _diagnostic_text(str(exc)),
        }
        for exc in capture["exceptions"]
    ]
    return {"crew_results": results, "raised_exceptions": raised}


def _ticket_support_response() -> SupportResponse:
    return _support_response(
        route="ticket",
        answer=(
            "Ticket TKT-0007 is In Progress, has a resolution time of 93.0 hours, "
            "is 13 days old, and escalation is recommended."
        ),
        grounded=True,
        fallback_triggered=False,
        record_id="TKT-0007",
        ticket_status="In Progress",
        resolution_time_hours=93.0,
        ticket_age_days=13,
        escalation_score=0.8016666666666667,
        escalation_recommended=True,
    )


def test_real_crew_failure_diagnostics_are_sanitized():
    result = CrewRunResult(
        response=None,
        audit=CrewAudit(
            trace_id="diagnostic-trace",
            route="ticket",
            status="kickoff_or_validation_failed",
            error="stage=kickoff; ValueError: api_key=private-value",
        ),
    )

    diagnostic = _real_crew_diagnostics({"results": [result], "exceptions": []})
    crew_result = diagnostic["crew_results"][0]

    assert crew_result["type(result)"].endswith(".CrewRunResult")
    assert crew_result["type(result.response)"] == "builtins.NoneType"
    assert crew_result["result.audit.status"] == "kickoff_or_validation_failed"
    assert crew_result["internal_exception_type"] == "ValueError"
    assert crew_result["internal_exception_message"] == "api_key=[REDACTED]"
    assert "private-value" not in json.dumps(diagnostic)


def test_valid_ask_returns_existing_support_fields_and_trace_id(tmp_path, monkeypatch):
    from api import app as app_module

    response = _support_response()
    async def response_for(request, **kwargs):
        return AskResponse(
            **response.model_dump(),
            trace_id=kwargs["trace_id"],
            session_id=request.session_id or "session-test",
        )

    monkeypatch.setattr(app_module, "ask_support_async", response_for)
    client = TestClient(create_app(log_path=tmp_path / "requests.jsonl"))

    result = client.post("/ask", json={"query": "What is the fallback?"})

    assert result.status_code == 200
    payload = result.json()
    assert payload["answer"] == response.answer
    assert payload["route"] == "fallback"
    assert payload["trace_id"] == result.headers["x-trace-id"]
    assert payload["session_id"] == "session-test"
    assert set(
        (
            "answer",
            "route",
            "grounded",
            "fallback_triggered",
            "sources",
            "record_id",
            "ticket_status",
            "resolution_time_hours",
            "ticket_age_days",
            "escalation_score",
            "escalation_recommended",
            "pii_masked",
        )
    ) <= set(payload)


def test_ask_ticket_fields_are_exposed_at_the_top_level(tmp_path, monkeypatch):
    from api import app as app_module

    ticket = _ticket_support_response()
    async def ticket_response_for(request, **kwargs):
        return AskResponse(
            **ticket.model_dump(),
            trace_id=kwargs["trace_id"],
            session_id=request.session_id or "session-ticket",
        )

    monkeypatch.setattr(app_module, "ask_support_async", ticket_response_for)
    client = TestClient(create_app(log_path=tmp_path / "requests.jsonl"))

    result = client.post(
        "/ask",
        json={"query": "Check the status of TKT-0007.", "session_id": "session-ticket"},
    )

    assert result.status_code == 200, result.text
    payload = result.json()
    assert payload["route"] == "ticket", payload
    assert payload["record_id"] == "TKT-0007", payload
    assert payload["ticket_status"] == "In Progress", payload
    assert payload["resolution_time_hours"] == 93.0, payload
    assert payload["ticket_age_days"] == 13, payload
    assert payload["escalation_score"] == 0.8016666666666667, payload
    assert payload["escalation_recommended"] is True, payload
    assert payload["trace_id"] == result.headers["x-trace-id"], payload
    assert payload["session_id"] == "session-ticket", payload
    assert "response" not in payload


def test_ask_validation_is_safe_and_logged(tmp_path):
    log_path = tmp_path / "nested" / "requests.jsonl"
    client = TestClient(create_app(log_path=log_path))

    result = client.post("/ask", json={"query": "   ", "phone": "9876543210"})

    assert result.status_code == 422
    assert result.json()["code"] == "invalid_request"
    assert result.json()["trace_id"] == result.headers["x-trace-id"]
    assert "9876543210" not in result.text
    entry = _records(log_path)[0]
    assert entry["endpoint"] == "/ask"
    assert entry["route"] == "invalid_request"
    assert entry["status"] == 422
    assert entry["duration_ms"] >= 0
    assert "query" not in entry


def test_service_logs_one_sanitized_failure_without_raw_phone_or_exception(tmp_path, monkeypatch):
    import api.service as service_module

    def fail_with_sensitive_exception(*args, **kwargs):
        raise RuntimeError("private-password=do-not-log")

    monkeypatch.setattr(service_module, "run_support_crew", fail_with_sensitive_exception)
    log_path = tmp_path / "requests.jsonl"
    client = TestClient(create_app(log_path=log_path))
    result = client.post(
        "/ask",
        json={
            "query": "Please call me at 9876543210, token=private-value",
            "session_id": "caller@example.com",
        },
    )

    assert result.status_code == 500
    assert "private-password" not in result.text
    text = log_path.read_text(encoding="utf-8")
    assert "9876543210" not in text
    assert "caller@example.com" not in text
    assert "private-value" not in text
    assert "do-not-log" not in text
    entries = _records(log_path)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["masked_query"].endswith("token=[REDACTED]")
    assert entry["pii_masked"] is True
    assert entry["error_type"] == "RuntimeError"
    assert entry["safe_error_message"] == "The support request could not be completed."
    assert entry["trace_id"] == result.headers["x-trace-id"]
    assert entry["duration_ms"] >= 0


def test_api_converts_dependency_error_to_safe_response(tmp_path, monkeypatch):
    from api import app as app_module

    async def unavailable(*args, **kwargs):
        raise APIServiceError(
            status_code=503,
            code="dependency_unavailable",
            message="The support service is temporarily unavailable.",
            trace_id=kwargs["trace_id"],
        )

    monkeypatch.setattr(app_module, "ask_support_async", unavailable)
    client = TestClient(create_app(log_path=tmp_path / "requests.jsonl"))

    result = client.post("/ask", json={"query": "What is the P1 response SLA?"})

    assert result.status_code == 503
    assert result.json()["code"] == "dependency_unavailable"
    assert result.json()["trace_id"] == result.headers["x-trace-id"]
    assert "CrewAI" not in result.text
    assert "Chroma" not in result.text


def test_missing_session_id_is_generated_as_uuid_by_service(tmp_path, monkeypatch):
    import api.service as service_module

    captured = {}

    def capture_session_then_fail(_query, *, session_id):
        captured["session_id"] = session_id
        raise RuntimeError("stop after observing generated ID")

    monkeypatch.setattr(service_module, "run_support_crew", capture_session_then_fail)
    with pytest.raises(APIServiceError):
        service_ask_support(
            AskRequest(query="Check support policies."),
            trace_id="trace-generated-session",
            log_path=tmp_path / "requests.jsonl",
        )
    from uuid import UUID

    assert str(UUID(captured["session_id"])) == captured["session_id"]


def test_service_wraps_real_ticket_result_as_flat_api_response(tmp_path, monkeypatch):
    import api.service as service_module

    crew_response = _ticket_support_response()
    crew_result = CrewRunResult(
        response=crew_response,
        audit=CrewAudit(
            trace_id="crew-trace",
            route="ticket",
            status="completed",
            kickoff_called=True,
            tool_invocations=[
                ToolInvocation(
                    invocation_id="1",
                    tool_name="ticket_status_lookup",
                    arguments={"record_id": "TKT-0007"},
                    status="completed",
                    result={
                        "record_id": "TKT-0007",
                        "status": "In Progress",
                        "resolution_time_hours": 93.0,
                        "days_since_created": 13,
                        "escalation_recommended": True,
                    },
                )
            ],
        ),
    )
    monkeypatch.setattr(
        service_module,
        "run_support_crew",
        lambda _query, *, session_id: crew_result,
    )
    log_path = tmp_path / "requests.jsonl"

    response = service_ask_support(
        AskRequest(
            query="Check the status of TKT-0007.",
            session_id="api-real-session-memory-test",
        ),
        trace_id="api-service-success-trace",
        transport="websocket",
        endpoint="/ws/chat",
        log_path=log_path,
    )

    assert response.route == "ticket"
    assert response.record_id == "TKT-0007"
    assert response.ticket_status == "In Progress"
    assert response.resolution_time_hours == 93.0
    assert response.ticket_age_days == 13
    assert response.escalation_score == 0.8016666666666667
    assert response.escalation_recommended is True
    assert response.session_id == "api-real-session-memory-test"
    assert response.trace_id == "api-service-success-trace"
    record = _records(log_path)[0]
    assert record["route"] == "ticket"
    assert record["status"] == 200
    assert record["session_id"] == "api-real-session-memory-test"


def test_async_support_adapter_runs_sync_service_in_worker_thread(monkeypatch):
    import asyncio
    import threading

    import anyio
    import api.service as service_module

    caller_thread = threading.get_ident()
    observed = {}

    def synchronous_service(request, **kwargs):
        observed["thread_id"] = threading.get_ident()
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            observed["has_running_loop"] = False
        else:
            observed["has_running_loop"] = True
        return AskResponse(
            **_support_response().model_dump(),
            trace_id=kwargs["trace_id"],
            session_id=request.session_id or "thread-session",
        )

    monkeypatch.setattr(service_module, "ask_support", synchronous_service)

    async def invoke_adapter():
        return await service_module.ask_support_async(
            AskRequest(query="Check support policy.", session_id="thread-session"),
            trace_id="thread-boundary-trace",
        )

    response = anyio.run(invoke_adapter)

    assert observed["thread_id"] != caller_thread
    assert observed["has_running_loop"] is False
    assert response.trace_id == "thread-boundary-trace"
    assert response.session_id == "thread-session"


def test_ticket_route_maps_existing_lookup_and_returns_safe_404(tmp_path):
    client = TestClient(create_app(log_path=tmp_path / "requests.jsonl"))

    result = client.get("/ticket/TKT-0007")
    missing = client.get("/ticket/TKT-9999")

    assert result.status_code == 200
    assert result.json() == {
        "record_id": "TKT-0007",
        "status": "In Progress",
        "resolution_time_hours": 93.0,
        "ticket_age_days": 13,
        "escalation_score": 0.8016666666666667,
        "escalation_recommended": True,
    }
    assert result.headers["x-trace-id"]
    assert missing.status_code == 404
    assert missing.json()["code"] == "ticket_not_found"
    assert "TKT-9999" not in missing.text
    entries = _records(tmp_path / "requests.jsonl")
    assert len(entries) == 2
    assert [item["status"] for item in entries] == [200, 404]
    assert all(item["endpoint"] == "/ticket/{record_id}" for item in entries)
    assert all("session_id" in item and item["session_id"] is None for item in entries)


def test_ask_returns_a_fresh_trace_id_for_each_request(tmp_path, monkeypatch):
    from api import app as app_module

    async def response_for(request, **kwargs):
        return AskResponse(
            **_support_response().model_dump(),
            trace_id=kwargs["trace_id"],
            session_id=request.session_id or "session-test",
        )

    monkeypatch.setattr(app_module, "ask_support_async", response_for)
    client = TestClient(create_app(log_path=tmp_path / "requests.jsonl"))

    first = client.post("/ask", json={"query": "first"})
    second = client.post("/ask", json={"query": "second"})

    assert first.json()["trace_id"] != second.json()["trace_id"]
    assert first.headers["x-trace-id"] == first.json()["trace_id"]
    assert second.headers["x-trace-id"] == second.json()["trace_id"]


def test_websocket_validates_messages_preserves_connection_session_and_traces(
    tmp_path,
    monkeypatch,
):
    import api.service as service_module

    seen_sessions = []

    def service_boundary(request, *, trace_id, **kwargs):
        seen_sessions.append(request.session_id)
        return AskResponse(
            **_ticket_support_response().model_dump(),
            trace_id=trace_id,
            session_id=request.session_id or "session-test",
        )

    monkeypatch.setattr(service_module, "ask_support", service_boundary)
    client = TestClient(create_app(log_path=tmp_path / "requests.jsonl"))

    with client.websocket_connect("/ws/chat") as websocket:
        websocket.send_text("{not-json")
        malformed = websocket.receive_json()
        websocket.send_json({"query": "first question", "session_id": "same-session"})
        first = websocket.receive_json()
        websocket.send_json({"query": "follow-up"})
        second = websocket.receive_json()

    assert malformed["code"] == "invalid_request"
    assert first["session_id"] == "same-session"
    assert second["session_id"] == "same-session"
    assert first["route"] == second["route"] == "ticket"
    assert first["record_id"] == second["record_id"] == "TKT-0007"
    assert first["ticket_status"] == "In Progress"
    from uuid import UUID

    UUID(first["trace_id"])
    UUID(second["trace_id"])
    assert first["trace_id"] != second["trace_id"]
    assert seen_sessions == ["same-session", "same-session"]
    assert "response" not in first
    assert "response" not in second
    assert _records(tmp_path / "requests.jsonl")[0]["route"] == "invalid_request"


def test_websocket_rejects_invalid_payload_safely_and_keeps_connection_open(
    tmp_path,
):
    client = TestClient(create_app(log_path=tmp_path / "requests.jsonl"))

    with client.websocket_connect("/ws/chat") as websocket:
        websocket.send_json({"query": " ", "password": "secret"})
        result = websocket.receive_json()

    assert result["code"] == "invalid_request"
    assert "secret" not in json.dumps(result)
    assert len(_records(tmp_path / "requests.jsonl")) == 1


def test_websocket_retains_supplied_session_after_first_service_error(
    tmp_path,
    monkeypatch,
):
    import api.service as service_module

    seen_sessions = []

    def fail_then_succeed(request, *, trace_id, **kwargs):
        seen_sessions.append(request.session_id)
        if len(seen_sessions) == 1:
            raise APIServiceError(
                status_code=500,
                code="service_error",
                message="The support request could not be completed.",
                trace_id=trace_id,
            )
        return AskResponse(
            **_ticket_support_response().model_dump(),
            trace_id=trace_id,
            session_id=request.session_id,
        )

    monkeypatch.setattr(service_module, "ask_support", fail_then_succeed)
    client = TestClient(create_app(log_path=tmp_path / "requests.jsonl"))
    with client.websocket_connect("/ws/chat") as websocket:
        websocket.send_json(
            {"query": "Check the status of TKT-0007.", "session_id": "kept-session"}
        )
        first = websocket.receive_json()
        websocket.send_json({"query": "Should that ticket be escalated?"})
        second = websocket.receive_json()

    assert first["code"] == "service_error", first
    assert second["record_id"] == "TKT-0007", second
    assert second["session_id"] == "kept-session", second
    assert seen_sessions == ["kept-session", "kept-session"]


def test_websocket_service_error_logs_one_masked_record_with_matching_trace(
    tmp_path,
    monkeypatch,
):
    import api.service as service_module

    def fail_safely(*args, **kwargs):
        raise RuntimeError("private-password=do-not-log")

    monkeypatch.setattr(service_module, "run_support_crew", fail_safely)
    log_path = tmp_path / "requests.jsonl"
    client = TestClient(create_app(log_path=log_path))
    with client.websocket_connect("/ws/chat") as websocket:
        websocket.send_json(
            {
                "query": "My number is 9876543210. What is my ticket status?",
                "session_id": "chat-session",
            }
        )
        result = websocket.receive_json()
    assert result["code"] == "service_error"
    raw_log = log_path.read_text(encoding="utf-8")
    assert "9876543210" not in raw_log
    assert "do-not-log" not in raw_log
    entries = _records(log_path)
    assert len(entries) == 1
    assert entries[0]["transport"] == "websocket"
    assert entries[0]["trace_id"] == result["trace_id"]
    assert entries[0]["pii_masked"] is True
    assert "[PHONE REDACTED]" in entries[0]["masked_query"]


def test_log_record_jsonl_has_required_fields_and_nonnegative_duration(tmp_path):
    from api.logging import append_request_log

    path = tmp_path / "requests.jsonl"
    append_request_log(
        trace_id="trace-test",
        transport="http",
        endpoint="/ask",
        session_id="session-test",
        route="policy",
        status=200,
        duration_ms=-2,
        query="What is P1?",
        log_path=path,
    )
    record = _records(path)[0]
    assert {
        "timestamp",
        "trace_id",
        "transport",
        "endpoint",
        "session_id",
        "route",
        "status",
        "duration_ms",
        "pii_masked",
        "masked_query",
    } <= set(record)
    assert record["duration_ms"] == 0
    assert re.match(r"\d{4}-\d{2}-\d{2}T", record["timestamp"])


def test_real_service_layer_ticket_response_when_runtime_exists(
    tmp_path,
    monkeypatch,
):
    import importlib.util

    if (
        importlib.util.find_spec("crewai") is None
        or importlib.util.find_spec("chromadb") is None
    ):
        pytest.skip("Real CrewAI+ChromaDB runtime unavailable in this environment.")

    from agents.crew import SESSION_MEMORY

    session_id = "service-real-ticket-test"
    SESSION_MEMORY.clear_session(session_id)
    capture = _capture_real_crew_calls(monkeypatch)
    log_path = tmp_path / "requests.jsonl"
    try:
        try:
            response = service_ask_support(
                AskRequest(
                    query="Check the status of TKT-0007.",
                    session_id=session_id,
                ),
                trace_id="service-real-ticket-test-trace",
                transport="http",
                endpoint="/ask",
                log_path=log_path,
            )
        except APIServiceError as error:
            raise AssertionError(
                {
                    "api_error": error.as_response().model_dump(),
                    **_real_crew_diagnostics(capture),
                    "request_logs": _records(log_path),
                }
            ) from None
    finally:
        SESSION_MEMORY.clear_session(session_id)

    assert response.route == "ticket"
    assert response.record_id == "TKT-0007"
    assert response.ticket_status == "In Progress"
    assert response.resolution_time_hours == 93.0
    assert response.escalation_recommended is True


def test_real_http_ask_uses_worker_adapter_when_runtime_exists(tmp_path, monkeypatch):
    import importlib.util

    if (
        importlib.util.find_spec("crewai") is None
        or importlib.util.find_spec("chromadb") is None
    ):
        pytest.skip("Real CrewAI+ChromaDB runtime unavailable in this environment.")

    from agents.crew import SESSION_MEMORY

    session_id = "api-real-http-ticket-test"
    SESSION_MEMORY.clear_session(session_id)
    capture = _capture_real_crew_calls(monkeypatch)
    log_path = tmp_path / "requests.jsonl"
    try:
        result = TestClient(create_app(log_path=log_path)).post(
            "/ask",
            json={
                "query": "Check the status of TKT-0007.",
                "session_id": session_id,
            },
        )
    finally:
        SESSION_MEMORY.clear_session(session_id)

    if result.status_code != 200:
        raise AssertionError(
            {
                "status_code": result.status_code,
                "response": result.json(),
                **_real_crew_diagnostics(capture),
                "request_logs": _records(log_path),
            }
        )
    payload = result.json()
    assert payload["route"] == "ticket", payload
    assert payload["record_id"] == "TKT-0007", payload
    assert payload["ticket_status"] == "In Progress", payload
    assert payload["resolution_time_hours"] == 93.0, payload
    assert payload["escalation_recommended"] is True, payload
    assert payload["session_id"] == session_id, payload
    assert payload["trace_id"] == result.headers["x-trace-id"], payload


def test_two_message_real_session_memory_integration_when_runtime_exists(
    tmp_path,
    monkeypatch,
):
    import importlib.util

    if (
        importlib.util.find_spec("crewai") is None
        or importlib.util.find_spec("chromadb") is None
    ):
        pytest.skip("Real CrewAI+ChromaDB runtime unavailable in this environment.")

    from agents.crew import SESSION_MEMORY

    session_id = "api-real-session-memory-test"
    SESSION_MEMORY.clear_session(session_id)
    capture = _capture_real_crew_calls(monkeypatch)
    client = TestClient(create_app(log_path=tmp_path / "requests.jsonl"))
    try:
        with client.websocket_connect("/ws/chat") as websocket:
            websocket.send_json(
                {"query": "Check the status of TKT-0007.", "session_id": session_id}
            )
            first = websocket.receive_json()
            websocket.send_json({"query": "Should that ticket be escalated?"})
            second = websocket.receive_json()
    finally:
        SESSION_MEMORY.clear_session(session_id)
    if first.get("route") != "ticket":
        raise AssertionError(
            {
                "first_response": first,
                **_real_crew_diagnostics(capture),
                "request_logs": _records(tmp_path / "requests.jsonl"),
            }
        )
    assert first["record_id"] == "TKT-0007", first
    assert first["ticket_status"] == "In Progress", first
    assert first["trace_id"], first
    assert first["session_id"] == session_id, first
    assert "response" not in first, first

    if second.get("route") != "ticket":
        raise AssertionError(
            {
                "second_response": second,
                **_real_crew_diagnostics(capture),
                "request_logs": _records(tmp_path / "requests.jsonl"),
            }
        )
    assert second["record_id"] == "TKT-0007", second
    assert second["escalation_recommended"] is True, second
    assert second["session_id"] == session_id, second
    assert second["trace_id"] != first["trace_id"], second
    assert "response" not in second, second
