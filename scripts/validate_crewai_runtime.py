"""Run policy, ticket, and session-isolation checks through real CrewAI kickoff."""

from __future__ import annotations

import json
import traceback
from typing import Any

from pydantic import ValidationError

from agents.crew import SESSION_MEMORY, run_support_crew
from agents.schemas import SupportResponse


CASES = (
    (
        "policy",
        "What is the response SLA for a P1 critical ticket?",
        "crewai-runtime-session-A",
        "policy",
        True,
    ),
    (
        "ticket",
        "Check status of TKT-0007.",
        "crewai-runtime-session-A",
        "ticket",
        True,
    ),
    (
        "session A follow-up",
        "Should that ticket be escalated?",
        "crewai-runtime-session-A",
        "ticket",
        True,
    ),
    (
        "session B isolation",
        "Should that ticket be escalated?",
        "crewai-runtime-session-B",
        "fallback",
        False,
    ),
)


def _response_data(response: Any) -> dict[str, Any] | None:
    if response is None:
        return None
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    if isinstance(response, dict):
        return response
    return {"unsupported_response_type": type(response).__name__}


def _pydantic_status(response: Any) -> tuple[str, bool]:
    if response is None:
        return "not run: no response", False
    try:
        SupportResponse.model_validate(_response_data(response))
    except ValidationError as exc:
        return f"invalid: {type(exc).__name__}: {exc}", False
    return "valid SupportResponse", True


def _validate_case(
    name: str,
    expected_route: str,
    expected_kickoff: bool,
    outcome: Any,
    pydantic_valid: bool,
) -> list[str]:
    errors: list[str] = []
    audit = outcome.audit
    if audit.route != expected_route:
        errors.append(f"expected route {expected_route}, got {audit.route}")
    if audit.kickoff_called is not expected_kickoff:
        errors.append(
            f"expected kickoff_called={expected_kickoff}, got {audit.kickoff_called}"
        )
    if outcome.response is None:
        errors.append(f"response is None: {audit.error or audit.status}")
    if not pydantic_valid:
        errors.append("final response did not validate as SupportResponse")

    expected_tools = {
        "policy": ["rag_lookup"],
        "ticket": ["ticket_status_lookup"],
        "session A follow-up": ["ticket_status_lookup"],
        "session B isolation": [],
    }[name]
    actual_tools = [item.tool_name for item in audit.tool_invocations]
    if actual_tools != expected_tools:
        errors.append(f"expected tool calls {expected_tools}, got {actual_tools}")

    if name in {"policy", "ticket", "session A follow-up"}:
        if audit.status != "completed":
            errors.append(f"expected completed status, got {audit.status}")
    else:
        if audit.status != "missing_ticket_context":
            errors.append(f"expected missing_ticket_context, got {audit.status}")

    if name == "policy" and outcome.response is not None:
        if outcome.response.fallback_triggered:
            errors.append("policy response unexpectedly used fallback")
        if outcome.response.sources:
            answer = outcome.response.answer.lower()
            for required_phrase in ("15 minutes", "4 hours"):
                if required_phrase not in answer:
                    errors.append(
                        f"grounded policy answer omitted {required_phrase!r}"
                    )
    if name in {"ticket", "session A follow-up"} and outcome.response is not None:
        if outcome.response.resolution_time_hours is None:
            errors.append("ticket response omitted resolution_time_hours")
    if name == "session B isolation" and outcome.response is not None:
        if outcome.response.record_id is not None:
            errors.append("session B inherited session A's ticket ID")
    return errors


def main() -> int:
    session_ids = {case[2] for case in CASES}
    for session_id in session_ids:
        SESSION_MEMORY.clear_session(session_id)

    failures: list[str] = []
    for name, query, session_id, expected_route, expected_kickoff in CASES:
        print(f"\n=== {name} ===")
        print(f"query: {query}")
        print(f"session_id: {session_id}")
        try:
            outcome = run_support_crew(query, session_id=session_id)
        except Exception as exc:
            traceback.print_exc()
            failures.append(f"{name}: uncaught {type(exc).__name__}: {exc}")
            continue

        validation_status, pydantic_valid = _pydantic_status(outcome.response)
        print(f"route: {outcome.audit.route}")
        print(f"kickoff_called: {outcome.audit.kickoff_called}")
        print(f"response: {json.dumps(_response_data(outcome.response), ensure_ascii=False)}")
        print(f"audit.status: {outcome.audit.status}")
        print(f"audit.error: {outcome.audit.error}")
        print(f"agents: {outcome.audit.participating_agents}")
        print(f"tool invocation count: {outcome.audit.tool_invocation_count}")
        print(
            "tool names: "
            + repr([item.tool_name for item in outcome.audit.tool_invocations])
        )
        print(f"Pydantic validation: {validation_status}")

        case_errors = _validate_case(
            name,
            expected_route,
            expected_kickoff,
            outcome,
            pydantic_valid,
        )
        failures.extend(f"{name}: {message}" for message in case_errors)

    if failures:
        print("\nCrewAI runtime validation FAILED:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("\nCrewAI runtime validation PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
