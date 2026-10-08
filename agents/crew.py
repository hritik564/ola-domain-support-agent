"""Three-role CrewAI orchestration with preflight guardrails and audits."""

from __future__ import annotations

import json
import re
import os
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

# Keep these assignments before CrewAI imports; do not permit telemetry override.
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from agents.audit import audit_context, new_audit
from agents.mock_llm import (
    MOCK_LLM_USES_CREWAI_BASE,
    MockLLM,
    MockLLMRuntimeError,
)
from agents.schemas import CrewAudit, CrewRunResult, SupportResponse
from guardrails.groundedness import (
    validate_policy_draft,
    validate_ticket_draft,
)
from guardrails.injection import check_prompt_injection
from guardrails.pii import mask_phone_numbers
from memory.session_memory import TicketSessionMemory
from tools.rag_tool import RAGLookupTool
from tools.ticket_tool import TicketStatusTool

try:
    from crewai import Agent, Crew, Process, Task

    CREWAI_AVAILABLE = True
    CREWAI_IMPORT_ERROR: str | None = None
except ImportError as exc:
    Agent = Crew = Process = Task = None  # type: ignore[assignment,misc]
    CREWAI_AVAILABLE = False
    CREWAI_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

SESSION_MEMORY = TicketSessionMemory()
_DIRECT_TICKET_ID = re.compile(r"\bTKT-\d+\b", re.IGNORECASE)
_IMPLICIT_TICKET_REFERENCE = re.compile(
    r"\b(?:that|this|same)\s+(?:support\s+)?ticket\b|"
    r"\bshould\s+it\s+be\s+escalated\b|"
    r"\b(?:check|update|escalate|close)\s+it\b",
    re.IGNORECASE,
)
_SUPPORT_SCOPE_CUE = re.compile(
    r"\b(?:support|ticket|billing|payment(?:s)?|charge(?:s|d)?|refund|"
    r"account|access|login|log\s+in|sign\s+in|password|priority|p[1-4]|sla|"
    r"response|resolution|escalat\w*|incident|outage|downtime|"
    r"service\s+(?:credit|unavailable|interruption)|support\s+channel|"
    r"business\s+hours|complaint|feedback|vip|privacy|retention|unresolved|"
    r"reopen|technical\s+issue|product\s+defect)\b",
    re.IGNORECASE,
)
_SAFE_BLOCKED_ANSWER = (
    "I can’t follow instructions that override the support assistant’s safety rules."
)
_MISSING_TICKET_ANSWER = (
    "I couldn’t find a ticket ID in this session. Please provide the ticket ID to check."
)
_OUT_OF_SCOPE_ANSWER = "I don't know based on the available support policies."


def _selected_route(query: str, session_id: str) -> tuple[str, str | None]:
    direct_match = _DIRECT_TICKET_ID.search(query)
    if direct_match:
        record_id = SESSION_MEMORY.resolve_record_id(session_id, query)
        return "ticket", record_id

    if _IMPLICIT_TICKET_REFERENCE.search(query):
        record_id = SESSION_MEMORY.resolve_record_id(session_id, query)
        if record_id:
            return "ticket", record_id
        return "fallback", None

    if not _SUPPORT_SCOPE_CUE.search(query):
        return "fallback", None

    return "policy", None


def _agent_bundle(
    *,
    route: Literal["policy", "ticket"],
    query: str,
    record_id: str | None,
    pii_masked: bool,
    audit: CrewAudit,
) -> tuple[Any, dict[str, Any]]:
    if not CREWAI_AVAILABLE or not MOCK_LLM_USES_CREWAI_BASE:
        raise RuntimeError(
            "CrewAI or its documented BaseLLM interface is unavailable here. "
            f"Import result: {CREWAI_IMPORT_ERROR or 'not installed'}"
        )

    llm = MockLLM(
        route=route,
        query=query,
        record_id=record_id,
        audit=audit,
        pii_masked=pii_masked,
    )
    retrieval_agent = Agent(
        role="Retrieval Agent",
        goal="Find only relevant, grounded support-policy evidence.",
        backstory="You use the existing policy RAG pipeline and never invent policy.",
        tools=[RAGLookupTool()],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )
    lookup_agent = Agent(
        role="Lookup Agent",
        goal="Return the dataset-backed status for one support ticket.",
        backstory="You use the existing deterministic ticket lookup only.",
        tools=[TicketStatusTool()],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )
    composer_agent = Agent(
        role="Response Composer",
        goal="Compose a concise response using only the upstream result.",
        backstory=(
            "You cannot call tools. Preserve grounded policy text and ticket "
            "fields, including resolution_time_hours, exactly as returned by "
            "the assigned specialist."
        ),
        tools=[],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )
    agents = {
        "retrieval": retrieval_agent,
        "lookup": lookup_agent,
        "composer": composer_agent,
    }

    if route == "policy":
        specialist_task = Task(
            description=(
                "Customer policy question: {query}\n"
                "Call the RAG tool with the customer's question. Return the "
                "tool result without adding policy claims."
            ),
            expected_output="The JSON result returned by the grounded RAG tool.",
            agent=retrieval_agent,
        )
    else:
        specialist_task = Task(
            description=(
                "Look up support ticket {record_id} with the assigned ticket "
                "status tool. Return its result without changing any fields."
            ),
            expected_output="The JSON result returned by the ticket lookup tool.",
            agent=lookup_agent,
        )

    composer_task = Task(
        description=(
            "Create a JSON SupportResponse from the preceding specialist result. "
            "For policy, preserve its answer, grounded flag, fallback flag, and "
                "sources exactly. For a ticket, preserve record_id, status, "
                "resolution_time_hours, age, and escalation fields from the result. "
            "Include route={route} and pii_masked={pii_masked}. Do not add facts."
        ),
        expected_output="One JSON object that validates as SupportResponse.",
        agent=composer_agent,
        context=[specialist_task],
        output_pydantic=SupportResponse,
    )
    crew = Crew(
        agents=[retrieval_agent, lookup_agent, composer_agent],
        tasks=[specialist_task, composer_task],
        process=Process.sequential,
        verbose=False,
        memory=False,
        cache=False,
    )
    return crew, agents


def _payload_from_candidate(candidate: Any) -> dict[str, Any]:
    if isinstance(candidate, BaseModel):
        return candidate.model_dump()
    if isinstance(candidate, Mapping):
        return dict(candidate)
    if isinstance(candidate, str):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "CrewAI output was not valid JSON SupportResponse"
            ) from exc
        if isinstance(parsed, Mapping):
            return dict(parsed)
        raise TypeError("CrewAI JSON output must be an object")
    raise TypeError(
        "CrewAI output must be a SupportResponse, mapping, or JSON object string; "
        f"received {type(candidate).__name__}"
    )


def _kickoff_payload(result: Any) -> dict[str, Any]:
    """Normalize CrewOutput, raw JSON, mappings, and Pydantic output."""
    if isinstance(result, Mapping):
        for key in ("pydantic", "json_dict", "raw", "output"):
            candidate = result.get(key)
            if candidate is not None:
                if isinstance(candidate, BaseException):
                    raise TypeError(
                        f"CrewAI returned an exception in its {key} output"
                    )
                return _payload_from_candidate(candidate)
        return dict(result)

    for key in ("pydantic", "json_dict", "raw", "output"):
        candidate = getattr(result, key, None)
        if candidate is not None:
            if isinstance(candidate, BaseException):
                raise TypeError(f"CrewAI returned an exception in its {key} output")
            return _payload_from_candidate(candidate)
    return _payload_from_candidate(result)


def _kickoff_response(result: Any, pii_masked: bool) -> SupportResponse:
    payload = _kickoff_payload(result)
    payload["pii_masked"] = pii_masked
    return SupportResponse.model_validate(payload)


def _tool_evidence(audit: CrewAudit, tool_name: str) -> dict[str, Any] | None:
    for invocation in reversed(audit.tool_invocations):
        if (
            invocation.tool_name == tool_name
            and invocation.status == "completed"
            and isinstance(invocation.result, dict)
        ):
            return invocation.result
    return None


def _exception_detail(exc: BaseException) -> str:
    chain: list[str] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    return " -> caused by ".join(chain)


def _diagnostic_stage(exc: Exception, audit: CrewAudit, current_stage: str) -> str:
    if any(item.status == "failed" for item in audit.tool_invocations):
        return "tool_call"
    if isinstance(exc, MockLLMRuntimeError):
        return exc.stage
    if isinstance(exc, ValidationError):
        return "pydantic_validation"
    if audit.error:
        match = re.search(r"\bstage=([a-z_]+);", audit.error)
        if match:
            return match.group(1)
    return current_stage


def run_support_crew(query: str, session_id: str) -> CrewRunResult:
    """Sanitize, route, call CrewAI kickoff, and validate the final response."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be a non-empty string")

    audit = new_audit()
    pii = mask_phone_numbers(query)
    audit.pii_masked = pii.masked
    audit.sanitized_input = pii.text

    injection = check_prompt_injection(pii.text)
    if injection.blocked:
        audit.route = "blocked"
        audit.status = "blocked_prompt_injection"
        audit.guardrail_reason = injection.matched_rule
        response = SupportResponse(
            answer=_SAFE_BLOCKED_ANSWER,
            route="blocked",
            grounded=False,
            fallback_triggered=True,
            pii_masked=pii.masked,
        )
        return CrewRunResult(response=response, audit=audit)

    route, record_id = _selected_route(pii.text, session_id)
    audit.route = route
    if route == "fallback":
        missing_ticket_context = bool(_IMPLICIT_TICKET_REFERENCE.search(pii.text))
        audit.status = (
            "missing_ticket_context" if missing_ticket_context else "out_of_scope"
        )
        response = SupportResponse(
            answer=(
                _MISSING_TICKET_ANSWER
                if missing_ticket_context
                else _OUT_OF_SCOPE_ANSWER
            ),
            route="fallback",
            grounded=False,
            fallback_triggered=True,
            pii_masked=pii.masked,
        )
        return CrewRunResult(response=response, audit=audit)

    participating_agents = (
        ["Retrieval Agent", "Response Composer"]
        if route == "policy"
        else ["Lookup Agent", "Response Composer"]
    )
    audit.participating_agents = participating_agents
    if not CREWAI_AVAILABLE or not MOCK_LLM_USES_CREWAI_BASE:
        audit.status = "blocked_dependency"
        audit.error = (
            "stage=crew_construction; DependencyUnavailable: CrewAI or its "
            "documented BaseLLM interface is unavailable; install dependencies "
            "in an environment where "
            f"the pinned ChromaDB wheel can be fetched. "
            f"Import result: {CREWAI_IMPORT_ERROR or 'not installed'}"
        )
        return CrewRunResult(response=None, audit=audit)

    stage = "crew_construction"
    try:
        with audit_context(audit):
            crew, _agents = _agent_bundle(
                route=route,
                query=pii.text,
                record_id=record_id,
                pii_masked=pii.masked,
                audit=audit,
            )
            audit.kickoff_called = True
            stage = "kickoff"
            crew_result = crew.kickoff(
                inputs={
                    "query": pii.text,
                    "record_id": record_id or "",
                    "route": route,
                    "pii_masked": pii.masked,
                }
            )
            stage = "final_response_parse"
            payload = _kickoff_payload(crew_result)
            payload["pii_masked"] = pii.masked
            stage = "pydantic_validation"
            response = SupportResponse.model_validate(payload)

        stage = "tool_call"
        if route == "policy":
            evidence = _tool_evidence(audit, "rag_lookup")
            if evidence is None:
                raise RuntimeError("CrewAI completed without a recorded RAG tool result")
            check = validate_policy_draft(response.answer, evidence)
        else:
            evidence = _tool_evidence(audit, "ticket_status_lookup")
            if evidence is None:
                raise RuntimeError(
                    "CrewAI completed without a recorded ticket lookup result"
                )
            check = validate_ticket_draft(response.model_dump(), evidence)
        stage = "groundedness_validation"
        if not check.accepted:
            audit.status = "groundedness_rejected"
            audit.guardrail_reason = check.reason
            audit.error = (
                "stage=groundedness_validation; GroundednessRejected: "
                f"{check.reason}"
            )
            return CrewRunResult(response=None, audit=audit)

        audit.status = "completed"
        return CrewRunResult(response=response, audit=audit)
    except Exception as exc:
        audit.status = "kickoff_or_validation_failed"
        stage = _diagnostic_stage(exc, audit, stage)
        detail = _exception_detail(exc)
        outer_detail = f"stage={stage}; {detail}"
        if audit.error and audit.error not in outer_detail:
            audit.error = f"{audit.error}; outer exception: {outer_detail}"
        else:
            audit.error = outer_detail
        return CrewRunResult(response=None, audit=audit)


def run_guarded_without_model(query: str) -> CrewRunResult:
    """Expose the same preflight path for guardrail tests and demos."""
    return run_support_crew(query, session_id="guardrail-demo")
