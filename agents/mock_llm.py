"""Deterministic, zero-key MOCK_LLM for CrewAI's documented BaseLLM API."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, Literal, NoReturn

from pydantic import BaseModel, Field, ValidationError

from agents.schemas import CrewAudit, SupportResponse


class MockLLMRuntimeError(RuntimeError):
    """A deterministic MOCK_LLM failure with an explicit execution stage."""

    def __init__(self, stage: str, message: str) -> None:
        self.stage = stage
        super().__init__(message)

try:
    from crewai.llms.base_llm import BaseLLM

    MOCK_LLM_USES_CREWAI_BASE = True
except ImportError:
    try:
        from crewai import BaseLLM

        MOCK_LLM_USES_CREWAI_BASE = True
    except ImportError:
        MOCK_LLM_USES_CREWAI_BASE = False

        class BaseLLM:
            """Import-time fallback only; it cannot execute a CrewAI crew."""

            def __init__(
                self,
                model: str,
                temperature: float = 0.0,
                **kwargs: Any,
            ) -> None:
                self.model = model
                self.temperature = temperature


_TICKET_ID = re.compile(r"\bTKT-\d+\b", re.IGNORECASE)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _message_items(messages: Any) -> list[Any]:
    if isinstance(messages, (list, tuple)):
        return list(messages)
    if isinstance(messages, Mapping) or any(
        hasattr(messages, name)
        for name in ("role", "type", "content", "tool_call_id")
    ):
        return [messages]
    return []


def _message_role(message: Any) -> str:
    role = _field(message, "role")
    if role is None:
        role = _field(message, "type", "")
    if not isinstance(role, str):
        role = getattr(role, "value", role)
    role = str(role).lower()
    return {
        "human": "user",
        "ai": "assistant",
        "tool_result": "tool",
        "function_result": "function",
        "function_call_output": "tool",
    }.get(role, role)


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="replace")
    if isinstance(content, BaseModel):
        content = content.model_dump(mode="json")
    if isinstance(content, Mapping):
        if content.get("type") == "text" and "text" in content:
            return _content_text(content["text"])
        return json.dumps(dict(content), ensure_ascii=False, default=str)
    if isinstance(content, (list, tuple)):
        parts = []
        for item in content:
            if isinstance(item, Mapping) and item.get("type") == "text":
                parts.append(_content_text(item.get("text", "")))
            else:
                parts.append(_content_text(item))
        return "\n".join(part for part in parts if part)
    if hasattr(content, "model_dump"):
        return json.dumps(content.model_dump(), ensure_ascii=False, default=str)
    return str(content)


class ToolResultNormalizationError(ValueError):
    """A tool result could not be converted to a JSON object."""


def _tool_result_diagnostic(payload: Any, detail: str) -> str:
    preview = repr(payload)[:300]
    return (
        f"{detail}; tool_result_type={type(payload).__name__}; "
        f"tool_result_preview={preview}"
    )


def _strip_json_fence(text: str) -> str:
    lines = text.splitlines()
    if (
        len(lines) >= 3
        and lines[0].strip().lower() in {"```", "```json"}
        and lines[-1].strip() == "```"
    ):
        return "\n".join(lines[1:-1]).strip()
    return text


def _extract_framed_json_object(text: str) -> dict[str, Any] | None:
    """Find one JSON object inside framing from an already-identified tool result."""
    for start, character in enumerate(text):
        if character != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for end in range(start, len(text)):
            character = text[end]
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                continue
            if character == '"':
                in_string = True
            elif character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    try:
                        candidate = json.loads(text[start : end + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(candidate, Mapping):
                        return dict(candidate)
                    break
    return None


def _parse_tool_result_payload(payload: Any) -> dict[str, Any]:
    """Normalize a real tool payload without parsing conversation or prompt text."""
    original_payload = payload
    if isinstance(payload, BaseModel) or callable(
        getattr(payload, "model_dump", None)
    ):
        try:
            payload = payload.model_dump(mode="json")
        except Exception as exc:
            raise ToolResultNormalizationError(
                _tool_result_diagnostic(
                    original_payload,
                    f"Pydantic tool result could not be serialized: {exc}",
                )
            ) from exc

    if isinstance(payload, Mapping):
        if payload.get("type") == "text" and "text" in payload:
            return _parse_tool_result_payload(payload["text"])
        return dict(payload)

    if isinstance(payload, (list, tuple)):
        if len(payload) == 1 and isinstance(payload[0], Mapping):
            return _parse_tool_result_payload(payload[0])
        if all(
            isinstance(part, Mapping)
            and part.get("type") == "text"
            and isinstance(part.get("text"), (str, bytes))
            for part in payload
        ):
            text_parts = [
                part["text"].decode("utf-8")
                if isinstance(part["text"], bytes)
                else part["text"]
                for part in payload
            ]
            return _parse_tool_result_payload("\n".join(text_parts))
        raise ToolResultNormalizationError(
            _tool_result_diagnostic(
                original_payload,
                "Tool result content blocks must contain text or one object.",
            )
        )

    if isinstance(payload, bytes):
        try:
            payload = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolResultNormalizationError(
                _tool_result_diagnostic(
                    original_payload,
                    f"Tool result bytes are not valid UTF-8: {exc}",
                )
            ) from exc

    if isinstance(payload, str):
        text = _strip_json_fence(payload.strip())
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            framed = _extract_framed_json_object(text)
            if framed is not None:
                return framed
            raise ToolResultNormalizationError(
                _tool_result_diagnostic(
                    original_payload,
                    f"Tool result was not valid JSON: {exc}",
                )
            ) from exc
        if isinstance(parsed, Mapping):
            return dict(parsed)
        raise ToolResultNormalizationError(
            _tool_result_diagnostic(
                original_payload,
                "Tool result JSON must be an object.",
            )
        )

    raise ToolResultNormalizationError(
        _tool_result_diagnostic(
            original_payload,
            "Tool result must be a mapping, Pydantic model, UTF-8 bytes, or JSON text.",
        )
    )


def _function_schema(tool: Any) -> Mapping[str, Any]:
    function = _field(tool, "function", tool)
    return function if isinstance(function, Mapping) else {}


def _tool_result_messages(messages: Any) -> list[Any]:
    """Extract actual tool/function results, not prompt examples or tool calls."""
    results: list[Any] = []
    result_types = {
        "function_call_output",
        "function_result",
        "tool_result",
        "tool_use_result",
    }
    for message in _message_items(messages):
        role = _message_role(message)
        message_type = _field(message, "type", "")
        if not isinstance(message_type, str):
            message_type = getattr(message_type, "value", message_type)
        message_type = str(message_type).lower()
        tool_call_id = (
            _field(message, "tool_call_id")
            or _field(message, "toolUseId")
            or _field(message, "tool_use_id")
        )
        function_payload = _field(message, "function")
        function_result_payload = isinstance(function_payload, Mapping) and any(
            key in function_payload for key in ("result", "output", "response")
        )
        mapping_result_payload = isinstance(message, Mapping) and any(
            key in message for key in ("tool_result", "function_result", "function_response")
        )
        is_result = (
            role in {"tool", "function"}
            or message_type in result_types
            or function_result_payload
            or mapping_result_payload
            or (
                bool(tool_call_id)
                and role not in {"assistant", "user", "system"}
            )
        )
        if not is_result:
            continue
        content = _field(message, "content")
        if content is None:
            for key in (
                "output",
                "result",
                "payload",
                "tool_result",
                "function_response",
                "function_result",
            ):
                content = _field(message, key)
                if content is not None:
                    break
        if content == "":
            for key in (
                "output",
                "result",
                "payload",
                "tool_result",
                "function_response",
                "function_result",
            ):
                candidate = _field(message, key)
                if candidate is not None and candidate != "":
                    content = candidate
                    break
        if (content is None or content == "") and isinstance(
            function_payload, Mapping
        ):
            for key in ("result", "output", "response"):
                content = function_payload.get(key)
                if content is not None and content != "":
                    break
        if content is not None and content != "":
            results.append(content)
    return results


def _message_text(messages: Any) -> str:
    if isinstance(messages, str):
        return messages
    return "\n".join(
        _content_text(_field(message, "content"))
        for message in _message_items(messages)
        if _message_role(message) in {"user", "assistant"}
    )


class MockLLM(BaseLLM):
    """No-network CrewAI LLM that calls supplied functions by schema."""

    route: Literal["policy", "ticket"] = "policy"
    query: str = ""
    record_id: str | None = None
    pii_masked: bool = False
    audit: CrewAudit | None = Field(default=None, exclude=True)

    def __init__(
        self,
        *,
        route: Literal["policy", "ticket"],
        query: str,
        audit: CrewAudit | None = None,
        record_id: str | None = None,
        pii_masked: bool = False,
    ) -> None:
        super().__init__(model="ola-mock-llm", temperature=0.0)
        self.route = route
        self.query = query
        self.record_id = record_id
        self.pii_masked = pii_masked
        self.audit = audit

    def supports_function_calling(self) -> bool:
        return True

    def supports_stop_words(self) -> bool:
        return True

    def get_context_window_size(self) -> int:
        return 8192

    def _select_tool(
        self,
        tools: list[Any],
        message_text: str,
    ) -> tuple[str, dict[str, str]] | None:
        expected_argument = "record_id" if self.route == "ticket" else "query"
        ticket_id = self.record_id
        if ticket_id is None:
            found_ids = _TICKET_ID.findall(f"{self.query}\n{message_text}")
            if found_ids:
                ticket_id = found_ids[-1].upper()

        for tool in tools:
            schema = _function_schema(tool)
            parameters = schema.get("parameters", {})
            if not isinstance(parameters, Mapping):
                continue
            properties = parameters.get("properties", {})
            if not isinstance(properties, Mapping):
                continue
            required_values = parameters.get("required", [])
            required = (
                set(required_values)
                if isinstance(required_values, (list, tuple, set))
                else set()
            )
            if expected_argument not in properties:
                continue
            if required and expected_argument not in required:
                continue
            if expected_argument == "record_id":
                if ticket_id is None:
                    continue
                name = schema.get("name")
                if isinstance(name, str) and name:
                    return name, {"record_id": ticket_id}
                continue
            name = schema.get("name")
            if isinstance(name, str) and name:
                return name, {"query": self.query}
        return None

    def _raise_runtime_error(
        self,
        stage: str,
        message: str,
        *,
        cause: Exception | None = None,
    ) -> NoReturn:
        error = MockLLMRuntimeError(stage, message)
        if self.audit is not None:
            self.audit.error = f"stage={stage}; {type(error).__name__}: {message}"
        if cause is not None:
            raise error from cause
        raise error

    def _response_from_evidence(self, evidence: Mapping[str, Any]) -> str:
        if self.route == "policy":
            required = ("answer", "grounded", "fallback_triggered", "sources")
            missing = [key for key in required if key not in evidence]
            if missing:
                self._raise_runtime_error(
                    "mock_llm_tool_result",
                    "RAG tool result is missing required fields: "
                    + ", ".join(missing),
                )
            if not isinstance(evidence["answer"], str):
                self._raise_runtime_error(
                    "mock_llm_tool_result",
                    "RAG tool result answer must be text.",
                )
            response_data = {
                "answer": evidence["answer"],
                "route": "policy",
                "grounded": evidence["grounded"],
                "fallback_triggered": evidence["fallback_triggered"],
                "sources": evidence["sources"],
                "pii_masked": self.pii_masked,
            }
        else:
            required = (
                "record_id",
                "status",
                "resolution_time_hours",
                "days_since_created",
                "escalation_score",
                "escalation_recommended",
            )
            missing = [key for key in required if key not in evidence]
            if missing:
                self._raise_runtime_error(
                    "mock_llm_tool_result",
                    "Ticket tool result is missing required fields: "
                    + ", ".join(missing),
                )
            record_id = evidence["record_id"]
            if self.record_id and record_id != self.record_id:
                self._raise_runtime_error(
                    "mock_llm_tool_result",
                    "Ticket tool result record ID does not match the routed ticket.",
                )
            resolution_time = evidence["resolution_time_hours"]
            age = evidence["days_since_created"]
            recommended = evidence["escalation_recommended"]
            answer = (
                f"Ticket {record_id} is {evidence['status']}, has a resolution "
                f"time of {resolution_time} hours, is {age} days old, and "
                f"escalation is {'recommended' if recommended else 'not recommended'}."
            )
            response_data = {
                "answer": answer,
                "route": "ticket",
                "grounded": True,
                "fallback_triggered": False,
                "record_id": record_id,
                "ticket_status": evidence["status"],
                "resolution_time_hours": resolution_time,
                "ticket_age_days": age,
                "escalation_score": evidence["escalation_score"],
                "escalation_recommended": recommended,
                "pii_masked": self.pii_masked,
            }
        try:
            response = SupportResponse.model_validate(response_data)
        except ValidationError as exc:
            self._raise_runtime_error(
                "mock_llm_tool_result",
                f"Tool result cannot form a valid SupportResponse: {exc}",
                cause=exc,
            )
        return response.model_dump_json()

    def _response_from_tool_result(self, result_payload: Any) -> str:
        try:
            evidence = _parse_tool_result_payload(result_payload)
        except ToolResultNormalizationError as exc:
            self._raise_runtime_error(
                "mock_llm_tool_result",
                str(exc),
                cause=exc,
            )
        expected_keys = (
            {"answer", "grounded", "fallback_triggered", "sources"}
            if self.route == "policy"
            else {"record_id", "resolution_time_hours"}
        )
        for key in (
            "tool_result",
            "function_result",
            "function_response",
            "result",
            "output",
            "content",
            "data",
        ):
            nested = evidence.get(key)
            if isinstance(nested, (str, bytes, BaseModel, Mapping)):
                try:
                    nested = _parse_tool_result_payload(nested)
                except ToolResultNormalizationError:
                    continue
            if isinstance(nested, Mapping) and expected_keys.issubset(nested):
                evidence = nested
                break
        missing = sorted(expected_keys - evidence.keys())
        if missing:
            self._raise_runtime_error(
                "mock_llm_tool_result",
                _tool_result_diagnostic(
                    result_payload,
                    "Tool result object is missing required evidence fields: "
                    + ", ".join(missing),
                ),
            )
        return self._response_from_evidence(evidence)

    def _response_from_audit(self) -> str:
        evidence: dict[str, Any] | None = None
        tool_name: str | None = None
        if self.audit is not None:
            for invocation in reversed(self.audit.tool_invocations):
                if invocation.status == "completed" and isinstance(
                    invocation.result, dict
                ):
                    evidence = invocation.result
                    tool_name = invocation.tool_name
                    break
        if evidence is None:
            self._raise_runtime_error(
                "mock_llm_tool_result",
                "No completed tool result is available for the final response.",
            )
        expected_tool = (
            "rag_lookup" if self.route == "policy" else "ticket_status_lookup"
        )
        if tool_name != expected_tool:
            self._raise_runtime_error(
                "mock_llm_tool_result",
                f"Expected {expected_tool}, received {tool_name}.",
            )
        return self._response_from_evidence(evidence)

    def call(
        self,
        messages: str | list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
        **kwargs: Any,
    ) -> str:
        # CrewAI executes native tool calls; the mock only emits calls and reads
        # the returned tool message. Framework metadata and function callbacks
        # are intentionally ignored.
        del (
            callbacks,
            available_functions,
            from_task,
            from_agent,
            response_model,
            kwargs,
        )
        actual_results = _tool_result_messages(messages)
        if actual_results:
            return self._response_from_tool_result(actual_results[-1])

        if tools:
            selected = self._select_tool(tools, _message_text(messages))
            if selected is None:
                self._raise_runtime_error(
                    "mock_llm_tool_call",
                    "No declared tool schema matches the routed arguments.",
                )
            function_name, arguments = selected
            return [
                {
                    "id": "mock-tool-call-1",
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": json.dumps(arguments),
                    },
                }
            ]

        return self._response_from_audit()
