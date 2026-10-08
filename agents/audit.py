"""Per-run audit collection shared with CrewAI tool wrappers."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any, Iterator
from uuid import uuid4

from agents.schemas import CrewAudit, ToolInvocation

_CURRENT_AUDIT: ContextVar[CrewAudit | None] = ContextVar(
    "ola_support_crew_audit",
    default=None,
)


def new_audit() -> CrewAudit:
    return CrewAudit(trace_id=uuid4().hex)


@contextmanager
def audit_context(audit: CrewAudit) -> Iterator[None]:
    token: Token[CrewAudit | None] = _CURRENT_AUDIT.set(audit)
    try:
        yield
    finally:
        _CURRENT_AUDIT.reset(token)


def current_audit() -> CrewAudit | None:
    return _CURRENT_AUDIT.get()


def begin_tool_invocation(
    tool_name: str,
    arguments: dict[str, Any],
) -> str | None:
    audit = current_audit()
    if audit is None:
        return None
    invocation_id = str(len(audit.tool_invocations) + 1)
    audit.tool_invocations.append(
        ToolInvocation(
            invocation_id=invocation_id,
            tool_name=tool_name,
            arguments=arguments,
        )
    )
    return invocation_id


def finish_tool_invocation(
    invocation_id: str | None,
    *,
    result: Any = None,
    error: str | None = None,
) -> None:
    audit = current_audit()
    if audit is None or invocation_id is None:
        return
    invocation = next(
        (
            item
            for item in audit.tool_invocations
            if item.invocation_id == invocation_id
        ),
        None,
    )
    if invocation is None:
        return
    if error is not None:
        invocation.status = "failed"
        invocation.error = error
    else:
        invocation.status = "completed"
        invocation.result = result
