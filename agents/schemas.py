"""Validated response and run-audit models for the support agent."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_serializer,
    model_validator,
)


class SourceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str | None = None
    chunk_id: str | None = None
    similarity: float | None = Field(default=None, ge=0.0, le=1.0)
    retrieval_origin: Literal["vector_search", "parent_expansion"] | None = None

    @model_serializer(mode="wrap")
    def serialize_without_unspecified_origin(self, handler):
        result = handler(self)
        if self.retrieval_origin is None:
            result.pop("retrieval_origin", None)
        return result


class SupportResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    route: Literal["policy", "ticket", "blocked", "fallback"]
    grounded: bool
    fallback_triggered: bool
    sources: list[SourceReference] = Field(default_factory=list)
    record_id: str | None = None
    ticket_status: str | None = None
    resolution_time_hours: float | None = Field(default=None, ge=0.0)
    ticket_age_days: int | None = Field(default=None, ge=0)
    escalation_score: float | None = Field(default=None, ge=0.0, le=1.0)
    escalation_recommended: bool | None = None
    pii_masked: bool = False

    @model_validator(mode="after")
    def validate_ticket_fields(self) -> SupportResponse:
        if self.route == "ticket":
            missing = [
                field
                for field in (
                    "record_id",
                    "ticket_status",
                    "resolution_time_hours",
                    "ticket_age_days",
                    "escalation_score",
                    "escalation_recommended",
                )
                if getattr(self, field) is None
            ]
            if missing:
                raise ValueError(
                    "ticket responses require fields: " + ", ".join(missing)
                )
        return self


class ToolInvocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invocation_id: str
    tool_name: str
    arguments: dict[str, Any]
    status: Literal["running", "completed", "failed"] = "running"
    result: Any = None
    error: str | None = None


class CrewAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    route: str | None = None
    status: str = "created"
    kickoff_called: bool = False
    participating_agents: list[str] = Field(default_factory=list)
    tool_invocations: list[ToolInvocation] = Field(default_factory=list)
    pii_masked: bool = False
    sanitized_input: str | None = None
    guardrail_reason: str | None = None
    error: str | None = None

    @property
    def tool_invocation_count(self) -> int:
        return len(self.tool_invocations)


class CrewRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response: SupportResponse | None
    audit: CrewAudit
