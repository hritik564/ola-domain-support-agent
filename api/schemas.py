"""Validated request and response models for the support API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agents.schemas import SupportResponse


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value

    @field_validator("session_id")
    @classmethod
    def session_id_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("session_id must not be blank")
        return value


class WebSocketRequest(AskRequest):
    """One validated chat message received over WebSocket."""


class AskResponse(SupportResponse):
    trace_id: str
    session_id: str


class WebSocketResponse(AskResponse):
    """The validated support response sent for one chat message."""


class TicketResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_id: str
    status: str
    resolution_time_hours: float = Field(ge=0.0)
    ticket_age_days: int = Field(ge=0)
    escalation_score: float = Field(ge=0.0, le=1.0)
    escalation_recommended: bool


class APIError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: Literal[
        "invalid_request",
        "ticket_not_found",
        "dependency_unavailable",
        "service_error",
        "request_budget_exceeded",
    ]
    message: str
    trace_id: str
