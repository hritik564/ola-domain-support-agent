"""Privacy-aware, append-only JSONL request logging."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from guardrails.pii import mask_phone_numbers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_PATH = PROJECT_ROOT / "logs" / "requests.jsonl"
_LOG_LOCK = Lock()
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|token|secret|"
    r"password|authorization)\b\s*[:=]\s*([^\s,;]+)"
)
_BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_TOKEN_PREFIX = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{8,}|gh[pousr]_[A-Za-z0-9]{12,}|xox[baprs]-[A-Za-z0-9-]{10,})\b")


class RequestLogRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: str
    trace_id: str
    transport: Literal["http", "websocket", "evaluation"]
    endpoint: str
    session_id: str | None = None
    route: str
    status: int = Field(ge=100, le=599)
    duration_ms: float = Field(ge=0.0)
    pii_masked: bool
    masked_query: str | None = None
    error_type: str | None = None
    safe_error_message: str | None = None
    batch_d: dict[str, Any] | None = None


def mask_log_text(value: str | None) -> tuple[str | None, bool]:
    """Mask supported phone numbers, emails, and recognizable credentials."""
    if value is None:
        return None, False
    phone_result = mask_phone_numbers(value)
    masked = phone_result.text
    changed = phone_result.masked
    masked, email_count = _EMAIL.subn("[EMAIL REDACTED]", masked)
    changed = changed or email_count > 0
    masked, credential_count = _CREDENTIAL_ASSIGNMENT.subn(
        lambda match: f"{match.group(1)}=[REDACTED]",
        masked,
    )
    changed = changed or credential_count > 0
    masked, bearer_count = _BEARER.subn("Bearer [REDACTED]", masked)
    changed = changed or bearer_count > 0
    masked, token_count = _TOKEN_PREFIX.subn("[TOKEN REDACTED]", masked)
    changed = changed or token_count > 0
    return masked, changed


def _sanitize_metadata(value: Any) -> Any:
    if isinstance(value, str):
        return mask_log_text(value)[0]
    if isinstance(value, dict):
        return {key: _sanitize_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_metadata(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_metadata(item) for item in value]
    return value


def append_request_log(
    *,
    trace_id: str,
    transport: Literal["http", "websocket", "evaluation"],
    endpoint: str,
    session_id: str | None,
    route: str,
    status: int,
    duration_ms: float,
    query: str | None = None,
    pii_masked: bool = False,
    error_type: str | None = None,
    safe_error_message: str | None = None,
    batch_d: dict[str, Any] | None = None,
    log_path: str | Path | None = None,
) -> RequestLogRecord:
    """Append one validated record without retaining the raw query."""
    safe_query, query_changed = mask_log_text(query)
    safe_session, session_changed = mask_log_text(session_id)
    record = RequestLogRecord(
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        trace_id=trace_id,
        transport=transport,
        endpoint=endpoint,
        session_id=safe_session,
        route=route,
        status=status,
        duration_ms=max(0.0, float(duration_ms)),
        pii_masked=bool(pii_masked or query_changed or session_changed),
        masked_query=safe_query,
        error_type=error_type[:120] if error_type else None,
        safe_error_message=safe_error_message,
        batch_d=_sanitize_metadata(batch_d) if batch_d is not None else None,
    )
    destination = Path(log_path) if log_path is not None else DEFAULT_LOG_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = record.model_dump()
    for optional_field in (
        "masked_query",
        "error_type",
        "safe_error_message",
        "batch_d",
    ):
        if payload[optional_field] is None:
            payload.pop(optional_field)
    line = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    with _LOG_LOCK:
        with destination.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
    return record
