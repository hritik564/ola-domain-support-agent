"""FastAPI application exposing support questions, ticket status, and chat."""

from __future__ import annotations

import time
import uuid
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.logging import append_request_log
from api.schemas import (
    APIError,
    AskRequest,
    AskResponse,
    TicketResponse,
    WebSocketResponse,
)
from api.service import (
    APIServiceError,
    ask_support_async,
    log_invalid_request,
    lookup_ticket,
    process_websocket_message,
)


def _safe_endpoint_path(path: str) -> str:
    if path == "/ask":
        return "/ask"
    if path.startswith("/ticket/"):
        return "/ticket/{record_id}"
    if path == "/ws/chat":
        return "/ws/chat"
    return "other"


def _connection_session_id(
    raw_message: str,
    current_session_id: str | None,
) -> str:
    """Retain a supplied ID before processing, including when processing fails."""
    try:
        body = json.loads(raw_message)
    except (json.JSONDecodeError, TypeError):
        body = None
    if isinstance(body, dict):
        requested_id = body.get("session_id")
        if isinstance(requested_id, str) and requested_id.strip():
            return requested_id
    return current_session_id or str(uuid.uuid4())


def create_app(*, log_path: str | Path | None = None) -> FastAPI:
    application = FastAPI(
        title="OLA Support Agent API",
        version="1.0.0",
        description="HTTP and WebSocket access to the existing support agent.",
    )

    @application.middleware("http")
    async def trace_http_requests(request: Request, call_next):
        trace_id = str(uuid.uuid4())
        request.state.trace_id = trace_id
        request.state.started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Trace-ID"] = trace_id
        response.headers["X-Request-Duration-Ms"] = (
            f"{max(0.0, (time.perf_counter() - request.state.started) * 1000):.3f}"
        )
        return response

    @application.exception_handler(APIServiceError)
    async def handle_service_error(
        _request: Request,
        exc: APIServiceError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.as_response().model_dump(mode="json"),
        )

    @application.exception_handler(RequestValidationError)
    async def handle_request_validation(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        trace_id = getattr(request.state, "trace_id", str(uuid.uuid4()))
        error = log_invalid_request(
            trace_id=trace_id,
            endpoint=_safe_endpoint_path(request.url.path),
            body=exc.body,
            transport="http",
            duration_ms=max(
                0.0,
                (time.perf_counter() - getattr(request.state, "started", time.perf_counter()))
                * 1000,
            ),
            log_path=log_path,
        )
        return JSONResponse(
            status_code=422,
            content=error.model_dump(mode="json"),
        )

    @application.exception_handler(Exception)
    async def handle_unexpected_error(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        trace_id = getattr(request.state, "trace_id", str(uuid.uuid4()))
        try:
            body = await request.json()
        except Exception:
            body = None
        query = body.get("query") if isinstance(body, dict) else None
        session_id = body.get("session_id") if isinstance(body, dict) else None
        append_request_log(
            trace_id=trace_id,
            transport="http",
            endpoint=_safe_endpoint_path(request.url.path),
            session_id=session_id if isinstance(session_id, str) else None,
            route="error",
            status=500,
            duration_ms=max(
                0.0,
                (time.perf_counter() - getattr(request.state, "started", time.perf_counter()))
                * 1000,
            ),
            query=query if isinstance(query, str) else None,
            error_type=type(exc).__name__,
            safe_error_message="The request could not be completed.",
            log_path=log_path,
        )
        return JSONResponse(
            status_code=500,
            content=APIError(
                code="service_error",
                message="The request could not be completed.",
                trace_id=trace_id,
            ).model_dump(mode="json"),
        )

    @application.post("/ask", response_model=AskResponse)
    async def ask_endpoint(request: AskRequest, http_request: Request) -> AskResponse:
        return await ask_support_async(
            request,
            trace_id=http_request.state.trace_id,
            transport="http",
            endpoint="/ask",
            started_at=http_request.state.started,
            log_path=log_path,
        )

    @application.get(
        "/ticket/{record_id}",
        response_model=TicketResponse,
    )
    def ticket_endpoint(record_id: str, request: Request) -> TicketResponse:
        return lookup_ticket(
            record_id,
            trace_id=request.state.trace_id,
            log_path=log_path,
        )

    @application.websocket("/ws/chat")
    async def websocket_chat(websocket: WebSocket) -> None:
        await websocket.accept()
        session_id: str | None = None
        try:
            while True:
                raw_message = await websocket.receive_text()
                session_id = _connection_session_id(raw_message, session_id)
                result = await process_websocket_message(
                    raw_message,
                    default_session_id=session_id,
                    log_path=log_path,
                )
                if isinstance(result, AskResponse):
                    session_id = result.session_id
                    result = WebSocketResponse.model_validate(result.model_dump())
                await websocket.send_json(result.model_dump(mode="json"))
        except WebSocketDisconnect:
            return

    return application


app = create_app()
