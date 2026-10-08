"""Application service layer calling the existing support and ticket logic."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal

import anyio
from pydantic import ValidationError

from agents.crew import run_support_crew
from agents.schemas import SupportResponse
from api.logging import append_request_log, mask_log_text
from api.schemas import APIError, AskRequest, AskResponse, TicketResponse, WebSocketRequest
from budgeting.request_budget import (
    DEFAULT_BUDGET_LIMITS,
    BudgetLimits,
    RequestBudget,
    estimate_request_budget,
    with_actual_output_tokens,
    with_elapsed_time,
)
from cache.normalized_cache import NormalizedResponseCache, build_cache_key
from review.autogen_review import (
    ReviewDependencyUnavailable,
    ReviewResult,
    review_support_response,
)
from tools.ticket_lookup import check_support_ticket_status


@dataclass(frozen=True)
class _CachedSupportResult:
    response: SupportResponse
    review: ReviewResult


_SUPPORT_RESPONSE_CACHE: NormalizedResponseCache[_CachedSupportResult] = (
    NormalizedResponseCache(max_entries=128)
)
REQUEST_BUDGET_LIMITS = DEFAULT_BUDGET_LIMITS


class APIServiceError(Exception):
    """An error with a safe, client-facing message and status code."""

    def __init__(
        self,
        *,
        status_code: int,
        code: Literal[
            "invalid_request",
            "ticket_not_found",
            "dependency_unavailable",
            "service_error",
            "request_budget_exceeded",
        ],
        message: str,
        trace_id: str,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.trace_id = trace_id

    def as_response(self) -> APIError:
        return APIError(code=self.code, message=self.message, trace_id=self.trace_id)


def _new_id() -> str:
    return str(uuid.uuid4())


def _batch_d_log_data(
    budget: RequestBudget,
    *,
    cache_hit: bool,
    cache_key: str | None,
    cache_reason: str,
    cache_size: int,
    review: ReviewResult | None,
    elapsed_ms: float,
    limits: BudgetLimits,
) -> dict[str, Any]:
    observed_budget = with_elapsed_time(budget, elapsed_ms, limits=limits)
    data: dict[str, Any] = {
        "budget": {
            **observed_budget.as_log_data(),
            "limits": {
                "max_input_tokens": limits.max_input_tokens,
                "max_output_tokens": limits.max_output_tokens,
                "max_total_tokens": limits.max_total_tokens,
                "max_runtime_ms": limits.max_runtime_ms,
            },
        },
        "cache": {
            "cache_hit": cache_hit,
            "cache_key": cache_key,
            "eligibility": cache_reason,
            "cache_size": cache_size,
            "max_entries": _SUPPORT_RESPONSE_CACHE.max_entries,
        },
    }
    if review is not None:
        data["review"] = {
            "approved": review.approved,
            "revised": review.revised,
            "reviewer_count": review.reviewer_count,
            "turn_count": review.turn_count,
        }
    return data


def _append_support_log(
    *,
    trace_id: str,
    transport: Literal["http", "websocket", "evaluation"],
    endpoint: str,
    session_id: str,
    route: str,
    status: int,
    started: float,
    query: str,
    pii_masked: bool,
    budget: RequestBudget,
    cache_hit: bool,
    cache_key: str | None,
    cache_reason: str,
    review: ReviewResult | None,
    limits: BudgetLimits | None = None,
    error_type: str | None = None,
    safe_error_message: str | None = None,
    log_path: str | Path | None = None,
) -> None:
    elapsed_ms = (time.perf_counter() - started) * 1000
    append_request_log(
        trace_id=trace_id,
        transport=transport,
        endpoint=endpoint,
        session_id=session_id,
        route=route,
        status=status,
        duration_ms=elapsed_ms,
        query=query,
        pii_masked=pii_masked,
        error_type=error_type,
        safe_error_message=safe_error_message,
        batch_d=_batch_d_log_data(
            budget,
            cache_hit=cache_hit,
            cache_key=cache_key,
            cache_reason=cache_reason,
            cache_size=_SUPPORT_RESPONSE_CACHE.size,
            review=review,
            elapsed_ms=elapsed_ms,
            limits=limits or REQUEST_BUDGET_LIMITS,
        ),
        log_path=log_path,
    )


def _cache_safe_response(response: SupportResponse) -> bool:
    serialized = json.dumps(
        response.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
    )
    _safe_text, pii_detected = mask_log_text(serialized)
    return not pii_detected


def ask_support(
    request: AskRequest,
    *,
    trace_id: str | None = None,
    transport: Literal["http", "websocket", "evaluation"] = "http",
    endpoint: str = "/ask",
    started_at: float | None = None,
    log_path: str | Path | None = None,
) -> AskResponse:
    """Call the existing CrewAI support service and return its validated response."""
    resolved_trace_id = trace_id or _new_id()
    resolved_session_id = request.session_id or _new_id()
    started = started_at if started_at is not None else time.perf_counter()
    query_masked, pii_masked = mask_log_text(request.query)
    session_masked, session_had_pii = mask_log_text(resolved_session_id)
    pii_masked = pii_masked or session_had_pii
    budget = estimate_request_budget(
        request.query,
        limits=REQUEST_BUDGET_LIMITS,
    )
    cache_key: str | None = None
    cache_reason = "budget_rejected"
    review_result: ReviewResult | None = None
    cache_hit = False

    if not budget.allowed:
        message = "This request exceeds the configured support request budget."
        _append_support_log(
            trace_id=resolved_trace_id,
            transport=transport,
            endpoint=endpoint,
            session_id=resolved_session_id,
            route="budget",
            status=413,
            started=started,
            query=request.query,
            pii_masked=pii_masked,
            budget=budget,
            cache_hit=False,
            cache_key=cache_key,
            cache_reason=cache_reason,
            review=None,
            error_type=budget.reason or "request_budget_exceeded",
            safe_error_message=message,
            limits=REQUEST_BUDGET_LIMITS,
            log_path=log_path,
        )
        raise APIServiceError(
            status_code=413,
            code="request_budget_exceeded",
            message=message,
            trace_id=resolved_trace_id,
        )

    cache_key, cache_reason = build_cache_key(
        request.query,
        session_id=resolved_session_id,
    )
    try:
        if cache_key is not None:
            cached = _SUPPORT_RESPONSE_CACHE.get(cache_key)
            if (
                cached.hit
                and cached.value is not None
                and cached.value.response.route in {"policy", "fallback"}
            ):
                cached_response = cached.value.response
                budget = with_actual_output_tokens(
                    budget,
                    len(cached_response.answer),
                    limits=REQUEST_BUDGET_LIMITS,
                )
                if not budget.allowed:
                    message = "This request exceeds the configured support request budget."
                    _append_support_log(
                        trace_id=resolved_trace_id,
                        transport=transport,
                        endpoint=endpoint,
                        session_id=resolved_session_id,
                        route="budget",
                        status=413,
                        started=started,
                        query=request.query,
                        pii_masked=pii_masked,
                        budget=budget,
                        cache_hit=False,
                        cache_key=cache_key,
                        cache_reason=cache_reason,
                        review=None,
                        error_type=budget.reason or "request_budget_exceeded",
                        safe_error_message=message,
                        limits=REQUEST_BUDGET_LIMITS,
                        log_path=log_path,
                    )
                    raise APIServiceError(
                        status_code=413,
                        code="request_budget_exceeded",
                        message=message,
                        trace_id=resolved_trace_id,
                    )
                cache_hit = True
                review_result = cached.value.review
                response = AskResponse.model_validate(
                    {
                        **cached_response.model_dump(),
                        "trace_id": resolved_trace_id,
                        "session_id": resolved_session_id,
                    }
                )
                _append_support_log(
                    trace_id=resolved_trace_id,
                    transport=transport,
                    endpoint=endpoint,
                    session_id=resolved_session_id,
                    route=response.route,
                    status=200,
                    started=started,
                    query=request.query,
                    pii_masked=pii_masked or response.pii_masked,
                    budget=budget,
                    cache_hit=True,
                    cache_key=cache_key,
                    cache_reason=cache_reason,
                    review=review_result,
                    limits=REQUEST_BUDGET_LIMITS,
                    log_path=log_path,
                )
                return response

        result = run_support_crew(
            request.query,
            session_id=resolved_session_id,
        )
        if result.response is None:
            blocked_dependency = result.audit.status == "blocked_dependency"
            status_code = 503 if blocked_dependency else 500
            error_code = (
                "dependency_unavailable" if blocked_dependency else "service_error"
            )
            message = (
                "The support service is temporarily unavailable."
                if blocked_dependency
                else "The support request could not be completed."
            )
            error = APIServiceError(
                status_code=status_code,
                code=error_code,
                message=message,
                trace_id=resolved_trace_id,
            )
            _append_support_log(
                trace_id=resolved_trace_id,
                transport=transport,
                endpoint=endpoint,
                session_id=resolved_session_id,
                route=result.audit.route or "unknown",
                status=status_code,
                started=started,
                query=request.query,
                pii_masked=pii_masked or result.audit.pii_masked,
                budget=budget,
                cache_hit=False,
                cache_key=cache_key,
                cache_reason=cache_reason,
                review=None,
                error_type=error_code,
                safe_error_message=message,
                limits=REQUEST_BUDGET_LIMITS,
                log_path=log_path,
            )
            raise error

        execution = review_support_response(result.response, result.audit)
        review_result = execution.result
        reviewed_support_response = SupportResponse.model_validate(
            result.response.model_copy(
                update={"answer": review_result.final_answer}
            ).model_dump()
        )
        budget = with_actual_output_tokens(
            budget,
            len(reviewed_support_response.answer),
            limits=REQUEST_BUDGET_LIMITS,
        )
        if not budget.allowed:
            message = "This request exceeds the configured support request budget."
            _append_support_log(
                trace_id=resolved_trace_id,
                transport=transport,
                endpoint=endpoint,
                session_id=resolved_session_id,
                route="budget",
                status=413,
                started=started,
                query=request.query,
                pii_masked=pii_masked or reviewed_support_response.pii_masked,
                budget=budget,
                cache_hit=False,
                cache_key=cache_key,
                cache_reason=cache_reason,
                review=review_result,
                error_type=budget.reason or "request_budget_exceeded",
                safe_error_message=message,
                limits=REQUEST_BUDGET_LIMITS,
                log_path=log_path,
            )
            raise APIServiceError(
                status_code=413,
                code="request_budget_exceeded",
                message=message,
                trace_id=resolved_trace_id,
            )
        response = AskResponse.model_validate(
            {
                **reviewed_support_response.model_dump(),
                "trace_id": resolved_trace_id,
                "session_id": resolved_session_id,
            }
        )
        if (
            cache_key is not None
            and response.route in {"policy", "fallback"}
            and _cache_safe_response(reviewed_support_response)
        ):
            cached_response = SupportResponse.model_validate(
                reviewed_support_response.model_dump()
            )
            _SUPPORT_RESPONSE_CACHE.put(
                cache_key,
                _CachedSupportResult(
                    response=cached_response,
                    review=review_result,
                ),
            )
        _append_support_log(
            trace_id=resolved_trace_id,
            transport=transport,
            endpoint=endpoint,
            session_id=resolved_session_id,
            route=response.route,
            status=200,
            started=started,
            query=request.query,
            pii_masked=pii_masked or response.pii_masked,
            budget=budget,
            cache_hit=False,
            cache_key=cache_key,
            cache_reason=cache_reason,
            review=review_result,
            limits=REQUEST_BUDGET_LIMITS,
            log_path=log_path,
        )
        return response
    except ReviewDependencyUnavailable:
        message = "The support service is temporarily unavailable."
        _append_support_log(
            trace_id=resolved_trace_id,
            transport=transport,
            endpoint=endpoint,
            session_id=resolved_session_id,
            route="error",
            status=503,
            started=started,
            query=request.query,
            pii_masked=pii_masked,
            budget=budget,
            cache_hit=cache_hit,
            cache_key=cache_key,
            cache_reason=cache_reason,
            review=review_result,
            limits=REQUEST_BUDGET_LIMITS,
            error_type="dependency_unavailable",
            safe_error_message=message,
            log_path=log_path,
        )
        raise APIServiceError(
            status_code=503,
            code="dependency_unavailable",
            message=message,
            trace_id=resolved_trace_id,
        ) from None
    except APIServiceError:
        raise
    except Exception as exc:
        message = "The support request could not be completed."
        _append_support_log(
            trace_id=resolved_trace_id,
            transport=transport,
            endpoint=endpoint,
            session_id=resolved_session_id,
            route="error",
            status=500,
            started=started,
            query=request.query,
            pii_masked=pii_masked,
            budget=budget,
            cache_hit=cache_hit,
            cache_key=cache_key,
            cache_reason=cache_reason,
            review=review_result,
            limits=REQUEST_BUDGET_LIMITS,
            error_type=type(exc).__name__,
            safe_error_message=message,
            log_path=log_path,
        )
        raise APIServiceError(
            status_code=500,
            code="service_error",
            message=message,
            trace_id=resolved_trace_id,
        ) from None


async def ask_support_async(
    request: AskRequest,
    *,
    trace_id: str | None = None,
    transport: Literal["http", "websocket", "evaluation"] = "http",
    endpoint: str = "/ask",
    started_at: float | None = None,
    log_path: str | Path | None = None,
) -> AskResponse:
    """Run the unchanged synchronous support service outside the event loop."""
    sync_call = partial(
        ask_support,
        request,
        trace_id=trace_id,
        transport=transport,
        endpoint=endpoint,
        started_at=started_at,
        log_path=log_path,
    )
    return await anyio.to_thread.run_sync(sync_call)


def lookup_ticket(
    record_id: str,
    *,
    trace_id: str | None = None,
    log_path: str | Path | None = None,
) -> TicketResponse:
    """Look up a ticket through the existing deterministic lookup function."""
    resolved_trace_id = trace_id or _new_id()
    started = time.perf_counter()
    try:
        ticket = check_support_ticket_status(record_id)
        response = TicketResponse(
            record_id=ticket["record_id"],
            status=ticket["status"],
            resolution_time_hours=ticket["resolution_time_hours"],
            ticket_age_days=ticket["days_since_created"],
            escalation_score=ticket["escalation_score"],
            escalation_recommended=ticket["escalation_recommended"],
        )
        append_request_log(
            trace_id=resolved_trace_id,
            transport="http",
            endpoint="/ticket/{record_id}",
            session_id=None,
            route="ticket",
            status=200,
            duration_ms=(time.perf_counter() - started) * 1000,
            log_path=log_path,
        )
        return response
    except ValueError:
        message = "The requested support ticket was not found."
        append_request_log(
            trace_id=resolved_trace_id,
            transport="http",
            endpoint="/ticket/{record_id}",
            session_id=None,
            route="ticket",
            status=404,
            duration_ms=(time.perf_counter() - started) * 1000,
            error_type="TicketNotFound",
            safe_error_message=message,
            log_path=log_path,
        )
        raise APIServiceError(
            status_code=404,
            code="ticket_not_found",
            message=message,
            trace_id=resolved_trace_id,
        ) from None
    except Exception as exc:
        message = "The ticket lookup could not be completed."
        append_request_log(
            trace_id=resolved_trace_id,
            transport="http",
            endpoint="/ticket/{record_id}",
            session_id=None,
            route="error",
            status=500,
            duration_ms=(time.perf_counter() - started) * 1000,
            error_type=type(exc).__name__,
            safe_error_message=message,
            log_path=log_path,
        )
        raise APIServiceError(
            status_code=500,
            code="service_error",
            message=message,
            trace_id=resolved_trace_id,
        ) from None


def _safe_validation_log_values(
    body: Any,
) -> tuple[str | None, str | None, bool]:
    if not isinstance(body, dict):
        return None, None, False
    query = body.get("query") if isinstance(body.get("query"), str) else None
    session_id = (
        body.get("session_id") if isinstance(body.get("session_id"), str) else None
    )
    _safe_query, query_masked = mask_log_text(query)
    _safe_session, session_masked = mask_log_text(session_id)
    return query, session_id, query_masked or session_masked


def log_invalid_request(
    *,
    trace_id: str,
    endpoint: str,
    body: Any,
    transport: Literal["http", "websocket"],
    duration_ms: float = 0.0,
    log_path: str | Path | None = None,
) -> APIError:
    query, session_id, pii_masked = _safe_validation_log_values(body)
    message = "The request is invalid."
    append_request_log(
        trace_id=trace_id,
        transport=transport,
        endpoint=endpoint,
        session_id=session_id,
        route="invalid_request",
        status=422,
        duration_ms=max(0.0, duration_ms),
        query=query,
        pii_masked=pii_masked,
        error_type="RequestValidationError",
        safe_error_message=message,
        log_path=log_path,
    )
    return APIError(
        code="invalid_request",
        message=message,
        trace_id=trace_id,
    )


async def process_websocket_message(
    raw_message: str,
    *,
    default_session_id: str | None = None,
    trace_id: str | None = None,
    log_path: str | Path | None = None,
) -> AskResponse | APIError:
    """Validate a WebSocket message and await the shared threaded service."""
    message_trace_id = trace_id or _new_id()
    started = time.perf_counter()
    try:
        body = json.loads(raw_message)
    except (json.JSONDecodeError, TypeError):
        message = "The message must be a valid JSON object."
        append_request_log(
            trace_id=message_trace_id,
            transport="websocket",
            endpoint="/ws/chat",
            session_id=default_session_id,
            route="invalid_request",
            status=422,
            duration_ms=(time.perf_counter() - started) * 1000,
            error_type="MalformedJSON",
            safe_error_message=message,
            log_path=log_path,
        )
        return APIError(
            code="invalid_request",
            message=message,
            trace_id=message_trace_id,
        )

    try:
        request = WebSocketRequest.model_validate(body)
    except ValidationError:
        return log_invalid_request(
            trace_id=message_trace_id,
            endpoint="/ws/chat",
            body=body,
            transport="websocket",
            duration_ms=(time.perf_counter() - started) * 1000,
            log_path=log_path,
        )

    if request.session_id is None and default_session_id is not None:
        request = request.model_copy(update={"session_id": default_session_id})
    try:
        return await ask_support_async(
            request,
            trace_id=message_trace_id,
            transport="websocket",
            endpoint="/ws/chat",
            started_at=started,
            log_path=log_path,
        )
    except APIServiceError as error:
        return error.as_response()
