"""Deterministic request-budget estimates for the support API."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class BudgetLimits:
    """Configurable, local-only request limits and illustrative cost rates."""

    max_input_tokens: int = 512
    max_output_tokens: int = 256
    max_total_tokens: int = 768
    max_runtime_ms: float = 30_000.0
    input_cost_per_million_usd: float = 0.15
    output_cost_per_million_usd: float = 0.60


DEFAULT_BUDGET_LIMITS = BudgetLimits()


@dataclass(frozen=True)
class RequestBudget:
    allowed: bool
    reason: str | None
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_total_tokens: int
    estimated_cost_usd: float
    actual_output_tokens: int | None = None
    actual_total_tokens: int | None = None
    actual_paid_cost_usd: float = 0.0
    elapsed_ms: float = 0.0
    runtime_budget_exceeded: bool = False

    def as_log_data(self) -> dict[str, bool | float | int | str | None]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "estimated_input_tokens": self.estimated_input_tokens,
            "estimated_output_tokens": self.estimated_output_tokens,
            "estimated_total_tokens": self.estimated_total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "actual_output_tokens": self.actual_output_tokens,
            "actual_total_tokens": self.actual_total_tokens,
            "actual_paid_cost_usd": self.actual_paid_cost_usd,
            "elapsed_ms": self.elapsed_ms,
            "runtime_budget_exceeded": self.runtime_budget_exceeded,
        }


def estimate_request_budget(
    query: str,
    *,
    limits: BudgetLimits = DEFAULT_BUDGET_LIMITS,
) -> RequestBudget:
    """Estimate tokens as ceil(characters/4); reserve the configured output cap."""
    input_tokens = math.ceil(len(query) / 4)
    output_tokens = limits.max_output_tokens
    total_tokens = input_tokens + output_tokens

    reason: str | None = None
    if input_tokens > limits.max_input_tokens:
        reason = "input_token_budget_exceeded"
    elif total_tokens > limits.max_total_tokens:
        reason = "total_token_budget_exceeded"

    estimated_cost = (
        input_tokens * limits.input_cost_per_million_usd
        + output_tokens * limits.output_cost_per_million_usd
    ) / 1_000_000

    return RequestBudget(
        allowed=reason is None,
        reason=reason,
        estimated_input_tokens=input_tokens,
        estimated_output_tokens=output_tokens,
        estimated_total_tokens=total_tokens,
        estimated_cost_usd=round(estimated_cost, 12),
    )


def with_elapsed_time(
    budget: RequestBudget,
    elapsed_ms: float,
    *,
    limits: BudgetLimits = DEFAULT_BUDGET_LIMITS,
) -> RequestBudget:
    """Attach observed runtime and flag (but do not interrupt) over-budget work."""
    observed = max(0.0, float(elapsed_ms))
    return replace(
        budget,
        elapsed_ms=round(observed, 3),
        runtime_budget_exceeded=observed > limits.max_runtime_ms,
    )


def with_actual_output_tokens(
    budget: RequestBudget,
    output_character_count: int,
    *,
    limits: BudgetLimits = DEFAULT_BUDGET_LIMITS,
) -> RequestBudget:
    """Measure completed answer size and apply the configured output/total caps."""
    actual_output = math.ceil(max(0, output_character_count) / 4)
    actual_total = budget.estimated_input_tokens + actual_output
    reason = budget.reason
    if reason is None and actual_output > limits.max_output_tokens:
        reason = "output_token_budget_exceeded"
    elif reason is None and actual_total > limits.max_total_tokens:
        reason = "total_token_budget_exceeded"
    return replace(
        budget,
        allowed=reason is None,
        reason=reason,
        actual_output_tokens=actual_output,
        actual_total_tokens=actual_total,
    )
