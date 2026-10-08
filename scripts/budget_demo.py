"""Render deterministic request-token, cost, and runtime budget examples."""

from __future__ import annotations

from pathlib import Path

from budgeting.request_budget import (
    DEFAULT_BUDGET_LIMITS,
    estimate_request_budget,
    with_actual_output_tokens,
    with_elapsed_time,
)


def render_demo() -> str:
    query = "What is the P1 SLA?"
    answer = (
        "For P1, the initial response is due within 15 minutes and the target "
        "resolution time is within 4 hours."
    )
    allowed = with_actual_output_tokens(
        estimate_request_budget(query),
        len(answer),
    )
    oversized = estimate_request_budget("x" * 2100)
    output_oversized = with_actual_output_tokens(
        estimate_request_budget(query),
        1025,
    )
    runtime_observation = with_elapsed_time(allowed, 30_250.0)

    return "\n".join(
        (
            "Batch D request-budget demo",
            "Token estimate: ceil(input characters / 4).",
            "Output estimate: reserve the configured output-token limit.",
            "Cost estimate uses illustrative local accounting rates, not provider billing.",
            f"Input rate per 1M tokens (USD): {DEFAULT_BUDGET_LIMITS.input_cost_per_million_usd}",
            f"Output rate per 1M tokens (USD): {DEFAULT_BUDGET_LIMITS.output_cost_per_million_usd}",
            "",
            "ALLOWED REQUEST",
            f"allowed: {str(allowed.allowed).lower()}",
            f"estimated_input_tokens: {allowed.estimated_input_tokens}",
            f"estimated_output_tokens: {allowed.estimated_output_tokens}",
            f"estimated_total_tokens: {allowed.estimated_total_tokens}",
            f"actual_output_tokens: {allowed.actual_output_tokens}",
            f"actual_total_tokens: {allowed.actual_total_tokens}",
            f"estimated_cost_usd: {allowed.estimated_cost_usd}",
            f"actual_paid_cost_usd: {allowed.actual_paid_cost_usd}",
            "",
            "OVERSIZED REQUEST",
            f"allowed: {str(oversized.allowed).lower()}",
            f"reason: {oversized.reason}",
            f"estimated_input_tokens: {oversized.estimated_input_tokens}",
            "",
            "OUTPUT OVERSIZE",
            f"allowed: {str(output_oversized.allowed).lower()}",
            f"reason: {output_oversized.reason}",
            f"actual_output_tokens: {output_oversized.actual_output_tokens}",
            "",
            "RUNTIME OBSERVATION",
            f"elapsed_ms: {runtime_observation.elapsed_ms}",
            f"runtime_budget_exceeded: {str(runtime_observation.runtime_budget_exceeded).lower()}",
            "Runtime overage is logged; synchronous work is not interrupted.",
            "",
            "Configured limits:",
            f"max_input_tokens: {DEFAULT_BUDGET_LIMITS.max_input_tokens}",
            f"max_output_tokens: {DEFAULT_BUDGET_LIMITS.max_output_tokens}",
            f"max_total_tokens: {DEFAULT_BUDGET_LIMITS.max_total_tokens}",
            f"max_runtime_ms: {DEFAULT_BUDGET_LIMITS.max_runtime_ms}",
            "",
        )
    )


def main() -> None:
    output = render_demo()
    destination = Path(__file__).resolve().parents[1] / "transcripts" / "budget_demo.txt"
    destination.write_text(output, encoding="utf-8")
    print(output, end="")
    print(f"Saved: {destination.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
