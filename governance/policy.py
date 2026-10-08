"""Least-autonomy tool access policy for support and review roles."""

from __future__ import annotations

AGENT_TOOL_ACCESS: dict[str, tuple[str, ...]] = {
    "Retrieval Agent": ("rag_lookup",),
    "Lookup Agent": ("ticket_status_lookup",),
    "Response Composer": (),
    "Policy Compliance Reviewer": (),
    "Final Editor": (),
}

AUTOGEN_AGENT_NAMES = {
    "Policy Compliance Reviewer": "policy_compliance_reviewer",
    "Final Editor": "final_editor",
}

PROHIBITED_ACTIONS = (
    "ticket mutation",
    "refund or compensation issuance",
    "account changes",
    "external communication",
    "billing or payment actions",
)

RISK_CLASSIFICATION = "MEDIUM"
MEDIUM_RISK_RATIONALE = (
    "Customer-support decisions affect user experience, and ticket escalation "
    "recommendations may affect operational handling.",
    "The system cannot issue refunds or payments, modify accounts, or close "
    "tickets automatically; write access is intentionally absent.",
    "A human/operator can review outcomes. Groundedness checks, PII masking, "
    "sanitized API errors, and audit logs reduce the remaining risk.",
)


def validate_tool_access_policy() -> None:
    """Fail explicitly if the documented least-autonomy boundary is weakened."""
    if AGENT_TOOL_ACCESS["Lookup Agent"] != ("ticket_status_lookup",):
        raise ValueError("Only the Lookup Agent may access ticket records.")
    if AGENT_TOOL_ACCESS["Retrieval Agent"] != ("rag_lookup",):
        raise ValueError("Only the Retrieval Agent may access policy retrieval.")
    for role in ("Response Composer", "Policy Compliance Reviewer", "Final Editor"):
        if AGENT_TOOL_ACCESS[role]:
            raise ValueError(f"{role} must not have operational tools.")
