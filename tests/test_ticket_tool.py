import json

from agents.audit import audit_context, new_audit
from tools.ticket_lookup import check_support_ticket_status
from tools.ticket_tool import TicketStatusTool


def test_ticket_tool_calls_existing_lookup_and_records_evidence():
    expected = check_support_ticket_status("TKT-0007")
    audit = new_audit()
    with audit_context(audit):
        raw = TicketStatusTool()._run("TKT-0007")
    result = json.loads(raw)
    assert result == expected
    assert audit.tool_invocation_count == 1
    assert audit.tool_invocations[0].tool_name == "ticket_status_lookup"
    assert audit.tool_invocations[0].arguments == {"record_id": "TKT-0007"}
    assert audit.tool_invocations[0].status == "completed"
