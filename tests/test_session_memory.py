from memory.session_memory import TicketSessionMemory


def test_ticket_reference_is_retained_only_within_its_session():
    memory = TicketSessionMemory()
    assert memory.resolve_record_id("session-A", "Check TKT-0007.") == "TKT-0007"
    assert (
        memory.resolve_record_id("session-A", "Should that ticket be escalated?")
        == "TKT-0007"
    )
    assert (
        memory.resolve_record_id("session-B", "Should that ticket be escalated?")
        is None
    )
    assert len(memory.get_session_history("session-A").messages) == 4
    assert len(memory.get_session_history("session-B").messages) == 2


def test_clear_session_removes_only_that_session_history():
    memory = TicketSessionMemory()
    memory.resolve_record_id("session-A", "Check TKT-0007.")
    memory.resolve_record_id("session-B", "Check TKT-0008.")
    memory.clear_session("session-A")
    assert memory.resolve_record_id("session-A", "Should that ticket be escalated?") is None
    assert memory.resolve_record_id("session-B", "Should that ticket be escalated?") == "TKT-0008"
