# Batch D risk register

**Classification: MEDIUM**

Customer-support decisions affect user experience, and ticket escalation
recommendations may affect operational handling. The service reads customer
support policies and ticket records, so answer quality and disclosure remain
material risks. A human/operator can review outcomes.

The workflow is read-only. It cannot issue refunds or payments, modify customer
accounts, close tickets automatically, contact customers, or write to ticket
records. Write access is intentionally absent.

Controls include evidence-grounded response validation, a two-turn AutoGen
review/editor pass with no tools, PII-aware logging and cache bypass, sanitized
client errors, bounded request estimates, and append-only request logs. The
remaining risk is primarily an incorrect or incomplete read-only answer.

## Tool-access matrix

| Role | Allowed tools | Denied capabilities |
| --- | --- | --- |
| Retrieval Agent | `rag_lookup` | Ticket lookup and all write/financial actions |
| Lookup Agent | `ticket_status_lookup` | Policy retrieval and all write/financial actions |
| Response Composer | None | All operational tools |
| Policy Compliance Reviewer | None | All operational tools |
| Final Editor | None | All operational tools |

The AutoGen review stage receives only a draft and a minimized evidence
summary. It cannot call ticket, retrieval, write, refund, account, payment, or
communication tools.
