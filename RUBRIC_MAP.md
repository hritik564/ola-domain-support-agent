# Capstone rubric map

This map covers each requested item A–AT. Statuses are `PASS`, `PARTIAL`, or
`MISSING`; the first value records the audit before Batch E documentation and
acceptance-runner fixes, and the second is the final status.

**Before Batch E:** 43 PASS, 3 PARTIAL, 0 MISSING. The partial items were the
grader-facing README, a single reproducible acceptance command, and a final
environment-specific evidence transcript.  
**After Batch E:** 46 PASS. Real CrewAI/ChromaDB checks remain blocked in the
current Replit environment, but the user supplied separate real-local evidence
for those integration criteria. Replit results are never represented as local
integration passes.

| Item | Requirement | Before → final | Implementation files | Test / transcript evidence | Validation command |
| --- | --- | --- | --- | --- | --- |
| A | Deterministic dataset with at least 40 tickets | PASS → PASS | `dataset.py` | `tests/test_dataset.py::test_dataset_passes_capstone_constraints`; final acceptance transcript | `python -m pytest -q tests/test_dataset.py` |
| B | Required ticket categories | PASS → PASS | `dataset.py:CATEGORIES` | `tests/test_dataset.py::test_every_required_category_has_at_least_three_records` | `python -m pytest -q tests/test_dataset.py` |
| C | Required ticket statuses | PASS → PASS | `dataset.py:STATUSES` | `tests/test_dataset.py::test_every_required_status_is_present` | `python -m pytest -q tests/test_dataset.py` |
| D | Escalated ticket share is 10–30% | PASS → PASS | `dataset.py:validate_dataset` | 50 records; 10 escalated (20.0%); final acceptance transcript | `python -m scripts.final_acceptance` |
| E | Generation is deterministic | PASS → PASS | `dataset.py:generate_support_tickets` | `tests/test_dataset.py::test_dataset_is_deterministic`; committed JSON compared by acceptance runner | `python -m pytest -q tests/test_dataset.py` |
| F | Exactly 12 knowledge-base documents | PASS → PASS | `knowledge_base/manifest.json`, `knowledge_base/loader.py`, 12 policy Markdown files | `tests/test_knowledge_base.py::test_exactly_twelve_documents_load`; final acceptance transcript | `python -m pytest -q tests/test_knowledge_base.py` |
| G | Two independent chunking strategies | PASS → PASS | `rag/chunking.py`, `rag/vector_store.py` | `tests/test_chunking.py`; 14 fixed and 27 sentence chunks in acceptance transcript | `python -m pytest -q tests/test_chunking.py` |
| H | SentenceTransformers embeddings | PASS → PASS | `rag/embeddings.py` | `tests/test_vector_store.py::test_local_embeddings_are_repeatable_and_dimensionally_consistent`; Replit offline smoke: all 384 dimensions finite | `python -m scripts.final_acceptance` |
| I | Separate Chroma collections | PASS → PASS | `rag/vector_store.py` | External local validation supplied with Batch E: 14 fixed / 27 sentence records; Replit transcript marks collection access blocked | `python -m scripts.final_acceptance` |
| J | Chroma upsert and repeatable index build | PASS → PASS | `rag/vector_store.py::_upsert_chunks` | `tests/test_vector_store.py::test_build_indexes_is_separate_idempotent_and_scoped`; external local validation | `python -m pytest -q tests/test_vector_store.py` |
| K | Empirically measured retrieval thresholds | PASS → PASS | `rag/calibration.py`, `rag/config.py`, `data/calibration_results.json` | `tests/test_calibration.py`; locked measured thresholds documented in README; external Chroma measurements | `python -m pytest -q tests/test_calibration.py` |
| L | At least five grounded in-scope examples and one out-of-scope example | PASS → PASS | `rag/grounded_generation.py`, `tests/test_grounded_generation.py` | Five in-scope plus one out-of-scope demonstration; external 15/15 evaluation report summarized in README | `python -m pytest -q tests/test_grounded_generation.py` |
| M | Exact safe out-of-scope fallback | PASS → PASS | `rag/grounded_generation.py`, `evaluation/end_to_end_eval.py` | `tests/test_grounded_generation.py::test_below_threshold_returns_exact_fallback`; `transcripts/end_to_end_evaluation.txt` | `python -m pytest -q tests/test_grounded_generation.py tests/test_end_to_end_eval.py` |
| N | Document-level precision and recall | PASS → PASS | `evaluation/retrieval_eval.py`, `data/retrieval_evaluation_results.json` | `tests/test_retrieval_eval.py`; externally supplied real-Chroma metrics documented in README | `python -m pytest -q tests/test_retrieval_eval.py` |
| O | Dataset-backed ticket lookup | PASS → PASS | `tools/ticket_lookup.py` | `tests/test_ticket_lookup.py`; external real CrewAI ticket lookup result supplied in conversation | `python -m pytest -q tests/test_ticket_lookup.py` |
| P | Deterministic escalation score and rule | PASS → PASS | `tools/ticket_lookup.py`, `evaluation/ticket_distribution.py`, `data/ticket_distribution.json` | `tests/test_ticket_lookup.py`, `tests/test_ticket_distribution.py`; acceptance transcript | `python -m pytest -q tests/test_ticket_lookup.py tests/test_ticket_distribution.py` |
| Q | At least three CrewAI agents | PASS → PASS | `agents/crew.py` | `tests/test_crew.py`; external local Batch D suite 17/17 as supplied | `python -m pytest -q tests/test_batch_d.py` |
| R | Real CrewAI kickoff | PASS → PASS | `agents/crew.py`, `scripts/validate_crewai_runtime.py` | Latest user-supplied local result: final acceptance PASS, 0 blocked, 171 passed / 1 intentional skip; Replit runtime explicitly blocked | `python -m scripts.validate_crewai_runtime` |
| S | Custom `BaseLLM` `MOCK_LLM` | PASS → PASS | `agents/mock_llm.py`, `tools/crewai_compat.py` | `tests/test_mock_llm.py::test_mock_llm_is_zero_key_and_uses_installed_base_when_available` | `python -m pytest -q tests/test_mock_llm.py` |
| T | Zero paid API keys for graded/demo path | PASS → PASS | `agents/mock_llm.py`, `review/autogen_review.py`, `.env.example` | Mock LLM tests and AutoGen replay demo; README names all four unnecessary provider keys | `python -m pytest -q tests/test_mock_llm.py tests/test_batch_d.py` |
| U | Telemetry disabled | PASS → PASS | `agents/__init__.py`, `agents/crew.py` | `tests/test_crew.py::test_telemetry_is_disabled_before_crewai_import` | `python -m pytest -q tests/test_crew.py` |
| V | Session memory and isolation | PASS → PASS | `memory/session_memory.py`, `agents/crew.py`, `api/app.py` | `tests/test_session_memory.py`; external real-runtime follow-up/isolation validation; `transcripts/memory_demo.txt` | `python -m pytest -q tests/test_session_memory.py` |
| W | Pydantic structured output | PASS → PASS | `agents/schemas.py`, `api/schemas.py` | `tests/test_agent_schemas.py`, `tests/test_api.py`; runtime validator checks `SupportResponse` | `python -m pytest -q tests/test_agent_schemas.py tests/test_api.py` |
| X | PII guardrail | PASS → PASS | `guardrails/pii.py`, `api/logging.py` | `tests/test_guardrails.py::test_synthetic_indian_phone_formats_are_masked`; redacted `transcripts/guardrails_demo.txt` | `python -m pytest -q tests/test_guardrails.py` |
| Y | Prompt-injection guardrail | PASS → PASS | `guardrails/injection.py`, `agents/crew.py` | `tests/test_guardrails.py::test_injection_is_blocked_before_any_tool_or_kickoff`; redacted guardrails transcript | `python -m pytest -q tests/test_guardrails.py` |
| Z | Groundedness guardrail | PASS → PASS | `guardrails/groundedness.py`, `agents/crew.py` | `tests/test_guardrails.py` grounded policy/ticket rejection tests; grounded-generation transcript evidence | `python -m pytest -q tests/test_guardrails.py tests/test_grounded_generation.py` |
| AA | Two HTTP endpoints and one WebSocket endpoint | PASS → PASS | `api/app.py`, `api/schemas.py` | `tests/test_api.py` HTTP, ticket, and WebSocket tests | `python -m pytest -q tests/test_api.py` |
| AB | JSONL observability logging | PASS → PASS | `api/logging.py`, `api/service.py` | `tests/test_api.py::test_log_record_jsonl_has_required_fields_and_nonnegative_duration`; sanitized-failure test | `python -m pytest -q tests/test_api.py` |
| AC | Fixed 15-query end-to-end evaluation | PASS → PASS | `evaluation/end_to_end_eval.py`, `data/end_to_end_evaluation_queries.json` | External local result supplied with Batch E: 15 passed, 0 failed, 0 blocked; Replit transcript shows 2 passed / 13 dependency-blocked | `python -m scripts.final_acceptance` |
| AD | AutoGen `RoundRobinGroupChat` | PASS → PASS | `review/autogen_review.py` | `tests/test_batch_d.py::test_autogen_review_uses_two_structured_roles_and_preserves_approved_answer`; AutoGen demo transcript | `python -m pytest -q tests/test_batch_d.py` |
| AE | Exactly two reviewer roles | PASS → PASS | `review/autogen_review.py` | AutoGen role/order assertions in `tests/test_batch_d.py`; `transcripts/autogen_review_demo.txt` | `python -m pytest -q tests/test_batch_d.py` |
| AF | `max_turns=2` | PASS → PASS | `review/autogen_review.py` | AutoGen team turn-count assertions and demo transcript | `python -m pytest -q tests/test_batch_d.py` |
| AG | Approved correct answer returned unchanged | PASS → PASS | `review/autogen_review.py` | Approved-answer case in `tests/test_batch_d.py`; demo transcript says `revised: false` | `python -m scripts.autogen_review_demo` |
| AH | Unsupported answer revised only to evidence | PASS → PASS | `review/autogen_review.py` | Rejection/correction tests in `tests/test_batch_d.py`; demo transcript says `revised: true` | `python -m scripts.autogen_review_demo` |
| AI | Governance and least autonomy | PASS → PASS | `governance/policy.py`, `governance/risk_register.md` | `tests/test_batch_d.py::test_governance_matrix_enforces_least_autonomy_and_medium_risk`; governance transcript | `python -m scripts.governance_demo` |
| AJ | Medium-risk classification | PASS → PASS | `governance/policy.py`, `governance/risk_register.md` | Governance policy test; `transcripts/governance_demo.txt` | `python -m scripts.governance_demo` |
| AK | Token budgets | PASS → PASS | `budgeting/request_budget.py`, `api/service.py` | `tests/test_batch_d.py::test_budget_uses_deterministic_token_cost_and_runtime_estimates`; budget transcript | `python -m scripts.budget_demo` |
| AL | Hypothetical cost accounting | PASS → PASS | `budgeting/request_budget.py` | Budget test verifies deterministic estimate and zero actual paid cost; budget transcript labels rates illustrative | `python -m scripts.budget_demo` |
| AM | Runtime measurement | PASS → PASS | `budgeting/request_budget.py`, `api/service.py` | Budget test checks elapsed time and overage; budget transcript | `python -m pytest -q tests/test_batch_d.py::test_budget_uses_deterministic_token_cost_and_runtime_estimates` |
| AN | Oversized request rejection | PASS → PASS | `api/service.py`, `budgeting/request_budget.py` | `tests/test_batch_d.py::test_oversized_request_is_rejected_before_crew_and_logged_safely` | `python -m pytest -q tests/test_batch_d.py` |
| AO | Normalized response cache | PASS → PASS | `cache/normalized_cache.py`, `api/service.py` | `tests/test_batch_d.py::test_cache_normalizes_safe_queries_and_bypasses_session_or_pii_context`; cache demo transcript | `python -m scripts.cache_demo` |
| AP | Cache hit/miss call-count proof | PASS → PASS | `api/service.py`, `cache/normalized_cache.py` | `tests/test_batch_d.py::test_service_cache_hits_normalized_policy_query_and_misses_different_query` | `python -m pytest -q tests/test_batch_d.py` |
| AQ | Session/PII cache safety | PASS → PASS | `cache/normalized_cache.py`, `api/service.py` | Cache bypass tests and demo cover ticket IDs, follow-ups, PII, and session scope | `python -m pytest -q tests/test_batch_d.py` |
| AR | Public-repo-ready README | PARTIAL → PASS | `README.md`, `.gitignore` | 32 named grader sections, architecture diagram, key/telemetry instructions, external/Replit evidence separation; `tests/test_submission_docs.py` | `python -m pytest -q tests/test_submission_docs.py` |
| AS | Reproducible run, test, evaluation, and submission commands | PARTIAL → PASS | `README.md`, `scripts/final_acceptance.py` | Sequential runner and current Replit transcript; exact GitHub command template in README | `python -m scripts.final_acceptance` |
| AT | Required evidence and transcripts | PARTIAL → PASS | `transcripts/`, `README.md`, `RUBRIC_MAP.md` | Nine required transcript paths verified by docs test and acceptance runner; no raw phone values in transcripts | `python -m pytest -q tests/test_submission_docs.py` |

## Environment notes

- `python -m scripts.final_acceptance` reports actual Replit checks. A blocked
  dependency returns exit code 2; a failed acceptance check returns 1.
- The Replit-safe test command intentionally excludes the two modules that
  require ChromaDB. It does not replace the externally supplied full local run.
- The current Replit run reports the 15-query evaluation and real Chroma/CrewAI
  checks as dependency-blocked; refer to `transcripts/final_acceptance.txt`.
- No Batch A–D application logic, thresholds, dataset, KB, API contract, or
  scoring formula was changed for this final audit.
