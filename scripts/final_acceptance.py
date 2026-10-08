"""Run capstone checks and report blocked integrations honestly.

Refreshes evaluation and demo transcripts, but does not change application
logic or treat a missing Chroma/CrewAI runtime as a passing integration.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPT_PATH = PROJECT_ROOT / "transcripts" / "final_acceptance.txt"
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import SUPPORT_TICKETS, generate_support_tickets, validate_dataset
from knowledge_base.loader import load_knowledge_base
from rag.chunking import chunk_by_sentence, chunk_fixed_size


def _summary_line(output: str) -> str:
    """Keep test command output concise and free of individual request payloads."""
    for line in reversed(output.splitlines()):
        if re.search(
            r"\b\d+ (?:passed|failed|error|errors|skipped|warnings?)\b",
            line,
        ):
            return line.strip()
    return "No pytest summary line was emitted."


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )


def _write_transcript(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents.rstrip() + "\n", encoding="utf-8")


class AcceptanceRun:
    def __init__(self) -> None:
        self.lines = [
            "Ola Domain Support Agent — final acceptance",
            "Environment: local validation runtime",
            "Secrets, prompts, raw queries, and raw PII are not printed.",
            "",
        ]
        self.statuses: list[str] = []

    def record(self, name: str, status: str, details: str) -> None:
        self.statuses.append(status)
        self.lines.append(f"{status}: {name} — {details}")

    def run(self) -> int:
        documents: list[dict[str, str]] = []
        fixed_chunks: list[dict[str, str | int]] = []
        sentence_chunks: list[dict[str, str | int]] = []

        try:
            generated = generate_support_tickets()
            report = validate_dataset(generated)
            if generated != SUPPORT_TICKETS:
                raise AssertionError("generated records differ from SUPPORT_TICKETS")
            data_path = PROJECT_ROOT / "data" / "support_tickets.json"
            if json.loads(data_path.read_text(encoding="utf-8")) != generated:
                raise AssertionError("stored dataset differs from deterministic output")
            self.record(
                "Dataset distribution",
                "PASS",
                (
                    f"{report['total_records']} deterministic tickets; "
                    f"escalated={report['escalated_count']} "
                    f"({report['escalated_percentage']:.1f}%); required category "
                    "and status checks passed."
                ),
            )
        except Exception as exc:
            self.record(
                "Dataset distribution",
                "FAIL",
                f"{type(exc).__name__}; details omitted from the transcript.",
            )

        try:
            documents = load_knowledge_base()
            fixed_chunks = chunk_fixed_size(documents)
            sentence_chunks = chunk_by_sentence(documents)
            if len(documents) != 12:
                raise AssertionError(f"expected 12 documents, found {len(documents)}")
            if (len(fixed_chunks), len(sentence_chunks)) != (14, 27):
                raise AssertionError(
                    "deterministic chunk counts changed from the locked "
                    "14 fixed / 27 sentence measurements"
                )
            self.record(
                "Knowledge base and chunk generation",
                "PASS",
                (
                    f"{len(documents)} validated documents; "
                    f"{len(fixed_chunks)} fixed chunks and "
                    f"{len(sentence_chunks)} sentence chunks."
                ),
            )
        except Exception as exc:
            self.record(
                "Knowledge base and chunk generation",
                "FAIL",
                f"{type(exc).__name__}; details omitted from the transcript.",
            )

        if importlib.util.find_spec("sentence_transformers") is None:
            self.record(
                "Local SentenceTransformers embedding",
                "BLOCKED",
                "sentence-transformers is not installed.",
            )
        else:
            try:
                import math

                from rag.embeddings import EMBEDDING_MODEL_NAME, LocalEmbeddingService

                vector = LocalEmbeddingService().embed_query(
                    "A support-policy embedding smoke check."
                )
                if not vector or not all(
                    math.isfinite(float(value)) for value in vector
                ):
                    raise AssertionError("embedding was empty or non-finite")
                self.record(
                    "Local SentenceTransformers embedding",
                    "PASS",
                    f"{EMBEDDING_MODEL_NAME}; dimension={len(vector)}; finite=true.",
                )
            except Exception as exc:
                self.record(
                    "Local SentenceTransformers embedding",
                    "FAIL",
                    f"{type(exc).__name__}; details omitted from the transcript.",
                )

        chroma_available = importlib.util.find_spec("chromadb") is not None
        if not chroma_available:
            self.record(
                "Chroma index counts",
                "BLOCKED",
                (
                    "chromadb is not installed; deterministic source chunk counts "
                    f"are fixed={len(fixed_chunks)}, sentence={len(sentence_chunks)}, "
                    "but persistent collection counts cannot be queried."
                ),
            )
        else:
            try:
                from rag.vector_store import build_indexes

                index_stats = build_indexes()
                fixed_count = index_stats["fixed"]["stored_records"]
                sentence_count = index_stats["sentence"]["stored_records"]
                if (fixed_count, sentence_count) != (14, 27):
                    raise AssertionError(
                        "stored collection counts did not match 14 / 27"
                    )
                self.record(
                    "Chroma index counts",
                    "PASS",
                    (
                        f"{index_stats['fixed']['collection']}={fixed_count}; "
                        f"{index_stats['sentence']['collection']}={sentence_count}."
                    ),
                )
            except Exception as exc:
                self.record(
                    "Chroma index counts",
                    "FAIL",
                    f"{type(exc).__name__}; details omitted from the transcript.",
                )

        if not chroma_available:
            self.record(
                "Retrieval calibration and evaluation",
                "BLOCKED",
                "requires a real ChromaDB collection; no stored scores were fabricated.",
            )
        else:
            try:
                from evaluation.retrieval_eval import evaluate_retrieval
                from rag.calibration import CALIBRATION_STRATEGIES, run_calibration

                calibration = run_calibration()
                for strategy in CALIBRATION_STRATEGIES:
                    if (
                        calibration["strategies"][strategy]["threshold"]["status"]
                        != "separated"
                    ):
                        raise AssertionError(
                            f"{strategy} calibration was not separated"
                        )
                retrieval = evaluate_retrieval()
                recommendation = retrieval["production_strategy_recommendation"]
                if (
                    retrieval["status"] != "completed"
                    or recommendation["status"] != "recommended"
                ):
                    raise AssertionError(
                        "retrieval evaluation did not return a completed recommendation"
                    )
                self.record(
                    "Retrieval calibration and evaluation",
                    "PASS",
                    (
                        "both strategy thresholds separated; "
                        f"{retrieval['query_count']} queries evaluated; "
                        f"recommended={recommendation['strategy']}."
                    ),
                )
            except Exception as exc:
                self.record(
                    "Retrieval calibration and evaluation",
                    "FAIL",
                    f"{type(exc).__name__}; details omitted from the transcript.",
                )

        crewai_available = importlib.util.find_spec("crewai") is not None
        if not crewai_available or not chroma_available:
            missing = [
                name
                for name, available in (
                    ("crewai", crewai_available),
                    ("chromadb", chroma_available),
                )
                if not available
            ]
            self.record(
                "CrewAI runtime validation",
                "BLOCKED",
                f"real runtime dependency unavailable: {', '.join(missing)}.",
            )
        else:
            try:
                from scripts.validate_crewai_runtime import main as validate_crew

                result = validate_crew()
                if result != 0:
                    raise AssertionError("real CrewAI runtime validator returned failure")
                self.record(
                    "CrewAI runtime validation",
                    "PASS",
                    "real policy, ticket, same-session follow-up, and isolation checks passed.",
                )
            except Exception as exc:
                self.record(
                    "CrewAI runtime validation",
                    "FAIL",
                    f"{type(exc).__name__}; details omitted from the transcript.",
                )

        try:
            from evaluation.end_to_end_eval import evaluate, write_results

            evaluation = evaluate()
            write_results(evaluation)
            details = (
                f"{evaluation['case_count']} cases; passed={evaluation['pass_count']}, "
                f"failed={evaluation['fail_count']}, "
                f"dependency-blocked={evaluation['blocked_count']}."
            )
            if evaluation["fail_count"]:
                self.record("15-query end-to-end evaluation", "FAIL", details)
            elif evaluation["blocked_count"]:
                self.record("15-query end-to-end evaluation", "BLOCKED", details)
            else:
                self.record("15-query end-to-end evaluation", "PASS", details)
        except Exception as exc:
            self.record(
                "15-query end-to-end evaluation",
                "FAIL",
                f"{type(exc).__name__}; details omitted from the transcript.",
            )

        self._run_demos()
        self._check_submission_docs()
        self._run_full_pytest()
        self._run_safe_pytest()
        self._run_dependency_check()

        failures = self.statuses.count("FAIL")
        blocked = self.statuses.count("BLOCKED")
        if failures:
            exit_code = 1
            overall = "FAIL"
        elif blocked:
            exit_code = 2
            overall = "BLOCKED"
        else:
            exit_code = 0
            overall = "PASS"
        self.lines.extend(
            (
                "",
                f"Overall: {overall}",
                f"Failed checks: {failures}",
                f"Dependency-blocked checks: {blocked}",
                f"Runner exit code: {exit_code}",
            )
        )
        return exit_code

    def _run_demos(self) -> None:
        try:
            import anyio

            from scripts.autogen_review_demo import _render_transcript

            autogen_text = anyio.run(_render_transcript)
            _write_transcript(
                PROJECT_ROOT / "transcripts" / "autogen_review_demo.txt",
                autogen_text,
            )
            required = ("approved: true", "revised: false", "approved: false", "revised: true")
            if not all(marker in autogen_text for marker in required):
                raise AssertionError("AutoGen review behavior markers were missing")
            self.record(
                "AutoGen review demo",
                "PASS",
                "offline two-role review preserved an approved answer and corrected an unsupported draft.",
            )
        except Exception as exc:
            self.record(
                "AutoGen review demo",
                "FAIL",
                f"{type(exc).__name__}; details omitted from the transcript.",
            )

        for name, module_name, expected in (
            (
                "Cache demo",
                "scripts.cache_demo",
                ("hit=true", "underlying deterministic lookup calls: 2", "pii request bypass: true"),
            ),
            (
                "Governance demo",
                "scripts.governance_demo",
                ("risk classification: medium", "least-autonomy check: passed"),
            ),
            (
                "Budget demo",
                "scripts.budget_demo",
                ("oversized request", "actual_paid_cost_usd: 0", "runtime observation"),
            ),
        ):
            try:
                module = __import__(module_name, fromlist=["render_demo"])
                text = module.render_demo()
                transcript_name = {
                    "Cache demo": "cache_demo.txt",
                    "Governance demo": "governance_demo.txt",
                    "Budget demo": "budget_demo.txt",
                }[name]
                _write_transcript(PROJECT_ROOT / "transcripts" / transcript_name, text)
                if not all(marker in text.casefold() for marker in expected):
                    raise AssertionError("required demo evidence markers were missing")
                self.record(name, "PASS", "deterministic local demo completed and transcript refreshed.")
            except Exception as exc:
                self.record(
                    name,
                    "FAIL",
                    f"{type(exc).__name__}; details omitted from the transcript.",
                )

    def _check_submission_docs(self) -> None:
        try:
            readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
            rubric_map = (PROJECT_ROOT / "RUBRIC_MAP.md").read_text(
                encoding="utf-8"
            )
            expected_ids = [
                *[chr(code) for code in range(ord("A"), ord("Z") + 1)],
                *[f"A{chr(code)}" for code in range(ord("A"), ord("T") + 1)],
            ]
            actual_ids = re.findall(r"^\|\s*([A-Z]{1,2})\s*\|", rubric_map, re.M)
            required_transcripts = (
                "crew_demo.txt",
                "memory_demo.txt",
                "guardrails_demo.txt",
                "end_to_end_evaluation.txt",
                "autogen_review_demo.txt",
                "cache_demo.txt",
                "governance_demo.txt",
                "budget_demo.txt",
                "final_acceptance.txt",
            )
            missing_transcripts = [
                name
                for name in required_transcripts
                if not (PROJECT_ROOT / "transcripts" / name).is_file()
            ]
            numbered_sections = re.findall(r"^## (\d+)\. ", readme, re.M)
            if numbered_sections != [str(number) for number in range(1, 33)]:
                raise AssertionError("README must contain all 32 numbered sections")
            if actual_ids != expected_ids:
                raise AssertionError("rubric map must list every A–AT requirement once")
            if missing_transcripts:
                raise AssertionError(
                    "required transcripts missing: " + ", ".join(missing_transcripts)
                )
            self.record(
                "Submission documentation and evidence",
                "PASS",
                "32 README sections, 46 rubric rows, and all nine required transcripts exist.",
            )
        except Exception as exc:
            self.record(
                "Submission documentation and evidence",
                "FAIL",
                f"{type(exc).__name__}; details omitted from the transcript.",
            )

    def _run_full_pytest(self) -> None:
        completed = _run_command([sys.executable, "-m", "pytest", "-q"])
        output = completed.stdout + "\n" + completed.stderr
        summary = _summary_line(output)
        missing_chroma_collection = (
            "ModuleNotFoundError: No module named 'chromadb'" in output
            or "ImportError while importing test module" in output
            and "chromadb" in output.casefold()
        )
        if completed.returncode == 0:
            self.record("Full pytest suite", "PASS", summary)
        elif missing_chroma_collection:
            self.record(
                "Full pytest suite",
                "BLOCKED",
                f"ChromaDB-dependent collection/execution unavailable; {summary}",
            )
        else:
            self.record(
                "Full pytest suite",
                "FAIL",
                f"pytest exited {completed.returncode}; {summary}",
            )

    def _run_safe_pytest(self) -> None:
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--ignore=tests/test_vector_store.py",
            "--ignore=tests/test_retrieval.py",
            "-rs",
        ]
        completed = _run_command(command)
        summary = _summary_line(completed.stdout + "\n" + completed.stderr)
        if completed.returncode == 0:
            self.record(
                "Replit-safe pytest suite",
                "PASS",
                f"{summary}; Chroma-dependent modules excluded explicitly.",
            )
        else:
            self.record(
                "Replit-safe pytest suite",
                "FAIL",
                f"pytest exited {completed.returncode}; {summary}",
            )

    def _run_dependency_check(self) -> None:
        completed = _run_command([sys.executable, "-m", "pip", "check"])
        output = (completed.stdout + "\n" + completed.stderr).strip()
        if completed.returncode == 0:
            self.record("Installed dependency consistency", "PASS", output or "No broken requirements found.")
        else:
            self.record(
                "Installed dependency consistency",
                "FAIL",
                f"pip check exited {completed.returncode}; details omitted.",
            )


def main() -> int:
    runner = AcceptanceRun()
    exit_code = runner.run()
    transcript = "\n".join(runner.lines).rstrip() + "\n"
    _write_transcript(TRANSCRIPT_PATH, transcript)
    print(transcript, end="")
    print(f"Saved: {TRANSCRIPT_PATH.relative_to(PROJECT_ROOT)}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
