import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_SECTIONS = (
    "Project overview",
    "Architecture",
    "Repository structure",
    "Dataset",
    "Knowledge base",
    "RAG pipeline",
    "Chunking strategies",
    "Calibration and thresholds",
    "Retrieval evaluation",
    "Ticket lookup and escalation score",
    "CrewAI agents",
    "MOCK_LLM / offline execution",
    "Session memory",
    "Guardrails",
    "FastAPI endpoints",
    "WebSocket flow",
    "JSONL logging",
    "15-query evaluation",
    "AutoGen review",
    "Governance",
    "Medium-risk rationale",
    "Request budgets",
    "Cache",
    "Security/privacy",
    "Installation",
    "Running the project",
    "Running tests",
    "Running evaluations and demos",
    "Offline grader instructions",
    "Known non-blocking deprecation warnings",
    "Evidence and transcripts",
    "Rubric map and GitHub preparation",
)
REQUIRED_TRANSCRIPTS = (
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
REQUIRED_RUBRIC_IDS = (
    [chr(code) for code in range(ord("A"), ord("Z") + 1)]
    + [f"A{chr(code)}" for code in range(ord("A"), ord("T") + 1)]
)


def test_readme_has_all_32_named_sections_and_architecture_flow():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    sections = re.findall(r"^## \d+\. (.+)$", readme, flags=re.MULTILINE)

    assert tuple(sections) == REQUIRED_SECTIONS
    assert "```mermaid" in readme
    for component in (
        "FastAPI request validation",
        "Request budget",
        "Session-scoped safe cache",
        "Retrieval Agent",
        "Lookup Agent",
        "Response Composer",
        "Policy Compliance Reviewer",
        "Final Editor",
        "Masked JSONL audit log",
        "Cache safety boundary",
    ):
        assert component in readme
    for command in (
        "git status",
        "git add README.md RUBRIC_MAP.md .gitignore .replit",
        "git commit -m",
        "git remote add origin",
        "git push -u origin main",
        "YOUR_GITHUB_USERNAME",
        "ola-domain-support-agent",
    ):
        assert command in readme


def test_rubric_map_covers_all_46_requirements_and_final_statuses():
    rubric = (PROJECT_ROOT / "RUBRIC_MAP.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\|\s*([A-Z]{1,2})\s*\|([^\n]*)$", rubric, re.MULTILINE)

    assert [item[0] for item in rows] == REQUIRED_RUBRIC_IDS
    assert all("PASS" in row for _, row in rows)
    assert "43 PASS, 3 PARTIAL, 0 MISSING" in rubric
    assert "46 PASS" in rubric


def test_required_transcripts_exist_without_raw_phone_numbers():
    transcript_dir = PROJECT_ROOT / "transcripts"
    phone_pattern = re.compile(r"(?<!\d)(?:\+?91[- ]?)?[6-9]\d{4}[- ]?\d{5}(?!\d)")

    for filename in REQUIRED_TRANSCRIPTS:
        transcript = transcript_dir / filename
        assert transcript.is_file()
        assert not phone_pattern.search(transcript.read_text(encoding="utf-8"))

    acceptance_script = (PROJECT_ROOT / "scripts" / "final_acceptance.py").read_text(
        encoding="utf-8"
    )
    acceptance_transcript = (transcript_dir / "final_acceptance.txt").read_text(
        encoding="utf-8"
    )
    assert "Environment: local validation runtime" in acceptance_script
    assert "Environment: local validation runtime" in acceptance_transcript
    assert "Environment: current Replit workspace" not in acceptance_script
    assert "Environment: current Replit workspace" not in acceptance_transcript
