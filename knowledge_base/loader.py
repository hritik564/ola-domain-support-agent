"""Load and validate the capstone support knowledge base."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = KNOWLEDGE_BASE_DIR / "manifest.json"
REQUIRED_DOCUMENT_IDS = tuple(f"KB-{number:03d}" for number in range(1, 13))


def load_knowledge_base(
    manifest_path: str | Path = MANIFEST_PATH,
) -> list[dict[str, str]]:
    """Load manifest entries and their policy text in manifest order."""
    manifest_file = Path(manifest_path).resolve()
    try:
        manifest: Any = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read knowledge-base manifest: {manifest_file}") from error

    if not isinstance(manifest, list):
        raise ValueError("Knowledge-base manifest must contain a JSON array.")
    if len(manifest) != len(REQUIRED_DOCUMENT_IDS):
        raise ValueError(
            f"Knowledge base must contain exactly {len(REQUIRED_DOCUMENT_IDS)} documents."
        )

    document_ids = [
        entry.get("document_id") if isinstance(entry, dict) else None
        for entry in manifest
    ]
    if len(set(document_ids)) != len(document_ids):
        raise ValueError("Knowledge-base manifest contains duplicate document IDs.")
    if set(document_ids) != set(REQUIRED_DOCUMENT_IDS):
        raise ValueError("Knowledge-base manifest must contain KB-001 through KB-012.")

    records: list[dict[str, str]] = []
    for entry in manifest:
        required_fields = {"document_id", "filename", "title", "topic"}
        if not required_fields.issubset(entry):
            raise ValueError(f"Manifest entry is missing fields: {entry!r}")

        filename = entry["filename"]
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"Invalid knowledge-base filename: {filename!r}")

        document_path = manifest_file.parent / filename
        try:
            markdown = document_path.read_text(encoding="utf-8")
        except OSError as error:
            raise ValueError(f"Knowledge-base document does not exist: {filename}") from error

        lines = markdown.splitlines()
        title = entry["title"]
        expected_id = f"document_id: {entry['document_id']}"
        expected_topic = f"topic: {entry['topic']}"
        if (
            not lines
            or lines[0] != f"# {title}"
            or expected_id not in lines
            or expected_topic not in lines
        ):
            raise ValueError(f"Document metadata does not match the manifest: {filename}")

        metadata_indexes = [
            index
            for index, line in enumerate(lines)
            if line.startswith(("document_id:", "topic:"))
        ]
        if len(metadata_indexes) != 2:
            raise ValueError(f"Document must contain one ID and one topic: {filename}")
        policy_text = "\n".join(
            line.strip()
            for line in lines[max(metadata_indexes) + 1 :]
            if line.strip()
        ).strip()
        if not policy_text:
            raise ValueError(f"Knowledge-base document has empty policy text: {filename}")

        records.append(
            {
                "document_id": entry["document_id"],
                "title": title,
                "topic": entry["topic"],
                "filename": filename,
                "text": policy_text,
            }
        )

    return records


def main() -> None:
    """Print loaded knowledge-base records as JSON for inspection."""
    print(json.dumps(load_knowledge_base(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
