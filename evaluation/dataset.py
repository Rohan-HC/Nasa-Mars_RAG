from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

VALID_CATEGORIES = {
    "factual_retrieval",
    "multi_evidence",
    "document_comparison",
    "unanswerable",
    "hallucination_prone",
}
VALID_BEHAVIORS = {"answer", "refuse"}


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    question: str
    category: str
    expected_behavior: str
    required_concepts: tuple[tuple[str, ...], ...]
    source_references: tuple[dict, ...]
    document_hashes: tuple[str, str] | None = None
    verification_note: str = ""

    @property
    def expects_answer(self) -> bool:
        return self.expected_behavior == "answer"


def _validate_case(record: dict, index: int) -> EvaluationCase:
    prefix = f"evaluation case {index}"
    required = {
        "id",
        "question",
        "category",
        "expected_behavior",
        "required_concepts",
        "source_references",
    }
    missing = required - record.keys()
    if missing:
        raise ValueError(f"{prefix} is missing fields: {sorted(missing)}")

    category = record["category"]
    if category not in VALID_CATEGORIES:
        raise ValueError(f"{prefix} has invalid category: {category}")

    behavior = record["expected_behavior"]
    if behavior not in VALID_BEHAVIORS:
        raise ValueError(f"{prefix} has invalid expected behavior: {behavior}")

    concepts = tuple(tuple(group) for group in record["required_concepts"])
    if behavior == "answer" and not concepts:
        raise ValueError(f"{prefix} must define required concepts")

    references = tuple(record["source_references"])
    for reference in references:
        file_hash = reference.get("document_sha256", "")
        pages = reference.get("pages", [])
        if not re.fullmatch(r"[a-f0-9]{64}", file_hash):
            raise ValueError(f"{prefix} has an invalid document SHA-256")
        if not pages or not all(isinstance(page, int) and page > 0 for page in pages):
            raise ValueError(f"{prefix} has invalid source pages")

    hashes = record.get("document_hashes")
    document_hashes = tuple(hashes) if hashes is not None else None
    if document_hashes is not None and len(document_hashes) != 2:
        raise ValueError(f"{prefix} must identify exactly two comparison documents")

    return EvaluationCase(
        case_id=record["id"],
        question=record["question"].strip(),
        category=category,
        expected_behavior=behavior,
        required_concepts=concepts,
        source_references=references,
        document_hashes=document_hashes,
        verification_note=record.get("verification_note", ""),
    )


def load_dataset(path: Path) -> list[EvaluationCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("cases")
    if not isinstance(records, list):
        raise ValueError("Evaluation dataset must contain a cases list")

    cases = [_validate_case(record, index) for index, record in enumerate(records, 1)]
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Evaluation case IDs must be unique")
    return cases
