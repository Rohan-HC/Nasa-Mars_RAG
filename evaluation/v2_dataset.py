from __future__ import annotations

import json
from pathlib import Path


def load_v2_dataset(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("corpus_type") != "synthetic":
        raise ValueError("Committed V2 fixtures must declare a synthetic corpus")
    if not payload.get("dataset_version") or not payload.get("cases"):
        raise ValueError("V2 dataset requires a version and cases")
    required = {
        "id",
        "category",
        "question",
        "expected_behavior",
        "answer",
        "evidence",
        "claims",
        "relevant_chunk_ids",
        "retrieved_chunk_ids",
        "claims_reviewed_by",
    }
    for case in payload["cases"]:
        missing = required - set(case)
        if missing:
            raise ValueError(f"Case {case.get('id', '<unknown>')} lacks {sorted(missing)}")
    return payload
