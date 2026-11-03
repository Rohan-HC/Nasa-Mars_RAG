from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pymupdf

from evaluation.dataset import EvaluationCase, load_dataset
from evaluation.metrics import normalize


def discover_documents(root: Path) -> dict[str, Path]:
    documents: dict[str, Path] = {}
    for path in root.rglob("*.pdf"):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        documents.setdefault(digest, path)
    return documents


def verify_case_sources(
    case: EvaluationCase,
    documents: dict[str, Path],
) -> list[str]:
    if not case.expects_answer:
        return []

    errors = []
    evidence_parts = []
    for reference in case.source_references:
        file_hash = reference["document_sha256"]
        path = documents.get(file_hash)
        if path is None:
            errors.append(f"{case.case_id}: PDF not found for {file_hash}")
            continue

        with pymupdf.open(path) as pdf:
            for page_number in reference["pages"]:
                if page_number > len(pdf):
                    errors.append(
                        f"{case.case_id}: page {page_number} is outside {path.name}"
                    )
                    continue
                page_text = pdf[page_number - 1].get_text("text")
                evidence_parts.append(page_text.replace("-\n", ""))

    evidence = normalize(" ".join(evidence_parts))
    for alternatives in case.required_concepts:
        if not any(normalize(term) in evidence for term in alternatives):
            errors.append(
                f"{case.case_id}: source pages do not contain any of {alternatives}"
            )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify positive benchmark cases against local PDF pages"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluation/datasets/mars_eval_v1.json"),
    )
    parser.add_argument("--document-root", type=Path, default=Path("data"))
    args = parser.parse_args()

    cases = load_dataset(args.dataset)
    documents = discover_documents(args.document_root)
    errors = [
        error
        for case in cases
        for error in verify_case_sources(case, documents)
    ]
    if errors:
        raise SystemExit("\n".join(errors))

    verified = sum(case.expects_answer for case in cases)
    negative = len(cases) - verified
    print(
        f"Verified {verified} positive cases against hashed PDF pages; "
        f"{negative} refusal cases require human absence review."
    )


if __name__ == "__main__":
    main()
