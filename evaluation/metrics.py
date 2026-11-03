from __future__ import annotations

import re
from statistics import fmean

from evaluation.dataset import EvaluationCase


REFUSAL_TEXT = "i cannot answer this from the available nasa documents"


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _concept_score(concepts: tuple[tuple[str, ...], ...], text: str) -> float:
    if not concepts:
        return 1.0
    normalized = normalize(text)
    matched = sum(
        any(normalize(alternative) in normalized for alternative in alternatives)
        for alternatives in concepts
    )
    return matched / len(concepts)


def _source_label(source: dict) -> str:
    return str(source.get("label", ""))


def _citation_labels(answer: str) -> tuple[set[str], bool]:
    labels: set[str] = set()
    valid_format = True
    for group in re.findall(r"\[([^\[\]]+)\]", answer):
        cleaned = group.strip()
        if re.fullmatch(r"[1-9]\d*(?:\s*,\s*[1-9]\d*)*", cleaned):
            labels.update(re.findall(r"[1-9]\d*", cleaned))
        elif re.fullmatch(r"S[1-9]\d*(?:\s*,\s*S[1-9]\d*)*", cleaned):
            labels.update(re.findall(r"S[1-9]\d*", cleaned))
        else:
            valid_format = False
    return labels, valid_format


def _reference_matches_source(reference: dict, source: dict) -> bool:
    source_hash = source.get("sha256")
    source_filename = source.get("filename") or source.get("title")
    same_document = (
        source_hash == reference["document_sha256"]
        if source_hash
        else source_filename == reference.get("filename")
    )
    return same_document and source.get("page") in reference["pages"]


def evaluate_response(
    case: EvaluationCase,
    result: dict,
    *,
    latency_seconds: float,
    k: int,
) -> dict:
    answer = str(result.get("answer", ""))
    sources = list(result.get("retrieved", []))[:k]
    cited_labels, citation_format_valid = _citation_labels(answer)
    available_labels = {_source_label(source) for source in sources}

    if case.source_references:
        matched_references = sum(
            any(_reference_matches_source(reference, source) for source in sources)
            for reference in case.source_references
        )
        source_recall = matched_references / len(case.source_references)
        retrieval_hit = matched_references > 0
    else:
        source_recall = None
        retrieval_hit = None

    correct_refusal = REFUSAL_TEXT in normalize(answer)
    concept_score = _concept_score(case.required_concepts, answer)
    answer_correct = concept_score == 1.0 if case.expects_answer else correct_refusal

    requires_citation = case.expects_answer and bool(answer.strip())
    citation_valid = (
        citation_format_valid
        and cited_labels.issubset(available_labels)
        and (bool(cited_labels) if requires_citation else True)
    )

    if case.expects_answer:
        cited_text = " ".join(
            str(source.get("text", ""))
            for source in sources
            if _source_label(source) in cited_labels
        )
        evidence_score = _concept_score(case.required_concepts, cited_text)
        evidence_supported = citation_valid and evidence_score == 1.0
    else:
        evidence_score = None
        evidence_supported = None

    return {
        "retrieval_hit_at_k": retrieval_hit,
        "source_recall_at_k": source_recall,
        "answer_concept_score": concept_score,
        "answer_correct": answer_correct,
        "citation_valid": citation_valid,
        "evidence_support_score": evidence_score,
        "evidence_supported": evidence_supported,
        "correct_refusal": correct_refusal if not case.expects_answer else None,
        "latency_seconds": round(latency_seconds, 4),
    }


def summarize(case_results: list[dict]) -> dict:
    def mean_metric(name: str) -> float | None:
        values = [case["metrics"][name] for case in case_results]
        available = [float(value) for value in values if value is not None]
        return round(fmean(available), 4) if available else None

    return {
        "retrieval_hit_at_k": mean_metric("retrieval_hit_at_k"),
        "source_recall_at_k": mean_metric("source_recall_at_k"),
        "answer_correctness": mean_metric("answer_correct"),
        "citation_validity": mean_metric("citation_valid"),
        "evidence_support": mean_metric("evidence_supported"),
        "correct_refusal": mean_metric("correct_refusal"),
        "mean_latency_seconds": mean_metric("latency_seconds"),
    }
