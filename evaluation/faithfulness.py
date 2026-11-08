from __future__ import annotations

import math
import re
from statistics import mean

REFUSAL_MARKERS = (
    "cannot answer",
    "insufficient evidence",
    "not enough evidence",
)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _is_refusal(answer: str) -> bool:
    lowered = answer.casefold()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def parse_answer_claims(answer: str) -> list[dict]:
    """Segment answer sentences and citations; this does not judge support."""
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", answer.strip())
        if sentence.strip()
    ]
    claims = []
    for index, sentence in enumerate(sentences, start=1):
        if _is_refusal(sentence):
            continue
        labels = [int(value) for value in re.findall(r"\[(\d+)\]", sentence)]
        text = re.sub(r"\s*\[\d+\]", "", sentence).strip()
        if text:
            claims.append(
                {
                    "id": f"claim-{index}",
                    "text": text,
                    "citation_labels": labels,
                    "citation_required": True,
                    "judgments": [],
                }
            )
    return claims


def evaluate_claims(record: dict) -> dict:
    evidence_labels = {int(item["label"]) for item in record.get("evidence", [])}
    bracket_contents = re.findall(r"\[([^\]]+)\]", record.get("answer", ""))
    format_valid = all(content.isdigit() for content in bracket_contents)
    answer_labels = {
        int(content) for content in bracket_contents if content.isdigit()
    }
    claims = record.get("claims", [])
    declared_labels = {
        int(label)
        for claim in claims
        for label in claim.get("citation_labels", [])
    }
    format_valid = format_valid and answer_labels == declared_labels
    existence_valid = declared_labels.issubset(evidence_labels)

    evaluable = 0
    supported = 0
    ambiguous = 0
    cited_claims = 0
    adequately_cited_claims = 0
    required_claims = 0
    adequately_covered_claims = 0
    assessments = []

    for claim in claims:
        judgments = claim.get("judgments", [])
        decisive = [
            item for item in judgments if item.get("verdict") in {"supported", "unsupported"}
        ]
        claim_supported = any(
            item.get("verdict") == "supported" for item in decisive
        )
        if decisive:
            evaluable += 1
            supported += int(claim_supported)
        else:
            ambiguous += 1

        labels = [int(label) for label in claim.get("citation_labels", [])]
        by_label = {
            int(item["evidence_label"]): item.get("verdict")
            for item in judgments
            if item.get("evidence_label") is not None
        }
        adequate = bool(labels) and all(
            label in evidence_labels and by_label.get(label) == "supported"
            for label in labels
        )
        if labels:
            cited_claims += 1
            adequately_cited_claims += int(adequate)
        if claim.get("citation_required", True):
            required_claims += 1
            adequately_covered_claims += int(adequate)
        assessments.append(
            {
                "claim_id": claim["id"],
                "text": claim["text"],
                "supported": claim_supported if decisive else None,
                "citation_labels": labels,
                "citation_adequate": adequate,
                "judgments": judgments,
            }
        )

    expects_refusal = record.get("expected_behavior") == "refuse"
    correct_refusal = (
        _is_refusal(record.get("answer", "")) and not claims
        if expects_refusal
        else None
    )
    return {
        "citation_format_valid": format_valid,
        "citation_existence_valid": existence_valid,
        "claim_faithfulness": _ratio(supported, evaluable),
        "citation_precision": _ratio(adequately_cited_claims, cited_claims),
        "citation_coverage": _ratio(
            adequately_covered_claims, required_claims
        ),
        "correct_refusal": correct_refusal,
        "denominators": {
            "evaluable_claims": evaluable,
            "supported_claims": supported,
            "ambiguous_claims": ambiguous,
            "claims_with_citations": cited_claims,
            "adequately_cited_claims": adequately_cited_claims,
            "citation_required_claims": required_claims,
            "adequately_covered_claims": adequately_covered_claims,
        },
        "claim_assessments": assessments,
    }


def retrieval_metrics(
    *,
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> dict[str, float | None]:
    if k < 1:
        raise ValueError("k must be positive")
    if not relevant_ids:
        return {
            "hit_at_k": None,
            "recall_at_k": None,
            "mrr_at_k": None,
            "ndcg_at_k": None,
        }
    ranked = retrieved_ids[:k]
    relevant_ranks = [
        rank for rank, chunk_id in enumerate(ranked, start=1) if chunk_id in relevant_ids
    ]
    hits = len(set(ranked) & relevant_ids)
    dcg = sum(1 / math.log2(rank + 1) for rank in relevant_ranks)
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return {
        "hit_at_k": float(bool(hits)),
        "recall_at_k": hits / len(relevant_ids),
        "mrr_at_k": 1 / relevant_ranks[0] if relevant_ranks else 0.0,
        "ndcg_at_k": dcg / idcg if idcg else None,
    }


def judge_agreement(pairs: list[dict]) -> dict[str, float | int | None]:
    reviewed = [
        pair
        for pair in pairs
        if pair.get("human_verdict") and pair.get("judge_verdict")
    ]
    return {
        "agreement": (
            sum(
                pair["human_verdict"] == pair["judge_verdict"]
                for pair in reviewed
            )
            / len(reviewed)
            if reviewed
            else None
        ),
        "reviewed_pairs": len(reviewed),
    }


def _mean_present(cases: list[dict], key: str) -> float | None:
    values = [case[key] for case in cases if case.get(key) is not None]
    return mean(values) if values else None


def aggregate_quality(cases: list[dict]) -> dict:
    totals = {
        key: sum(case["denominators"][key] for case in cases)
        for key in (
            "evaluable_claims",
            "supported_claims",
            "ambiguous_claims",
            "claims_with_citations",
            "adequately_cited_claims",
            "citation_required_claims",
            "adequately_covered_claims",
        )
    }
    latencies = sorted(float(case["latency_seconds"]) for case in cases)

    def percentile(fraction: float) -> float | None:
        if not latencies:
            return None
        index = max(0, math.ceil(fraction * len(latencies)) - 1)
        return latencies[index]

    refusal_cases = [
        case for case in cases if case.get("correct_refusal") is not None
    ]
    return {
        "case_count": len(cases),
        "retrieval_hit_at_k": _mean_present(cases, "hit_at_k"),
        "retrieval_recall_at_k": _mean_present(cases, "recall_at_k"),
        "retrieval_mrr_at_k": _mean_present(cases, "mrr_at_k"),
        "retrieval_ndcg_at_k": _mean_present(cases, "ndcg_at_k"),
        "answer_correctness": _mean_present(cases, "answer_correct"),
        "citation_format_validity": _mean_present(
            cases, "citation_format_valid"
        ),
        "citation_existence_validity": _mean_present(
            cases, "citation_existence_valid"
        ),
        "claim_faithfulness": _ratio(
            totals["supported_claims"], totals["evaluable_claims"]
        ),
        "citation_precision": _ratio(
            totals["adequately_cited_claims"], totals["claims_with_citations"]
        ),
        "citation_coverage": _ratio(
            totals["adequately_covered_claims"],
            totals["citation_required_claims"],
        ),
        "correct_refusal_rate": (
            mean(case["correct_refusal"] for case in refusal_cases)
            if refusal_cases
            else None
        ),
        "latency_seconds": {
            "mean": mean(latencies) if latencies else None,
            "p50": percentile(0.50),
            "p95": percentile(0.95),
        },
        "denominators": totals,
    }
