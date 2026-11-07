from __future__ import annotations

import math
from statistics import mean
from time import perf_counter


def compare_retrieval_modes(cases, retrieve, *, modes, k: int) -> dict:
    sample_ids = [case["id"] for case in cases]
    output = {"sample_ids": sample_ids, "k": k, "modes": {}}
    for mode in modes:
        recalls = []
        latencies = []
        answer_scores = []
        citation_scores = []
        for case in cases:
            started = perf_counter()
            output_for_case = retrieve(mode, case, k)
            latencies.append((perf_counter() - started) * 1000)
            if isinstance(output_for_case, dict):
                retrieved = output_for_case["retrieved_ids"]
                if output_for_case.get("answer_correct") is not None:
                    answer_scores.append(bool(output_for_case["answer_correct"]))
                if output_for_case.get("citation_supported") is not None:
                    citation_scores.append(bool(output_for_case["citation_supported"]))
            else:
                retrieved = output_for_case
            relevant = set(case["relevant_ids"])
            recalls.append(
                len(set(retrieved[:k]) & relevant) / len(relevant)
                if relevant
                else 0.0
            )
        ordered = sorted(latencies)

        def percentile(fraction):
            return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]

        output["modes"][mode] = {
            "recall_at_k": mean(recalls) if recalls else None,
            "answer_correctness": mean(answer_scores) if answer_scores else None,
            "citation_support": mean(citation_scores) if citation_scores else None,
            "latency_ms": {
                "p50": percentile(0.50) if ordered else None,
                "p95": percentile(0.95) if ordered else None,
            },
        }
    return output
