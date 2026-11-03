from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from document_comparison import compare_documents
from evaluation.dataset import EvaluationCase, load_dataset
from evaluation.metrics import evaluate_response, summarize
from evaluation.report import build_report
from rag_nasa import answer_question


def _source_summary(source: dict) -> dict:
    public_fields = ("label", "title", "filename", "sha256", "page", "distance")
    return {key: source[key] for key in public_fields if key in source}


def live_response(case: EvaluationCase) -> dict:
    if case.document_hashes:
        return compare_documents(
            question=case.question,
            document_a=case.document_hashes[0],
            document_b=case.document_hashes[1],
        )
    return answer_question(case.question)


def run_evaluation(
    cases: list[EvaluationCase],
    *,
    dataset_name: str,
    response_function=live_response,
    execution_mode: str = "live",
    k: int = 4,
) -> dict:
    results = []
    for case in cases:
        started = perf_counter()
        error = None
        try:
            response = response_function(case)
        except Exception as exc:  # Keep partial benchmark results for diagnosis.
            response = {"answer": "", "retrieved": []}
            error = f"{type(exc).__name__}: {exc}"
        latency = perf_counter() - started
        metrics = evaluate_response(case, response, latency_seconds=latency, k=k)
        results.append(
            {
                "id": case.case_id,
                "question": case.question,
                "category": case.category,
                "expected_behavior": case.expected_behavior,
                "answer": response.get("answer", ""),
                "retrieved": [
                    _source_summary(source)
                    for source in response.get("retrieved", [])
                ],
                "metrics": metrics,
                "error": error,
            }
        )
        print(
            f"[{len(results)}/{len(cases)}] {case.case_id}: "
            f"{'error' if error else 'complete'} ({latency:.2f}s)",
            flush=True,
        )

    return {
        "metadata": {
            "dataset": dataset_name,
            "execution_mode": execution_mode,
            "case_count": len(cases),
            "k": k,
            "generated_at_utc": datetime.now(UTC).isoformat(),
        },
        "summary": summarize(results),
        "cases": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the NASA Mars RAG benchmark")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluation/datasets/mars_eval_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/results/baseline.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("evaluation/results/baseline.md"),
    )
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be at least 1")

    cases = load_dataset(args.dataset)
    payload = run_evaluation(cases, dataset_name=str(args.dataset), k=args.k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = build_report(payload)
    args.report.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
