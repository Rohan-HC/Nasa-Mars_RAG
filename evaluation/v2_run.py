from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

from evaluation.faithfulness import (
    aggregate_quality,
    evaluate_claims,
    judge_agreement,
    retrieval_metrics,
)
from evaluation.semantic_judge import OllamaSemanticJudge
from evaluation.v2_dataset import load_v2_dataset

DEFAULT_DATASET = Path("evaluation/datasets/document_intelligence_eval_v2.json")


def apply_semantic_judge(dataset: dict, judge) -> dict:
    judged_dataset = copy.deepcopy(dataset)
    calibration = []
    for case in judged_dataset["cases"]:
        for claim in case["claims"]:
            human_by_label = {
                int(item["evidence_label"]): item["verdict"]
                for item in claim.get("judgments", [])
                if item.get("evidence_label") is not None
            }
            judgments = []
            for evidence in case["evidence"]:
                result = judge.judge(claim["text"], evidence["text"])
                label = int(evidence["label"])
                judgment = {"evidence_label": label, **result}
                if label in human_by_label:
                    judgment["human_verdict"] = human_by_label[label]
                    calibration.append(
                        {
                            "human_verdict": human_by_label[label],
                            "judge_verdict": result["verdict"],
                        }
                    )
                judgments.append(judgment)
            claim["judgments"] = judgments
    judged_dataset["calibration"] = calibration
    return judged_dataset


def run_v2_evaluation(
    dataset: dict,
    *,
    dataset_path: Path,
    k: int = 4,
    run_mode: str = "offline_human_fixture",
    judge_name: str = "human-reviewed synthetic fixture",
    models: list[str] | None = None,
) -> dict:
    case_results = []
    for case in dataset["cases"]:
        claim_metrics = evaluate_claims(case)
        retrieval = retrieval_metrics(
            retrieved_ids=case["retrieved_chunk_ids"],
            relevant_ids=set(case["relevant_chunk_ids"]),
            k=k,
        )
        case_results.append(
            {
                "id": case["id"],
                "category": case["category"],
                "expected_behavior": case["expected_behavior"],
                "answer_correct": bool(case["reference_answer_correct"]),
                "latency_seconds": float(case.get("latency_seconds", 0.0)),
                **retrieval,
                **claim_metrics,
            }
        )
    raw_dataset = dataset_path.read_bytes()
    payload = {
        "provenance": {
            "dataset": dataset_path.name,
            "dataset_revision": dataset["dataset_version"],
            "dataset_sha256": hashlib.sha256(raw_dataset).hexdigest(),
            "run_mode": run_mode,
            "judge": judge_name,
            "models": models or [],
            "k": k,
            "generated_at": datetime.now(UTC).isoformat(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "timing_note": "Fixture latencies are test inputs, not system performance measurements.",
        },
        "summary": aggregate_quality(case_results),
        "judge_calibration": judge_agreement(dataset.get("calibration", [])),
        "cases": case_results,
    }
    return payload


def build_v2_report(payload: dict) -> str:
    summary = payload["summary"]

    def show(value):
        return "N/A" if value is None else f"{value:.3f}"

    return "\n".join(
        [
            "# Document Intelligence Evaluation V2",
            "",
            f"Run mode: `{payload['provenance']['run_mode']}`",
            f"Dataset revision: `{payload['provenance']['dataset_revision']}`",
            "",
            "This is a deterministic synthetic scorer validation, not a live NASA benchmark.",
            "",
            f"- Cases: {summary['case_count']}",
            f"- Retrieval Recall@K: {show(summary['retrieval_recall_at_k'])}",
            f"- Claim faithfulness: {show(summary['claim_faithfulness'])}",
            f"- Citation precision: {show(summary['citation_precision'])}",
            f"- Citation coverage: {show(summary['citation_coverage'])}",
            f"- Correct refusal rate: {show(summary['correct_refusal_rate'])}",
            "",
            "Ambiguous judgments are excluded from faithfulness denominators and reported in JSON.",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--ollama-judge-model")
    args = parser.parse_args()
    dataset = load_v2_dataset(args.dataset)
    run_options = {}
    if args.ollama_judge_model:
        dataset = apply_semantic_judge(
            dataset,
            OllamaSemanticJudge(args.ollama_judge_model),
        )
        run_options = {
            "run_mode": "local_ollama_semantic_judge",
            "judge_name": f"ollama:{args.ollama_judge_model}",
            "models": [args.ollama_judge_model],
        }
    payload = run_v2_evaluation(
        dataset, dataset_path=args.dataset, k=args.k, **run_options
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        args.output.with_suffix(".md").write_text(
            build_v2_report(payload) + "\n", encoding="utf-8"
        )
    else:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
