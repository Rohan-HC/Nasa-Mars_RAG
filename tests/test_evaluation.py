import json
from pathlib import Path

from evaluation.dataset import EvaluationCase, load_dataset
from evaluation.metrics import evaluate_response
from evaluation.report import build_report
from evaluation.run import run_evaluation
from evaluation.verify_sources import verify_case_sources


DATASET = Path("evaluation/datasets/mars_eval_v1.json")


def test_dataset_contains_twenty_source_referenced_cases():
    cases = load_dataset(DATASET)

    assert len(cases) == 20
    assert {
        "factual_retrieval",
        "multi_evidence",
        "document_comparison",
        "unanswerable",
        "hallucination_prone",
    } == {case.category for case in cases}
    assert all(case.source_references for case in cases if case.expects_answer)


def test_answer_metrics_keep_correctness_and_citation_validity_separate():
    case = EvaluationCase(
        case_id="fixture-1",
        question="Which mineral was reported?",
        category="factual_retrieval",
        expected_behavior="answer",
        required_concepts=(("olivine",),),
        source_references=(
            {
                "document_sha256": "a" * 64,
                "filename": "fixture.pdf",
                "pages": [2],
            },
        ),
    )
    result = {
        "answer": "Olivine was reported [2].",
        "retrieved": [
            {
                "label": 1,
                "title": "fixture.pdf",
                "sha256": "a" * 64,
                "page": 2,
                "text": "The sample contained olivine.",
            }
        ],
    }

    metrics = evaluate_response(case, result, latency_seconds=0.25, k=3)

    assert metrics["answer_correct"] is True
    assert metrics["citation_valid"] is False
    assert metrics["evidence_supported"] is False
    assert metrics["retrieval_hit_at_k"] is True
    assert metrics["latency_seconds"] == 0.25


def test_refusal_metric_accepts_configured_insufficient_evidence_response():
    case = EvaluationCase(
        case_id="fixture-2",
        question="What is not in the corpus?",
        category="unanswerable",
        expected_behavior="refuse",
        required_concepts=(),
        source_references=(),
    )
    result = {
        "answer": "I cannot answer this from the available NASA documents.",
        "retrieved": [],
    }

    metrics = evaluate_response(case, result, latency_seconds=0.1, k=3)

    assert metrics["correct_refusal"] is True
    assert metrics["answer_correct"] is True
    assert metrics["citation_valid"] is True


def test_report_labels_fixture_results_without_presenting_them_as_live_scores():
    payload = {
        "metadata": {
            "dataset": "fixture.json",
            "execution_mode": "fixture",
            "case_count": 1,
        },
        "summary": {
            "retrieval_hit_at_k": 1.0,
            "answer_correctness": 1.0,
            "citation_validity": 1.0,
            "evidence_support": 1.0,
            "correct_refusal": None,
            "mean_latency_seconds": 0.1,
        },
        "cases": [],
    }

    report = build_report(payload)

    assert "Execution mode: `fixture`" in report
    assert "not a live NASA RAG benchmark" in report


def test_committed_results_are_machine_readable_when_present():
    result_path = Path("evaluation/results/baseline.json")
    if not result_path.exists():
        return

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["execution_mode"] in {"live", "fixture"}
    assert payload["metadata"]["case_count"] == len(payload["cases"])


def test_saved_results_omit_full_source_passage_text():
    case = EvaluationCase(
        case_id="fixture-3",
        question="Which mineral was reported?",
        category="factual_retrieval",
        expected_behavior="answer",
        required_concepts=(("olivine",),),
        source_references=(
            {
                "document_sha256": "a" * 64,
                "filename": "fixture.pdf",
                "pages": [2],
            },
        ),
    )

    payload = run_evaluation(
        [case],
        dataset_name="fixture.json",
        execution_mode="fixture",
        response_function=lambda _: {
            "answer": "Olivine was reported [1].",
            "retrieved": [
                {
                    "label": 1,
                    "title": "fixture.pdf",
                    "page": 2,
                    "text": "The sample contained olivine.",
                }
            ],
        },
    )

    assert payload["cases"][0]["retrieved"] == [
        {"label": 1, "title": "fixture.pdf", "page": 2}
    ]


def test_source_verifier_checks_concepts_on_declared_pdf_pages(pdf_file):
    case = EvaluationCase(
        case_id="fixture-4",
        question="What was observed?",
        category="factual_retrieval",
        expected_behavior="answer",
        required_concepts=(("Jezero crater",), ("layered rocks",)),
        source_references=(
            {
                "document_sha256": "a" * 64,
                "filename": "fixture.pdf",
                "pages": [1],
            },
        ),
    )

    assert verify_case_sources(case, {"a" * 64: pdf_file}) == []
