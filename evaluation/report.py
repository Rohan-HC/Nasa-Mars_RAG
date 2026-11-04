from __future__ import annotations


def _format_metric(value: float | None, *, seconds: bool = False) -> str:
    if value is None:
        return "n/a"
    if seconds:
        return f"{value:.3f} s"
    return f"{value:.1%}"


def build_report(payload: dict) -> str:
    metadata = payload["metadata"]
    summary = payload["summary"]
    mode = metadata["execution_mode"]
    lines = [
        "# NASA Mars RAG benchmark",
        "",
        f"- Dataset: `{metadata['dataset']}`",
        f"- Execution mode: `{mode}`",
        f"- Cases: {metadata['case_count']}",
        f"- Retrieval K: {metadata.get('k', 'n/a')}",
    ]
    if metadata.get("postprocessing"):
        lines.append(f"- Postprocessing: {metadata['postprocessing']}")
    lines.append("")
    if mode != "live":
        lines.extend(
            [
                "> Fixture execution verifies the scoring pipeline; it is not a live NASA RAG benchmark.",
                "",
            ]
        )
    lines.extend(
        [
            "| Metric | Result |",
            "|---|---:|",
            f"| Retrieval Hit@K | {_format_metric(summary['retrieval_hit_at_k'])} |",
            f"| Source Recall@K | {_format_metric(summary.get('source_recall_at_k'))} |",
            f"| Answer correctness | {_format_metric(summary['answer_correctness'])} |",
            f"| Citation validity | {_format_metric(summary['citation_validity'])} |",
            f"| Evidence-supported answers | {_format_metric(summary['evidence_support'])} |",
            f"| Correct refusal | {_format_metric(summary['correct_refusal'])} |",
            f"| Mean response latency | {_format_metric(summary['mean_latency_seconds'], seconds=True)} |",
            "",
            "## Interpretation",
            "",
            "Source recall counts each verified document-page pair. Answer correctness is deterministic concept coverage, not an LLM judge. Citation validity checks syntax and whether labels exist; it does not prove that a cited passage supports a claim. Evidence support checks whether required benchmark concepts occur in the cited passages and remains a lexical proxy rather than full factual verification.",
            "",
            "## Case results",
            "",
            "| ID | Category | Correct | Citation | Evidence | Latency |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for case in payload.get("cases", []):
        metrics = case["metrics"]
        lines.append(
            "| {id} | {category} | {correct} | {citation} | {evidence} | {latency:.3f} s |".format(
                id=case["id"],
                category=case["category"],
                correct="yes" if metrics["answer_correct"] else "no",
                citation="yes" if metrics["citation_valid"] else "no",
                evidence=(
                    "n/a"
                    if metrics["evidence_supported"] is None
                    else "yes" if metrics["evidence_supported"] else "no"
                ),
                latency=metrics["latency_seconds"],
            )
        )
    lines.append("")
    return "\n".join(lines)
