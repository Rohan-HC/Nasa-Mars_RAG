import re


REFUSAL_PHRASES = (
    "i cannot answer this from the available nasa documents",
    "insufficient evidence for comparison",
    "insufficient evidence for a reliable comparison",
)


def is_refusal(answer: str) -> bool:
    normalized = " ".join(answer.casefold().split())
    return normalized.removesuffix(".") in REFUSAL_PHRASES


def _document_identity(source: dict) -> str:
    file_hash = str(source.get("sha256", "")).casefold()
    if re.fullmatch(r"[a-f0-9]{64}", file_hash):
        return f"sha256:{file_hash}"
    filename = str(source.get("filename") or source.get("title") or "")
    return f"filename:{filename.casefold()}"


def validate_citations(
    answer: str,
    sources: list[dict],
    *,
    style: str,
    require_all_documents: bool = False,
) -> dict:
    if style not in {"numeric", "source"}:
        raise ValueError("Citation style must be 'numeric' or 'source'")

    errors = []
    cited_labels: set[str] = set()
    available_labels = {str(source["label"]) for source in sources}
    groups = re.findall(r"\[([^\[\]]+)\]", answer)
    label_pattern = r"[1-9]\d*" if style == "numeric" else r"S[1-9]\d*"
    group_pattern = rf"{label_pattern}(?:\s*,\s*{label_pattern})*"

    for group in groups:
        cleaned = group.strip()
        if not re.fullmatch(group_pattern, cleaned):
            errors.append(f"Invalid citation format: [{group}]")
            continue

        for label in re.findall(label_pattern, cleaned):
            if label not in available_labels:
                errors.append(f"Unknown citation: {label}")
            else:
                cited_labels.add(label)

    refusal = is_refusal(answer)
    if answer.strip() and not refusal and not cited_labels:
        errors.append("Substantive answer is missing citations")

    if require_all_documents and not refusal:
        source_documents = {_document_identity(source) for source in sources}
        source_by_label = {
            str(source["label"]): _document_identity(source)
            for source in sources
        }
        cited_documents = {
            source_by_label[label]
            for label in cited_labels
            if label in source_by_label
        }
        if len(source_documents) > 1 and cited_documents != source_documents:
            errors.append("Answer does not cite all compared documents")

    factual_grounding = (
        {"status": "not_applicable", "reason": "The response is a refusal."}
        if refusal
        else {
            "status": "not_verified",
            "reason": "Valid citation labels do not prove that claims are supported.",
        }
    )
    citation_valid = not errors
    return {
        "passed": citation_valid,
        "errors": errors,
        "citation_valid": citation_valid,
        "factual_grounding": factual_grounding,
    }
