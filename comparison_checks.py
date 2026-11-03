import re


def validate_comparison_response(response: dict) -> list[str]:
    answer = response["answer"]
    sources = response["retrieved"]
    errors = []
    source_labels = {source["label"]: source["filename"] for source in sources}
    cited_labels = set()

    for group in re.findall(r"\[([^\[\]]+)\]", answer):
        if not re.fullmatch(
            r"S[1-9]\d*(?:\s*,\s*S[1-9]\d*)*",
            group.strip(),
        ):
            errors.append(f"Invalid citation format: [{group}]")
            continue

        for label in re.findall(r"S[1-9]\d*", group):
            if label not in source_labels:
                errors.append(f"Unknown citation: {label}")
            else:
                cited_labels.add(label)

    cited_documents = {
        source_labels[label]
        for label in cited_labels
    }
    available_documents = set(source_labels.values())

    if len(available_documents) > 1 and cited_documents != available_documents:
        errors.append("Answer does not cite both documents")

    return errors
