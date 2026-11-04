from citation_validation import validate_citations


def validate_comparison(response: dict) -> dict:
    validation = validate_citations(
        response["answer"],
        response["retrieved"],
        style="source",
        require_all_documents=True,
    )
    validation["errors"] = [
        (
            "Answer does not cite both documents"
            if error == "Answer does not cite all compared documents"
            else error
        )
        for error in validation["errors"]
    ]
    validation["passed"] = not validation["errors"]
    validation["citation_valid"] = validation["passed"]
    return validation


def validate_comparison_response(response: dict) -> list[str]:
    return validate_comparison(response)["errors"]
