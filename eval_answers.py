from citation_validation import validate_citations
from rag_nasa import answer_question


def check_citations(result: dict) -> bool:
    validation = validate_citations(
        result["answer"],
        result["retrieved"],
        style="numeric",
    )
    return validation["citation_valid"]


if __name__ == "__main__":
    question = "Which minerals formed through aqueous alteration?"

    result = answer_question(question)

    print("Answer:", result["answer"])

    print("Citation integrity:", check_citations(result))
