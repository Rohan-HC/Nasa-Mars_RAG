import re

from rag_nasa import answer_question


def check_citations(result: dict) -> bool:
    answer = result["answer"]
    sources = result["retrieved"]

    cited_ids = {int(number) for number in re.findall(r"\[(\d+)\]", answer)}

    available_ids = {source["label"] for source in sources}

    return cited_ids.issubset(available_ids)


if __name__ == "__main__":
    question = "Which minerals formed through aqueous alteration?"

    result = answer_question(question)

    print("Answer:", result["answer"])

    print("Citation integrity:", check_citations(result))
