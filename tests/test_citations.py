from comparison_checks import validate_comparison_response
from eval_answers import check_citations


SOURCES = [
    {"label": "S1", "filename": "crater-floor.pdf"},
    {"label": "S2", "filename": "delta-front.pdf"},
]


def test_comparison_accepts_known_citations_covering_both_documents():
    errors = validate_comparison_response(
        {
            "answer": "Igneous evidence [S1]. Sedimentary evidence [S2].",
            "retrieved": SOURCES,
        }
    )

    assert errors == []


def test_comparison_rejects_unknown_citation():
    errors = validate_comparison_response(
        {
            "answer": "Igneous evidence [S1]. Sedimentary evidence [S99].",
            "retrieved": SOURCES,
        }
    )

    assert "Unknown citation: S99" in errors


def test_answer_citation_check_rejects_nonexistent_source():
    result = {
        "answer": "The rocks are igneous [2].",
        "retrieved": [{"label": 1}],
    }

    assert check_citations(result) is False
