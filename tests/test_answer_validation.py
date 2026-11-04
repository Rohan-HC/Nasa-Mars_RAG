from citation_validation import validate_citations


NUMERIC_SOURCES = [
    {"label": 1, "title": "mars.pdf", "page": 1, "text": "Olivine."},
    {"label": 2, "title": "mars.pdf", "page": 2, "text": "Carbonate."},
]


def test_substantive_answer_without_citation_is_invalid():
    validation = validate_citations(
        "The formation contains olivine.",
        NUMERIC_SOURCES,
        style="numeric",
    )

    assert validation["citation_valid"] is False
    assert "Substantive answer is missing citations" in validation["errors"]


def test_canonical_refusal_does_not_require_citation():
    validation = validate_citations(
        "I cannot answer this from the available NASA documents.",
        [],
        style="numeric",
    )

    assert validation["citation_valid"] is True
    assert validation["factual_grounding"]["status"] == "not_applicable"


def test_unknown_numeric_citation_is_invalid():
    validation = validate_citations(
        "The formation contains olivine [3].",
        NUMERIC_SOURCES,
        style="numeric",
    )

    assert "Unknown citation: 3" in validation["errors"]


def test_valid_citation_does_not_claim_factual_grounding():
    validation = validate_citations(
        "The formation contains olivine [1].",
        NUMERIC_SOURCES,
        style="numeric",
    )

    assert validation["citation_valid"] is True
    assert validation["factual_grounding"] == {
        "status": "not_verified",
        "reason": "Valid citation labels do not prove that claims are supported.",
    }


def test_comparison_requires_citations_from_both_documents():
    sources = [
        {"label": "S1", "filename": "a.pdf"},
        {"label": "S2", "filename": "b.pdf"},
    ]

    validation = validate_citations(
        "Only document A is discussed [S1].",
        sources,
        style="source",
        require_all_documents=True,
    )

    assert "Answer does not cite all compared documents" in validation["errors"]


def test_comparison_rejects_invalid_bracket_format():
    validation = validate_citations(
        "The findings differ [source S1].",
        [{"label": "S1", "filename": "a.pdf"}],
        style="source",
    )

    assert "Invalid citation format: [source S1]" in validation["errors"]
