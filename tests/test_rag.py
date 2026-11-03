from unittest.mock import Mock, patch

from langchain_core.documents import Document

import rag_nasa


def test_answer_refuses_when_retrieval_returns_no_evidence():
    with (
        patch("rag_nasa.search_nasa", return_value=[]),
        patch.object(rag_nasa, "llm") as llm,
    ):
        result = rag_nasa.answer_question("What happened on Mars?")

    assert result["answer"] == (
        "I cannot answer this from the available NASA documents."
    )
    assert result["retrieved"] == []
    llm.invoke.assert_not_called()


def test_answer_uses_mocked_model_and_returns_retrieved_source():
    document = Document(
        page_content="The sample contains olivine.",
        metadata={"filename": "mars.pdf", "page": 2},
    )
    model_response = Mock(content="The sample contains olivine [1].")
    model = Mock()
    model.invoke.return_value = model_response

    with (
        patch("rag_nasa.search_nasa", return_value=[(document, 0.1)]),
        patch.object(rag_nasa, "llm", model),
    ):
        result = rag_nasa.answer_question("What mineral was found?")

    assert result["answer"] == "The sample contains olivine [1]."
    assert result["retrieved"][0]["label"] == 1
    assert result["retrieved"][0]["title"] == "mars.pdf"
