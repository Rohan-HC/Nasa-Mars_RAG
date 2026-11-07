from unittest.mock import Mock, patch

import httpx
import pytest
from langchain_core.documents import Document

import compare_nasa
from service_errors import ModelTimeoutError


def fake_search(*, question, k, filename, version_ids=None):
    return [
        (
            Document(
                page_content=f"Evidence from {filename}",
                metadata={"filename": filename, "page": 1},
            ),
            0.1,
        )
    ]


def test_campaign_comparison_withholds_uncited_model_output():
    model = Mock()
    model.invoke.return_value = Mock(content="The campaigns differ.")

    with (
        patch("compare_nasa.search_nasa", side_effect=fake_search),
        patch.object(compare_nasa, "llm", model),
    ):
        result = compare_nasa.compare_campaigns("Compare the campaigns")

    assert result["answer"] == "Insufficient evidence for a reliable comparison."
    assert result["validation"]["citation_valid"] is False
    assert set(result["timings_ms"]) == {"retrieval", "model", "total"}


def test_campaign_comparison_converts_model_timeout():
    model = Mock()
    model.invoke.side_effect = httpx.TimeoutException("timed out")

    with (
        patch("compare_nasa.search_nasa", side_effect=fake_search),
        patch.object(compare_nasa, "llm", model),
    ):
        with pytest.raises(ModelTimeoutError):
            compare_nasa.compare_campaigns("Compare the campaigns")
