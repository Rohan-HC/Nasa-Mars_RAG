from unittest.mock import patch
from uuid import uuid4

import pymupdf
from fastapi.testclient import TestClient

import api

HASH_A = "a" * 64
HASH_B = "b" * 64


def test_liveness_does_not_require_external_services():
    response = TestClient(api.app).get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_rejects_whitespace_question_without_calling_model(authenticated_api):
    with patch("api.answer_question") as answer_question:
        response = authenticated_api["client"].post(
            "/ask",
            headers=authenticated_api["headers"],
            json={"question": "   "},
        )

    assert response.status_code == 422
    answer_question.assert_not_called()


def test_compare_rejects_unknown_document_without_model_call(authenticated_api):
    with patch("api.compare_documents") as compare_documents:
        response = authenticated_api["client"].post(
            "/compare/documents",
            headers=authenticated_api["headers"],
            json={
                "document_a": str(uuid4()),
                "document_b": str(uuid4()),
                "question": "Compare the evidence",
            },
        )

    assert response.status_code == 404
    compare_documents.assert_not_called()


def test_campaign_comparison_rejects_oversized_question(authenticated_api):
    with patch("api.compare_campaigns") as compare_campaigns:
        response = authenticated_api["client"].post(
            "/compare/campaigns",
            headers=authenticated_api["headers"],
            json={"question": "x" * 1001},
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    compare_campaigns.assert_not_called()


def test_document_comparison_rejects_oversized_question(authenticated_api):
    with patch("api.compare_documents") as compare_documents:
        response = authenticated_api["client"].post(
            "/compare/documents",
            headers=authenticated_api["headers"],
            json={
                "document_a": str(uuid4()),
                "document_b": str(uuid4()),
                "question": "x" * 1001,
            },
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    compare_documents.assert_not_called()


def test_upload_rejects_malformed_pdf_and_removes_temporary_file(authenticated_api):
    response = authenticated_api["client"].post(
        "/documents/upload",
        headers=authenticated_api["headers"],
        files={"file": ("fake.pdf", b"not a pdf", "application/pdf")},
    )

    assert response.status_code == 422
    assert response.json()["error"]["message"] == (
        "Uploaded file is not a readable PDF"
    )
    assert list(authenticated_api["upload_dir"].iterdir()) == []


def test_upload_indexes_valid_pdf_without_real_embeddings(authenticated_api):
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Mars sample science")
    content = document.tobytes()
    document.close()

    with patch.object(
        authenticated_api["app"].state.indexing_service, "process", create=True
    ) as process:
        response = authenticated_api["client"].post(
            "/documents/upload",
            headers=authenticated_api["headers"],
            files={"file": ("mars.pdf", content, "application/pdf")},
        )

    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    process.assert_not_called()
