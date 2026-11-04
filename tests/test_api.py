from unittest.mock import patch

import pymupdf
from fastapi.testclient import TestClient

import api


HASH_A = "a" * 64
HASH_B = "b" * 64


def test_liveness_does_not_require_external_services():
    response = TestClient(api.app).get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_rejects_whitespace_question_without_calling_model():
    with patch("api.answer_question") as answer_question:
        response = TestClient(api.app).post("/ask", json={"question": "   "})

    assert response.status_code == 422
    answer_question.assert_not_called()


def test_compare_rejects_unknown_document_without_model_call():
    with (
        patch(
            "api.list_documents",
            return_value=[{"sha256": HASH_A, "filename": "a.pdf"}],
        ),
        patch("api.compare_documents") as compare_documents,
    ):
        response = TestClient(api.app).post(
            "/compare/documents",
            json={
                "document_a": HASH_A,
                "document_b": HASH_B,
                "question": "Compare the evidence",
            },
        )

    assert response.status_code == 404
    compare_documents.assert_not_called()


def test_campaign_comparison_rejects_oversized_question():
    with patch("api.compare_campaigns") as compare_campaigns:
        response = TestClient(api.app).post(
            "/compare/campaigns",
            json={"question": "x" * 1001},
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    compare_campaigns.assert_not_called()


def test_document_comparison_rejects_oversized_question():
    hash_a = "a" * 64
    hash_b = "b" * 64
    with (
        patch(
            "api.list_documents",
            return_value=[
                {"sha256": hash_a, "filename": "a.pdf"},
                {"sha256": hash_b, "filename": "b.pdf"},
            ],
        ),
        patch("api.compare_documents") as compare_documents,
    ):
        response = TestClient(api.app).post(
            "/compare/documents",
            json={
                "document_a": hash_a,
                "document_b": hash_b,
                "question": "x" * 1001,
            },
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    compare_documents.assert_not_called()


def test_upload_rejects_malformed_pdf_and_removes_temporary_file(tmp_path):
    with patch.object(api, "UPLOAD_DIR", tmp_path):
        response = TestClient(api.app).post(
            "/documents/upload",
            files={"file": ("fake.pdf", b"not a pdf", "application/pdf")},
        )

    assert response.status_code == 422
    assert response.json()["error"]["message"] == (
        "Uploaded file is not a readable PDF"
    )
    assert list(tmp_path.iterdir()) == []


def test_upload_indexes_valid_pdf_without_real_embeddings(tmp_path):
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Mars sample science")
    content = document.tobytes()
    document.close()

    with (
        patch.object(api, "UPLOAD_DIR", tmp_path),
        patch(
            "api.index_uploaded_pdf",
            return_value={"status": "indexed", "chunks": 1},
        ),
    ):
        response = TestClient(api.app).post(
            "/documents/upload",
            files={"file": ("mars.pdf", content, "application/pdf")},
        )

    assert response.status_code == 201
    assert response.json()["index_status"] == "indexed"
    assert response.json()["chunks"] == 1
