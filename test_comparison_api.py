import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from api import app


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_UNKNOWN = "c" * 64


class TestComparisonAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

        # Replace the real document catalog
        self.catalog_patch = patch(
            "api.list_documents",
            return_value=[
                {"sha256": HASH_A, "filename": "paper_a.pdf"},
                {"sha256": HASH_B, "filename": "paper_b.pdf"},
            ],
        )

        self.catalog_patch.start()
        self.addCleanup(self.catalog_patch.stop)

    def test_valid_comparison(self):
        fake_response = {
            "answer": "Finding A [S1]. Finding B [S2].",
            "retrieved": [
                {"label": "S1", "filename": "paper_a.pdf"},
                {"label": "S2", "filename": "paper_b.pdf"},
            ],
            "validation": {
                "passed": True,
                "errors": [],
            },
        }

        with patch(
            "api.compare_documents",
            return_value=fake_response,
        ) as mock_compare:
            response = self.client.post(
                "/compare/documents",
                json={
                    "document_a": HASH_A,
                    "document_b": HASH_B,
                    "question": "Compare these papers",
                },
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), fake_response)
            mock_compare.assert_called_once_with(
                question="Compare these papers",
                document_a=HASH_A,
                document_b=HASH_B,
            )

    def test_same_document_rejected(self):
        response = self.client.post(
            "/compare/documents",
            json={
                "document_a": HASH_A,
                "document_b": HASH_A,
                "question": "Compare these papers",
            },
        )

        self.assertEqual(response.status_code, 422)

    def test_unknown_document_rejected(self):
        response = self.client.post(
            "/compare/documents",
            json={
                "document_a": HASH_A,
                "document_b": HASH_UNKNOWN,
                "question": "Compare these papers",
            },
        )

        self.assertEqual(response.status_code, 404)

    def test_invalid_hash_rejected(self):
        response = self.client.post(
            "/compare/documents",
            json={
                "document_a": "invalid-hash",
                "document_b": HASH_B,
                "question": "Compare these papers",
            },
        )

        self.assertEqual(response.status_code, 422)

    def test_empty_question_rejected(self):
        response = self.client.post(
            "/compare/documents",
            json={
                "document_a": HASH_A,
                "document_b": HASH_B,
                "question": "   ",
            },
        )

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
