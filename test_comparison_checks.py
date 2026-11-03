import unittest

from comparison_checks import validate_comparison_response


class TestComparisonChecks(unittest.TestCase):
    def setUp(self):
        self.sources = [
            {"label": "S1", "filename": "mars_report.pdf"},
            {"label": "S3", "filename": "williams.pdf"},
        ]

    def test_valid_comparison(self):
        response = {
            "answer": "Igneous [S1]. Sedimentary [S3].",
            "retrieved": self.sources,
        }

        self.assertEqual(validate_comparison_response(response), [])

    def test_unknown_citation(self):
        response = {
            "answer": "Igneous [S1]. Sedimentary [S99].",
            "retrieved": self.sources,
        }

        errors = validate_comparison_response(response)

        self.assertIn("Unknown citation: S99", errors)

    def test_bibliography_citation(self):
        response = {
            "answer": "Igneous [S1]. Sedimentary [18].",
            "retrieved": self.sources,
        }

        errors = validate_comparison_response(response)

        self.assertIn("Invalid citation format: [18]", errors)

    def test_single_document_cited(self):
        response = {
            "answer": "Igneous rocks [S1].",
            "retrieved": self.sources,
        }

        errors = validate_comparison_response(response)

        self.assertIn(
            "Answer does not cite both documents",
            errors,
        )

    def test_grouped_citations(self):
        response = {
            "answer": "Both campaigns differ [S1, S3].",
            "retrieved": self.sources,
        }

        self.assertEqual(
            validate_comparison_response(response),
            [],
        )

    def test_invalid_citation_range(self):
        response = {
            "answer": (
                "The rocks have heterogeneous textures "
                "[S10-13]. Their origins differ [S1][S3]."
            ),
            "retrieved": self.sources,
        }

        errors = validate_comparison_response(response)

        self.assertIn(
            "Invalid citation format: [S10-13]",
            errors,
        )


if __name__ == "__main__":
    unittest.main()
