from search_nasa import search_nasa


TEST_CASES = [
    {
        "question": "Which minerals formed through aqueous alteration?",
        "expected_phrase": "carbonate, iron oxides",
        "expected_page": 1,
    },
    {
        "question": "What evidence supports an igneous origin for Maaz and Seitah?",
        "expected_phrase": "olivine-rich cumulate",
        "expected_page": 1,
    },
    {
        "question": "What rock cores did Perseverance collect?",
        "expected_phrase": "Montdenier and Montagnac",
        "expected_page": 1,
    },
]


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def evaluate_retrieval():
    passed = 0

    for case in TEST_CASES:
        results = search_nasa(case["question"])
        expected_phrase = normalize(case["expected_phrase"])
        match = False

        for index, (document, distance) in enumerate(results, start=1):
            print(f"\nResult {index} (distance: {distance:.4f})")

            title = document.metadata.get("title", "Unknown title")
            page = document.metadata.get("page", "Unknown page")
            print(f"Title: {title}")
            print(f"Page: {page}")
            print(f"\n{document.page_content[:500]}")
            print(f"\n{document.page_content[:500]}")

            phrase_found = expected_phrase in normalize(document.page_content)
            page_matches = document.metadata.get("page") == case["expected_page"]

            if phrase_found and page_matches:
                match = True

        if match:
            passed += 1
            print(f"\nPASS: {case['question']}")
        else:
            print(f"\nFAIL: {case['question']}")

    hit_at_3 = passed / len(TEST_CASES) if TEST_CASES else 0
    print(f"\nHit@3: {passed}/{len(TEST_CASES)} ({hit_at_3:.2%})")


if __name__ == "__main__":
    evaluate_retrieval()
