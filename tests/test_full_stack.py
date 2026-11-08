from __future__ import annotations

import os
import time

import pymupdf
import pytest
import requests

pytestmark = pytest.mark.skipif(
    os.getenv("NASA_RAG_FULL_STACK") != "1",
    reason="requires an explicitly started API, worker, and Ollama",
)


def test_opt_in_full_stack_upload_and_query(tmp_path):
    base_url = os.getenv("NASA_RAG_SMOKE_URL", "http://127.0.0.1:8000").rstrip("/")
    username = os.environ["NASA_RAG_SMOKE_USERNAME"]
    password = os.environ["NASA_RAG_SMOKE_PASSWORD"]

    token_response = requests.post(
        f"{base_url}/auth/token",
        json={"username": username, "password": password},
        timeout=10,
    )
    token_response.raise_for_status()
    headers = {
        "Authorization": f"Bearer {token_response.json()['access_token']}"
    }

    path = tmp_path / "full-stack-smoke.pdf"
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text(
        (72, 72),
        "The synthetic Zephyr instrument measured basalt in Aurora crater.",
    )
    pdf.save(path)
    pdf.close()

    with path.open("rb") as stream:
        upload = requests.post(
            f"{base_url}/documents/upload",
            headers=headers,
            files={"file": (path.name, stream, "application/pdf")},
            timeout=10,
        )
    assert upload.status_code == 202
    payload = upload.json()
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            job = requests.get(
                f"{base_url}/jobs/{payload['job_id']}", headers=headers, timeout=10
            )
            job.raise_for_status()
            state = job.json()["status"]
            if state == "succeeded":
                break
            if state in {"failed", "cancelled"}:
                pytest.fail(f"ingestion ended in {state}")
            time.sleep(1)
        else:
            pytest.fail("ingestion did not complete within 120 seconds")

        answer = requests.post(
            f"{base_url}/ask",
            headers=headers,
            json={
                "question": "What did the Zephyr instrument measure?",
                "document_ids": [payload["document_id"]],
                "retrieval_mode": "hybrid",
            },
            timeout=120,
        )
        answer.raise_for_status()
        assert answer.json()["retrieved"]
    finally:
        requests.delete(
            f"{base_url}/documents/{payload['document_id']}",
            headers=headers,
            timeout=10,
        ).raise_for_status()
