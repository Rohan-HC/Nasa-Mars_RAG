import json
import logging
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

import api
from logging_config import JsonFormatter
from nasa_rag import worker


def test_structured_logs_include_operational_fields_and_drop_sensitive_extras():
    record = logging.LogRecord(
        name="nasa_rag.worker",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="ingestion_job_complete",
        args=(),
        exc_info=None,
    )
    record.request_id = "request-1"
    record.job_id = "job-1"
    record.worker_id = "worker-1"
    record.stage = "embed"
    record.attempt_count = 2
    record.retrieval_mode = "hybrid"
    record.skip_count = 1
    record.reindex_count = 0
    record.prompt = "private prompt"
    record.document_text = "private body"
    record.auth_token = "secret token"
    record.storage_path = "C:/private/file.pdf"

    payload = json.loads(JsonFormatter().format(record))
    serialized = json.dumps(payload)

    assert payload["job_id"] == "job-1"
    assert payload["stage"] == "embed"
    assert payload["retrieval_mode"] == "hybrid"
    assert payload["skip_count"] == 1
    assert "private" not in serialized
    assert "secret token" not in serialized


def test_browser_ui_contains_authenticated_job_and_replacement_workflow():
    html = Path("web/index.html").read_text(encoding="utf-8")
    javascript = Path("web/app.js").read_text(encoding="utf-8")

    assert 'id="login-form"' in html
    assert 'id="logout-button"' in html
    assert 'id="replace-button"' in html
    assert 'id="retrieval-mode"' in html
    assert "sessionStorage" in javascript
    assert "Authorization" in javascript
    assert "/auth/token" in javascript
    assert "/jobs/" in javascript
    assert "document_id" in javascript
    assert "setTimeout" in javascript


def test_static_ui_assets_are_served_without_authentication():
    client = TestClient(api.app)

    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200


def test_compose_is_loopback_only_and_requires_external_secret():
    compose = Path("compose.yaml").read_text(encoding="utf-8")

    assert "127.0.0.1:8000:8000" in compose
    assert "NASA_RAG_JWT_SECRET: ${NASA_RAG_JWT_SECRET:?" in compose
    assert "host.docker.internal:11434" in compose
    assert "nasa_rag.worker" in compose
    assert "default-password" not in compose


def test_docker_build_context_excludes_all_local_env_variants():
    patterns = Path(".dockerignore").read_text(encoding="utf-8").splitlines()

    assert ".env*" in patterns
    assert "!.env.example" in patterns


def test_worker_process_configures_structured_logging_before_polling():
    ingestion_worker = Mock()
    with (
        patch("nasa_rag.worker.configure_logging") as configure,
        patch("nasa_rag.worker.build_worker", return_value=ingestion_worker),
    ):
        worker.main()

    configure.assert_called_once_with()
    ingestion_worker.run_forever.assert_called_once_with()
