import json
import logging
import sys
from datetime import UTC, datetime


LOG_FIELDS = (
    "request_id",
    "method",
    "path",
    "status_code",
    "duration_ms",
    "retrieval_ms",
    "model_ms",
    "error_code",
    "error_type",
    "error_location",
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for field in LOG_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception_type"] = record.exc_info[0].__name__
        return json.dumps(payload, separators=(",", ":"))


def configure_logging() -> None:
    root = logging.getLogger()
    if any(getattr(handler, "_nasa_rag_json", False) for handler in root.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler._nasa_rag_json = True
    root.addHandler(handler)
    root.setLevel(logging.INFO)
