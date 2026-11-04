class ServiceError(Exception):
    def __init__(self, *, code: str, message: str, status_code: int):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class ModelTimeoutError(ServiceError):
    def __init__(self):
        super().__init__(
            code="model_timeout",
            message="The local language model timed out",
            status_code=504,
        )


class OllamaUnavailableError(ServiceError):
    def __init__(self):
        super().__init__(
            code="ollama_unavailable",
            message="The local AI service is unavailable",
            status_code=503,
        )


class RetrievalUnavailableError(ServiceError):
    def __init__(self):
        super().__init__(
            code="retrieval_unavailable",
            message="The document index is unavailable",
            status_code=503,
        )


class IndexingFailedError(ServiceError):
    def __init__(self):
        super().__init__(
            code="indexing_failed",
            message="PDF indexing failed; the temporary upload was removed",
            status_code=503,
        )
