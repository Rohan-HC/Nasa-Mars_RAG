import httpx
from ollama import ResponseError

from service_errors import (
    ModelTimeoutError,
    OllamaUnavailableError,
    RetrievalUnavailableError,
)


def retrieve(search_function, *args, **kwargs):
    try:
        return search_function(*args, **kwargs)
    except FileNotFoundError as error:
        raise RetrievalUnavailableError() from error
    except httpx.TimeoutException as error:
        raise ModelTimeoutError() from error
    except (httpx.HTTPError, ResponseError, ConnectionError) as error:
        raise OllamaUnavailableError() from error


def invoke_model(model, messages):
    try:
        return model.invoke(messages)
    except httpx.TimeoutException as error:
        raise ModelTimeoutError() from error
    except (httpx.HTTPError, ResponseError, ConnectionError) as error:
        raise OllamaUnavailableError() from error
