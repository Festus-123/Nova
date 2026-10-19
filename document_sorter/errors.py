"""
Request-level errors raised by the document sorter.

Each error carries a stable machine-readable ``code`` and the HTTP status the
API layer should use. Messages are safe to return to API clients: they never
contain document content, secrets, or stack traces.
"""

from typing import Any, Dict, Optional


class SortingError(Exception):
    code = "internal_error"
    http_status = 500

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        http_status: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        if code:
            self.code = code
        if http_status:
            self.http_status = http_status
        self.details = details or {}


class InvalidRequestError(SortingError):
    code = "invalid_request"
    http_status = 422


class NoFilesError(SortingError):
    code = "no_files"
    http_status = 400


class RequestTooLargeError(SortingError):
    code = "request_too_large"
    http_status = 413


class StorageError(SortingError):
    code = "temporary_storage_failure"
    http_status = 500


class ModelUnavailableError(SortingError):
    code = "model_unavailable"
    http_status = 502


class ModelOutputError(SortingError):
    """The model kept returning output that failed validation."""

    code = "malformed_model_output"
    http_status = 502


class SortingTimeoutError(SortingError):
    code = "timeout"
    http_status = 504
