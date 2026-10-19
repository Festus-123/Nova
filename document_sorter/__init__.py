"""
Nebula document understanding & sorting capability.

Generic and client-agnostic: callers supply files, a sorting schema, optional
allowed values and optional external context; the engine returns structured,
validated groups. See docs/DOCUMENT_SORTING_API.md for the HTTP contract.
"""

from .config import SorterSettings
from .engine import DocumentSortingEngine
from .errors import SortingError
from .ingestion import UploadInput
from .models import (
    AdditionalContext,
    Confidence,
    DocumentGroup,
    DocumentSortingResponse,
    DuplicateFile,
    ErrorResponse,
    FieldSource,
    GroupStatus,
    IsolationReason,
    RejectedFile,
    SortingRequest,
    SourceFileRef,
)

__all__ = [
    "AdditionalContext",
    "Confidence",
    "DocumentGroup",
    "DocumentSortingEngine",
    "DocumentSortingResponse",
    "DuplicateFile",
    "ErrorResponse",
    "FieldSource",
    "GroupStatus",
    "IsolationReason",
    "RejectedFile",
    "SorterSettings",
    "SortingError",
    "SortingRequest",
    "SourceFileRef",
    "UploadInput",
]
