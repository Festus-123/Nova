"""
LangChain tool exposing the sorter to the general Nebula agent for local files.

The HTTP endpoint does NOT go through the agent; it calls the engine directly.
"""

import json
from pathlib import Path
from typing import List

from langchain_core.tools import tool

from .engine import DocumentSortingEngine
from .errors import SortingError
from .ingestion import UploadInput
from .models import SortingRequest


@tool
def sort_documents(
    file_paths: List[str],
    fields: List[str],
    allowed_values_json: str = "",
    additional_context_json: str = "",
) -> str:
    """
    Group and classify local document files (PDF, JPG, PNG, WebP, DOCX) by a
    schema of metadata fields. Related pages/files are grouped together;
    anything that cannot be classified safely is returned as 'isolated'.

    Args:
        file_paths: Local paths of the files, in upload/page order.
        fields: Field names to classify by, e.g. ["document_type", "year"].
        allowed_values_json: Optional JSON object mapping fields to permitted values.
        additional_context_json: Optional JSON object of externally known field values.
    """
    try:
        request = SortingRequest.model_validate(
            {
                "schema": fields,
                "allowed_values": json.loads(allowed_values_json) if allowed_values_json.strip() else {},
                "additional_context": json.loads(additional_context_json) if additional_context_json.strip() else None,
            }
        )
    except Exception as exc:
        return f"Invalid sorting request: {exc}"

    handles = []
    try:
        uploads = []
        for raw in file_paths:
            path = Path(raw.strip().strip("'\""))
            if not path.is_file():
                return f"File not found: {path}"
            handle = open(path, "rb")
            handles.append(handle)
            uploads.append(UploadInput(filename=path.name, stream=handle))
        result = DocumentSortingEngine().sort(request, uploads)
        return result.model_dump_json(by_alias=True, indent=2)
    except SortingError as exc:
        return f"Document sorting failed ({exc.code}): {exc.message}"
    finally:
        for handle in handles:
            handle.close()
