import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path

from .models import (
    DocumentSortingResponse,
    DocumentGroup,
    IsolatedGroup,
    RejectedFile,
    DuplicateFile,
    PageRepresentation,
)
from .extractor import (
    extract_pages_from_file,
    compute_file_hash,
)
from .analyzer import analyze_and_group_pages


logger = logging.getLogger("nebula_sorter")

# Maximum permitted request size: 100 Megabytes
MAX_REQUEST_SIZE_BYTES = 100 * 1024 * 1024


class DocumentSortingEngine:
    """
    Generic Document Understanding and Sorting Engine for Nebula.
    Accepts raw file bytes/streams, dynamic schemas, and contextual batch constraints.
    """

    def __init__(self, llm=None):
        self.llm = llm

    def sort_documents(
        self,
        files: List[Dict[str, Any]],  # Each item: {"filename": str, "content": bytes, "file_id": Optional[str]}
        schema: List[str],
        allowed_values: Optional[Dict[str, List[str]]] = None,
        additional_context: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> DocumentSortingResponse:
        """
        Execute full document sorting pipeline across a contextual batch of files.
        """
        req_id = request_id or str(uuid.uuid4())
        logger.info(f"Starting document sorting request {req_id} with {len(files)} file(s).")

        # 1. VALIDATION: Schema validation
        if not schema or not isinstance(schema, list) or len(schema) == 0:
            raise ValueError("Invalid sorting schema: must provide a non-empty list of field names.")

        for field in schema:
            if not isinstance(field, str) or not field.strip():
                raise ValueError(f"Invalid schema field name: '{field}'. Must be non-empty strings.")

        # 2. VALIDATION: Files presence & Request size check
        if not files or len(files) == 0:
            return DocumentSortingResponse(
                status="error",
                request_id=req_id,
                rejected=[RejectedFile(file_id="none", original_filename="none", reason="No files provided in request.")],
                summary={"total_files": 0, "status": "no_files"},
            )

        total_size = sum(len(f.get("content", b"")) for f in files)
        if total_size > MAX_REQUEST_SIZE_BYTES:
            raise ValueError(
                f"Request size ({total_size / (1024*1024):.2f} MB) exceeds maximum permitted limit of 100 MB."
            )

        # 3. EXTRACTION, DEDUPLICATION, & REJECTION
        seen_hashes: Dict[str, str] = {}  # hash -> file_id
        rejected: List[RejectedFile] = []
        duplicates: List[DuplicateFile] = []
        all_pages: List[PageRepresentation] = []

        for idx, file_item in enumerate(files, start=1):
            filename = file_item.get("filename", f"file_{idx}")
            content = file_item.get("content", b"")
            file_id = file_item.get("file_id") or f"file_{idx:03d}"

            if len(content) == 0:
                rejected.append(
                    RejectedFile(
                        file_id=file_id,
                        original_filename=filename,
                        reason="Uploaded file is empty (0 bytes).",
                    )
                )
                continue

            file_hash = compute_file_hash(content)
            if file_hash in seen_hashes:
                duplicates.append(
                    DuplicateFile(
                        file_id=file_id,
                        original_filename=filename,
                        duplicate_of=seen_hashes[file_hash],
                        reason="Identical cryptographic file hash",
                    )
                )
                continue

            seen_hashes[file_hash] = file_id

            # Extract pages
            pages, rej = extract_pages_from_file(
                file_id=file_id,
                original_filename=filename,
                content=content,
                start_global_index=len(all_pages),
            )

            if rej:
                rejected.append(rej)
            else:
                all_pages.extend(pages)

        # 4. PAGE RELATIONSHIP ANALYSIS & GROUPING
        classified_groups: List[DocumentGroup] = []
        isolated_groups: List[IsolatedGroup] = []

        if all_pages:
            classified_groups, isolated_groups = analyze_and_group_pages(
                llm=self.llm,
                pages=all_pages,
                schema=schema,
                allowed_values=allowed_values,
                additional_context=additional_context,
            )

        # 5. OVERALL STATUS
        overall_status = "success"
        if not classified_groups and (isolated_groups or rejected):
            overall_status = "partial" if isolated_groups else "error"
        elif isolated_groups or rejected:
            overall_status = "partial"

        summary = {
            "total_files": len(files),
            "total_pages": len(all_pages),
            "classified_groups": len(classified_groups),
            "isolated_groups": len(isolated_groups),
            "rejected_files": len(rejected),
            "duplicate_files": len(duplicates),
        }

        logger.info(
            f"Sorting request {req_id} completed: {len(classified_groups)} classified groups, "
            f"{len(isolated_groups)} isolated, {len(rejected)} rejected, {len(duplicates)} duplicates."
        )

        return DocumentSortingResponse(
            status=overall_status,
            request_id=req_id,
            groups=classified_groups,
            isolated=isolated_groups,
            rejected=rejected,
            duplicates=duplicates,
            summary=summary,
        )
