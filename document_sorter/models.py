from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class SourceFileInfo(BaseModel):
    """Identifies the source file and pages that formed part of a document group."""
    file_id: str = Field(..., description="Unique ID for the uploaded file")
    original_filename: str = Field(..., description="Original filename as uploaded")
    page_numbers: List[int] = Field(default_factory=list, description="1-based page numbers contributing to this group")


class DocumentGroup(BaseModel):
    """A successfully understood and classified document group."""
    group_id: str = Field(..., description="Unique ID for the grouped document")
    source_files: List[SourceFileInfo] = Field(..., description="List of source files and page numbers")
    classification: Dict[str, Optional[str]] = Field(..., description="Extracted metadata matching the dynamic schema")
    proposed_filename: str = Field(..., description="Proposed standardized filename")
    status: str = Field(default="classified", description="'classified'")
    confidence: str = Field(default="high", description="'high', 'medium', or 'low'")
    evidence: List[str] = Field(default_factory=list, description="Reasoning and evidence points explaining the classification")
    warnings: List[str] = Field(default_factory=list, description="Any non-fatal anomalies or warnings")


class IsolatedGroup(BaseModel):
    """A document group that cannot be safely classified without human review."""
    group_id: str = Field(..., description="Unique ID for the isolated group")
    source_files: List[SourceFileInfo] = Field(..., description="List of source files and page numbers")
    partial_classification: Dict[str, Optional[str]] = Field(default_factory=dict, description="Any metadata extracted so far")
    proposed_filename: Optional[str] = Field(default=None, description="Proposed filename if available")
    status: str = Field(default="isolated", description="'isolated'")
    reason: str = Field(..., description="Human-readable explanation of why this group is isolated")
    evidence: List[str] = Field(default_factory=list, description="Observed evidence")
    warnings: List[str] = Field(default_factory=list, description="Any warnings")


class RejectedFile(BaseModel):
    """A file that is unusable, corrupted, or unsupported."""
    file_id: str = Field(..., description="Unique ID for the file")
    original_filename: str = Field(..., description="Original uploaded filename")
    reason: str = Field(..., description="Reason for rejection (e.g. corrupted, unsupported format)")


class DuplicateFile(BaseModel):
    """A file identified as an exact or probable duplicate of another uploaded file."""
    file_id: str = Field(..., description="File ID of the duplicate")
    original_filename: str = Field(..., description="Original uploaded filename")
    duplicate_of: str = Field(..., description="File ID of the original file")
    reason: str = Field(default="Identical cryptographic file hash", description="Reason for marking as duplicate")


class DocumentSortingResponse(BaseModel):
    """The complete structured machine-readable result of a document-sorting request."""
    status: str = Field(default="success", description="'success', 'partial', or 'error'")
    request_id: str = Field(..., description="Unique tracking ID for this request")
    groups: List[DocumentGroup] = Field(default_factory=list, description="Classified document groups")
    isolated: List[IsolatedGroup] = Field(default_factory=list, description="Isolated documents requiring human review")
    rejected: List[RejectedFile] = Field(default_factory=list, description="Rejected unusable files")
    duplicates: List[DuplicateFile] = Field(default_factory=list, description="Identified duplicate files")
    summary: Optional[Dict[str, Any]] = Field(default=None, description="Summary statistics of processed files and groups")


class PageRepresentation(BaseModel):
    """Internal model representing an extracted page of a document."""
    page_index: int = Field(..., description="Zero-based global page index across the batch")
    file_id: str = Field(..., description="Source file ID")
    original_filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="'pdf', 'image', or 'docx'")
    page_number: int = Field(..., description="1-based page number within source file")
    text_content: str = Field(default="", description="Extracted text from page")
    ocr_applied: bool = Field(default=False, description="Whether OCR was used")
    headers: List[str] = Field(default_factory=list, description="Detected title/header lines")
    question_numbers: List[int] = Field(default_factory=list, description="Detected question numbers")
    course_codes: List[str] = Field(default_factory=list, description="Detected course codes")
    has_boundary_header: bool = Field(default=False, description="Whether page has an explicit top-level exam/course header")
    file_hash: Optional[str] = Field(default=None, description="SHA256 hash of the parent file")
