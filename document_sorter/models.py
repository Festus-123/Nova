"""
Public request/response contract of the document sorter.

These Pydantic models are the stable API surface. Everything a client sends is
validated here before any processing happens, and every response the engine
returns is an instance of ``DocumentSortingResponse``.
"""

import re
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError


ENGINE_VERSION = "1.0.0"

FIELD_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:\-]{1,128}$")

MAX_SCHEMA_FIELDS = 20
MAX_ALLOWED_VALUES_PER_FIELD = 5000
MAX_VALUE_LENGTH = 200

# Keys of `additional_context` that carry metadata about the context itself
# rather than a value for a schema field.
CONTEXT_META_KEYS = {"source", "notes"}


def _invalid(code: str, message: str) -> PydanticCustomError:
    """Validation error whose type is a stable API error code."""
    return PydanticCustomError(code, "{message}", {"message": message})


# ============================================================
# ENUMS
# ============================================================

class Confidence(str, Enum):
    """
    Qualitative confidence. No numeric score is exposed on purpose.

    high         - every required field is directly supported by document
                   content and page relationships are unambiguous.
    medium       - fields are supported, but at least one relies on
                   neighbouring-page / batch context or minor ambiguity exists.
    low          - evidence is weak or partly conflicting.
    insufficient - evidence does not support a classification.
    """

    high = "high"
    medium = "medium"
    low = "low"
    insufficient = "insufficient"


class GroupStatus(str, Enum):
    classified = "classified"
    isolated = "isolated"


class FieldSource(str, Enum):
    """Where a classification value came from."""

    document = "document"            # read from the group's own pages
    batch_context = "batch_context"  # inferred from related pages in the batch
    administrator = "administrator"  # supplied in additional_context (not from the document)


class IsolationReason(str, Enum):
    missing_required_field = "missing_required_field"
    ambiguous_value = "ambiguous_value"
    ambiguous_boundary = "ambiguous_boundary"
    unreadable_content = "unreadable_content"
    conflicting_metadata = "conflicting_metadata"
    unsupported_content = "unsupported_content"
    insufficient_context = "insufficient_context"
    multiple_possible_classifications = "multiple_possible_classifications"
    invalid_schema_value = "invalid_schema_value"
    low_confidence = "low_confidence"
    model_output_invalid = "model_output_invalid"


class RejectionReason(str, Enum):
    empty_file = "empty_file"
    unsupported_type = "unsupported_type"
    corrupted_file = "corrupted_file"
    encrypted_file = "encrypted_file"
    unreadable_content = "unreadable_content"
    unrelated_content = "unrelated_content"
    page_limit_exceeded = "page_limit_exceeded"


# ============================================================
# REQUEST
# ============================================================

class AdditionalContext(BaseModel):
    """
    External context supplied by the client, typically an administrator
    correction during reprocessing. Values here are never treated as if they
    were read from the document; they are reported with
    ``field_source = "administrator"``.
    """

    model_config = ConfigDict(extra="forbid")

    source: str = Field(default="administrator", max_length=64)
    field_values: Dict[str, str] = Field(default_factory=dict)
    notes: Optional[str] = Field(default=None, max_length=2000)


class SortingRequest(BaseModel):
    """Validated, normalized sorting request (independent of transport)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    task: Literal["document_sorting"] = "document_sorting"
    request_id: Optional[str] = None
    schema_fields: List[str] = Field(..., alias="schema")
    required_fields: Optional[List[str]] = None
    field_descriptions: Dict[str, str] = Field(default_factory=dict)
    allowed_values: Dict[str, List[str]] = Field(default_factory=dict)
    additional_context: Optional[AdditionalContext] = None
    previous_result: Optional[Dict[str, Any]] = None

    @field_validator("request_id")
    @classmethod
    def _check_request_id(cls, value):
        if value is None:
            return None
        value = value.strip()
        if not value:
            return None
        if not REQUEST_ID_PATTERN.match(value):
            raise _invalid("invalid_request_id", 
                "request_id must be 1-128 characters of letters, digits, '.', '_', ':' or '-'."
            )
        return value

    @field_validator("schema_fields")
    @classmethod
    def _check_schema(cls, fields):
        if not fields:
            raise _invalid("malformed_schema", "schema must contain at least one field name.")
        if len(fields) > MAX_SCHEMA_FIELDS:
            raise _invalid("malformed_schema", f"schema may contain at most {MAX_SCHEMA_FIELDS} fields.")
        cleaned = []
        for field in fields:
            if not isinstance(field, str):
                raise _invalid("malformed_schema", "schema field names must be strings.")
            name = field.strip()
            if not FIELD_NAME_PATTERN.match(name):
                raise _invalid("malformed_schema", 
                    f"Invalid schema field name {field!r}: use lowercase letters, digits and "
                    "underscores, starting with a letter (max 64 characters)."
                )
            if name in CONTEXT_META_KEYS:
                raise _invalid("malformed_schema", f"{name!r} is reserved and cannot be a schema field.")
            if name in cleaned:
                raise _invalid("malformed_schema", f"Duplicate schema field {name!r}.")
            cleaned.append(name)
        return cleaned

    @model_validator(mode="before")
    @classmethod
    def _normalize_inputs(cls, data):
        if not isinstance(data, dict):
            return data
        data = dict(data)
        schema = data.get("schema", data.get("schema_fields"))
        schema_names = [s.strip() for s in schema if isinstance(s, str)] if isinstance(schema, list) else []

        # allowed_values: accept keys that equal a schema field or its simple
        # plural ("level" <- "levels", "class" <- "classes").
        raw_allowed = data.get("allowed_values")
        if raw_allowed is None:
            data["allowed_values"] = {}
        elif isinstance(raw_allowed, dict) and schema_names:
            normalized: Dict[str, Any] = {}
            for key, values in raw_allowed.items():
                field = _resolve_field_key(str(key), schema_names)
                if field is None:
                    raise _invalid("invalid_allowed_values", 
                        f"allowed_values key {key!r} does not match any schema field {schema_names}."
                    )
                if field in normalized:
                    raise _invalid("invalid_allowed_values", f"allowed_values defines field {field!r} more than once.")
                normalized[field] = values
            data["allowed_values"] = normalized

        # additional_context: accept the flat form
        #   {"course": "EEE 306", "source": "administrator", "notes": "..."}
        # as well as the explicit {"field_values": {...}} form.
        raw_context = data.get("additional_context")
        if isinstance(raw_context, dict):
            if not raw_context:
                data["additional_context"] = None
            elif "field_values" not in raw_context:
                field_values = {}
                meta = {}
                for key, value in raw_context.items():
                    if key in CONTEXT_META_KEYS:
                        meta[key] = value
                    else:
                        field_values[key] = value
                data["additional_context"] = {**meta, "field_values": field_values}
        return data

    @field_validator("allowed_values")
    @classmethod
    def _check_allowed_values(cls, allowed):
        cleaned: Dict[str, List[str]] = {}
        for field, values in allowed.items():
            if not isinstance(values, list):
                raise _invalid("invalid_allowed_values", f"allowed_values[{field!r}] must be a list of strings.")
            if len(values) > MAX_ALLOWED_VALUES_PER_FIELD:
                raise _invalid("invalid_allowed_values", 
                    f"allowed_values[{field!r}] has more than {MAX_ALLOWED_VALUES_PER_FIELD} entries."
                )
            out = []
            for value in values:
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    value = str(value)
                if not isinstance(value, str) or not value.strip():
                    raise _invalid("invalid_allowed_values", f"allowed_values[{field!r}] must contain non-empty strings.")
                if len(value) > MAX_VALUE_LENGTH:
                    raise _invalid("invalid_allowed_values", f"allowed_values[{field!r}] contains a value over {MAX_VALUE_LENGTH} characters.")
                if value.strip() not in out:
                    out.append(value.strip())
            cleaned[field] = out
        return cleaned

    @model_validator(mode="after")
    def _check_cross_field(self):
        schema = self.schema_fields

        if self.required_fields is None:
            self.required_fields = list(schema)
        else:
            unknown = [f for f in self.required_fields if f not in schema]
            if unknown:
                raise _invalid("malformed_schema", f"required_fields {unknown} are not in the schema.")
            self.required_fields = [f for f in schema if f in self.required_fields]

        descriptions = {}
        for field, text in self.field_descriptions.items():
            if field not in schema:
                raise _invalid("malformed_schema", f"field_descriptions key {field!r} is not in the schema.")
            if not isinstance(text, str) or len(text) > 500:
                raise _invalid("malformed_schema", f"field_descriptions[{field!r}] must be a string of at most 500 characters.")
            if text.strip():
                descriptions[field] = text.strip()
        self.field_descriptions = descriptions

        if self.additional_context:
            cleaned = {}
            for field, value in self.additional_context.field_values.items():
                if field not in schema:
                    raise _invalid("invalid_additional_context", 
                        f"additional_context field {field!r} is not in the schema {schema}."
                    )
                if value is None:
                    continue
                if not isinstance(value, str):
                    value = str(value)
                value = value.strip()
                if not value:
                    continue
                if len(value) > MAX_VALUE_LENGTH:
                    raise _invalid("invalid_additional_context", f"additional_context value for {field!r} is too long.")
                allowed = self.allowed_values.get(field)
                if allowed:
                    from .classification import match_allowed_value  # local import avoids a cycle

                    matched = match_allowed_value(value, allowed)
                    if matched is None:
                        raise _invalid("invalid_additional_context", 
                            f"additional_context value {value!r} for {field!r} is not one of the allowed values."
                        )
                    value = matched
                cleaned[field] = value
            self.additional_context.field_values = cleaned
        return self


def _resolve_field_key(key: str, schema: List[str]) -> Optional[str]:
    key = key.strip()
    if key in schema:
        return key
    for field in schema:
        if key in (field + "s", field + "es") or (field.endswith("y") and key == field[:-1] + "ies"):
            return field
    return None


# ============================================================
# RESPONSE
# ============================================================

class SourceFileRef(BaseModel):
    """Which raw file (and which of its pages) contributed to a group."""

    file_id: str
    original_filename: str
    page_numbers: List[int] = Field(default_factory=list, description="1-based page numbers within the file")


class DocumentGroup(BaseModel):
    """One logical document assembled from one or more pages/files."""

    group_id: str
    status: GroupStatus
    source_files: List[SourceFileRef]
    classification: Dict[str, Optional[str]] = Field(
        description="One entry per schema field, in schema order. null when unknown."
    )
    field_sources: Dict[str, Optional[FieldSource]] = Field(
        description="Provenance of each classification value. null when the value is null."
    )
    proposed_filename: Optional[str] = Field(
        default=None,
        description="Suggested filename only; the client owns the final storage name. Null when isolated.",
    )
    confidence: Confidence
    reason: Optional[str] = Field(default=None, description="Human-readable isolation reason (isolated groups).")
    isolation_reasons: List[IsolationReason] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    possible_duplicate_of: Optional[str] = Field(
        default=None, description="group_id of an earlier group with the same content/classification."
    )


class RejectedFile(BaseModel):
    file_id: str
    original_filename: str
    reason_code: RejectionReason
    reason: str


class DuplicateFile(BaseModel):
    """A file byte-identical to an earlier file in the same request (not analysed again)."""

    file_id: str
    original_filename: str
    duplicate_of_file_id: str
    reason: str = "Identical file content (SHA-256 match) to an earlier file in this request."


class SortingSummary(BaseModel):
    total_files: int
    accepted_files: int
    total_bytes: int
    total_pages: int
    classified_groups: int
    isolated_groups: int
    rejected_files: int
    duplicate_files: int
    model_calls: int
    processing_ms: int


class DocumentSortingResponse(BaseModel):
    """
    status:
      success       - every accepted page ended up in a classified group and
                      no file was rejected.
      partial       - at least one group is classified, but some content is
                      isolated or some files were rejected.
      needs_review  - nothing could be classified; see isolated / rejected.
    """

    status: Literal["success", "partial", "needs_review"]
    task: Literal["document_sorting"] = "document_sorting"
    request_id: str
    engine_version: str = ENGINE_VERSION
    schema_fields: List[str] = Field(alias="schema", serialization_alias="schema")
    groups: List[DocumentGroup] = Field(default_factory=list, description="Classified groups")
    isolated: List[DocumentGroup] = Field(default_factory=list, description="Groups that need human review")
    rejected: List[RejectedFile] = Field(default_factory=list)
    duplicates: List[DuplicateFile] = Field(default_factory=list)
    summary: SortingSummary

    model_config = ConfigDict(populate_by_name=True)


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    request_id: Optional[str] = None
    error: ErrorBody
