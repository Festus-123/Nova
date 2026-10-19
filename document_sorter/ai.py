"""
AI reasoning stages of the sorter.

Two model-backed stages, each with strict output validation:

  Stage A - page analysis (vision + text): what each page *shows* - headers,
            identifiers, question numbering, whether it starts/ends
            mid-content, layout, and any schema values visible on it.
  Stage B - batch reasoning (text only, over Stage A observations): page
            relationships, document boundaries, groups, and per-group values
            for the requested schema with evidence and confidence.

The model never sees a client name and the engine never branches on field
names: the schema, allowed values and context all come from the request.

Raw model output is never trusted: it is parsed, validated with Pydantic and,
on failure, the model is asked once (configurable) to repair it. If it still
fails, ModelOutputError is raised and the caller isolates the affected pages.
"""

import base64
import json
import logging
import threading
from typing import Any, Dict, List, Literal, Optional, Protocol, Sequence

from pydantic import BaseModel, Field, ValidationError, field_validator

from .config import SorterSettings
from .errors import ModelOutputError, ModelUnavailableError
from .pages import Page

logger = logging.getLogger("nebula.sorter.ai")

MAX_ALLOWED_VALUES_IN_PROMPT = 300
MAX_PAGE_TEXT_IN_PROMPT = 3500


# ============================================================
# MODEL OUTPUT SCHEMAS
# ============================================================

def _clip(value: Any, limit: int) -> Any:
    if isinstance(value, str):
        return value.strip()[:limit]
    return value


class FieldObservation(BaseModel):
    value: str
    evidence: str = ""

    @field_validator("value", "evidence", mode="before")
    @classmethod
    def _clip_text(cls, v):
        return _clip(v if v is not None else "", 300)


class PageAnalysis(BaseModel):
    page_index: int
    readable: bool
    content_kind: str = ""
    header_text: Optional[str] = None
    starts_new_document: bool = False
    identifiers: List[str] = Field(default_factory=list)
    question_numbers: List[int] = Field(default_factory=list)
    starts_mid_content: bool = False
    ends_mid_content: bool = False
    layout_signature: str = ""
    field_observations: Dict[str, FieldObservation] = Field(default_factory=dict)
    notes: str = ""

    @field_validator("content_kind", "layout_signature", mode="before")
    @classmethod
    def _short(cls, v):
        return _clip(v or "", 160)

    @field_validator("header_text", mode="before")
    @classmethod
    def _header(cls, v):
        return _clip(v, 400) if v else None

    @field_validator("notes", mode="before")
    @classmethod
    def _notes(cls, v):
        return _clip(v or "", 400)

    @field_validator("identifiers", mode="before")
    @classmethod
    def _ids(cls, v):
        return [str(i).strip()[:80] for i in (v or []) if str(i).strip()][:12]

    @field_validator("question_numbers", mode="before")
    @classmethod
    def _questions(cls, v):
        out = []
        for item in v or []:
            try:
                number = int(item)
            except (TypeError, ValueError):
                continue
            if 0 < number <= 500 and number not in out:
                out.append(number)
        return out[:100]


class PageAnalysisBatch(BaseModel):
    pages: List[PageAnalysis]


class PlannedField(BaseModel):
    value: Optional[str] = None
    source: Literal["page_content", "batch_context", "administrator", "not_found"] = "not_found"
    evidence: str = ""
    alternatives: List[str] = Field(default_factory=list)

    @field_validator("value", mode="before")
    @classmethod
    def _value(cls, v):
        if v is None:
            return None
        text = str(v).strip()
        if not text or text.lower() in ("null", "none", "unknown", "n/a", "not found"):
            return None
        return text[:200]

    @field_validator("evidence", mode="before")
    @classmethod
    def _evidence(cls, v):
        return _clip(v or "", 400)

    @field_validator("alternatives", mode="before")
    @classmethod
    def _alts(cls, v):
        return [str(a).strip()[:200] for a in (v or []) if str(a).strip()][:5]


class PlannedGroup(BaseModel):
    page_indices: List[int]
    decision: Literal["classified", "isolated"]
    isolation_reason_code: Optional[str] = None
    isolation_reason: Optional[str] = None
    confidence: Literal["high", "medium", "low", "insufficient"]
    relationship_evidence: List[str] = Field(default_factory=list)
    fields: Dict[str, PlannedField] = Field(default_factory=dict)
    conflicts: List[str] = Field(default_factory=list)

    @field_validator("relationship_evidence", "conflicts", mode="before")
    @classmethod
    def _lists(cls, v):
        return [str(e).strip()[:400] for e in (v or []) if str(e).strip()][:12]

    @field_validator("isolation_reason", mode="before")
    @classmethod
    def _reason(cls, v):
        return _clip(v, 500) if v else None


class PageRelationship(BaseModel):
    page_index: int
    relationship: Literal[
        "first_page", "continuation", "same_document", "new_document",
        "duplicate", "unrelated", "unclear",
    ]
    related_page_index: Optional[int] = None


class GroupingPlan(BaseModel):
    page_relationships: List[PageRelationship] = Field(default_factory=list)
    groups: List[PlannedGroup]


# ============================================================
# MODEL INTERFACE
# ============================================================

class SortingContext(BaseModel):
    """Everything the model needs to know about the request (no file bytes)."""

    schema_fields: List[str]
    required_fields: List[str]
    field_descriptions: Dict[str, str] = Field(default_factory=dict)
    allowed_values: Dict[str, List[str]] = Field(default_factory=dict)
    administrator_values: Dict[str, str] = Field(default_factory=dict)
    context_source: Optional[str] = None
    context_notes: Optional[str] = None
    previous_result: Optional[Dict[str, Any]] = None


class SorterModel(Protocol):
    calls: int

    def analyze_pages(self, pages: Sequence[Page], context: SortingContext) -> List[PageAnalysis]: ...

    def plan_groups(
        self, pages: Sequence[Page], analyses: Sequence[PageAnalysis], context: SortingContext
    ) -> GroupingPlan: ...


# ============================================================
# PROMPTS
# ============================================================

REASONING_RULES = """
RULES - DO NOT GUESS:
- Use only evidence actually present: page content, headers, identifiers
  (e.g. course codes, document numbers, titles), question numbering, page
  numbers, visual/typographic continuity, shared instructions or
  headers/footers, document structure, neighbouring pages and batch context.
- One file is NOT automatically one document. Several images/pages can form
  one document, and one file can contain several documents.
- A page WITHOUT a header is not automatically unknown: inspect its
  relationship with the surrounding pages (question numbers continuing,
  sentence or question continuing, same layout/handwriting/typography).
- A NEW header/identifier usually starts a new document. When it appears,
  verify that the preceding pages really belong to the previous document.
- Filenames are weak hints at most (they are often like IMG_001.jpg).
- When evidence conflicts, state the conflict, lower confidence, and isolate
  when the conflict affects the classification.
- When several groupings or values are plausible and the evidence does not
  clearly favour one, isolate the affected group instead of choosing.
- Never invent a value that is not supported by evidence. Leave it null.
- Evidence is not only literal labels: what a document clearly IS (e.g. an
  examination question paper, a lab manual, a certificate) supports the
  allowed value that names that kind of document. State that reasoning in
  the evidence. Guessing means choosing without such support.
- Report only the value itself: for an identifier/code field give the code
  alone (e.g. "ABC 101", not "ABC 101: Introduction to X"), without labels
  such as "Course:" or "Session:".
""".strip()


def _describe_fields(context: SortingContext) -> str:
    if not context.field_descriptions:
        return ""
    lines = [f"- {f}: {context.field_descriptions[f]}" for f in context.schema_fields if f in context.field_descriptions]
    return "FIELD MEANINGS (from the client):\n" + "\n".join(lines) + "\n"


def _describe_allowed(context: SortingContext) -> str:
    if not context.allowed_values:
        return "No restricted values: report values exactly as they appear in the documents (value only)."
    lines = []
    for field in context.schema_fields:
        values = context.allowed_values.get(field)
        if not values:
            lines.append(
                f"- {field}: unrestricted (report the value as printed, value only - no descriptive title or label)"
            )
            continue
        shown = values[:MAX_ALLOWED_VALUES_IN_PROMPT]
        more = len(values) - len(shown)
        suffix = f" ... and {more} more (if the document shows a value not listed here, report it as printed)" if more > 0 else ""
        lines.append(f"- {field}: one of {json.dumps(shown)}{suffix}")
    return "\n".join(lines)


def _describe_admin(context: SortingContext) -> str:
    parts = []
    if context.administrator_values:
        parts.append(
            "EXTERNAL CONTEXT supplied by the client "
            f"(source: {context.context_source or 'administrator'}). These values were NOT read from the "
            "documents; use them as authoritative for these fields and report source='administrator':\n"
            + json.dumps(context.administrator_values, ensure_ascii=False)
        )
    if context.context_notes:
        parts.append("Notes from the client (external, not document content): " + context.context_notes)
    if context.previous_result:
        parts.append(
            "PREVIOUS RESULT for these documents (for reference; re-evaluate, do not copy blindly):\n"
            + json.dumps(context.previous_result, ensure_ascii=False)[:4000]
        )
    return "\n\n".join(parts) if parts else "None."


def page_analysis_prompt(pages: Sequence[Page], context: SortingContext) -> str:
    field_list = ", ".join(context.schema_fields)
    return f"""You are the page-analysis stage of Nebula's document understanding engine.

Describe each page below factually. Do not group pages or decide the final
classification yet; another stage does that using your observations.

Fields the client wants to organise documents by: {field_list}
{_describe_fields(context)}Valid values (for recognising them on the page):
{_describe_allowed(context)}

For every page, report:
- readable: false only if the page content cannot be read at all.
- content_kind: short description (e.g. "examination question paper", "lecture notes", "blank page", "photo of unrelated object").
- header_text: verbatim title/header block text if the page has one, else null.
- starts_new_document: true if the page visibly begins a document (title block, institution header, course/document title, "Question 1" with a fresh header).
- identifiers: identifying codes/titles visible on the page (course codes, document numbers, titles).
- question_numbers: question/item numbers visible on the page, in order.
- starts_mid_content / ends_mid_content: page begins/ends in the middle of a question, sentence or table.
- layout_signature: short description of layout, typography or handwriting that helps match pages of the same document.
- field_observations: ONLY for fields listed above whose value is visibly stated on THIS page:
  {{"<field>": {{"value": "...", "evidence": "where/how it appears"}}}}. Omit fields not visible.
- notes: anything else relevant (page numbers, "continued", instructions shared across pages).

{REASONING_RULES}

Return ONLY a JSON object of the form:
{{"pages": [{{"page_index": <int>, "readable": true, "content_kind": "", "header_text": null,
 "starts_new_document": false, "identifiers": [], "question_numbers": [], "starts_mid_content": false,
 "ends_mid_content": false, "layout_signature": "", "field_observations": {{}}, "notes": ""}}]}}
with exactly one entry per page_index listed below.
"""


def _page_block(page: Page) -> str:
    text = page.text[:MAX_PAGE_TEXT_IN_PROMPT] if page.text else ""
    truncated = " [text truncated]" if page.text and len(page.text) > MAX_PAGE_TEXT_IN_PROMPT else ""
    meta = f"\nFile metadata: {json.dumps(page.file_metadata)}" if page.file_metadata else ""
    image_note = f"\n({len(page.images)} image(s) of this page follow)" if page.images else ""
    return (
        f"=== PAGE page_index={page.index} | file '{page.original_filename}' page {page.page_number} "
        f"| type {page.file_kind} | text source: {page.text_source}{truncated} ==={meta}{image_note}\n"
        f"{text if text else '[no extracted text]'}\n"
    )


def grouping_prompt(pages: Sequence[Page], analyses: Sequence[PageAnalysis], context: SortingContext) -> str:
    observations = []
    for page, analysis in zip(pages, analyses):
        record = analysis.model_dump()
        record["file"] = page.original_filename
        record["file_page_number"] = page.page_number
        record["printed_page_number"] = page.signals.printed_page_number
        record["detected_question_numbers_from_text"] = page.signals.question_numbers
        observations.append(record)

    return f"""You are the batch-reasoning stage of Nebula's document understanding engine.

The pages below were uploaded together as ONE contextual batch, listed in
upload order. Using the page observations, determine:
1. the relationship of each page to the previous pages
   (first_page, continuation, same_document, new_document, duplicate, unrelated, unclear);
2. document groups: every page_index must appear in EXACTLY ONE group;
3. for each group, a value for every schema field, with evidence.

SCHEMA FIELDS (in order): {json.dumps(context.schema_fields)}
REQUIRED FIELDS (a group missing any of these must be isolated): {json.dumps(context.required_fields)}
{_describe_fields(context)}VALID VALUES:
{_describe_allowed(context)}

ADDITIONAL CONTEXT:
{_describe_admin(context)}

{REASONING_RULES}

Field sources:
- "page_content": stated on one of the group's own pages.
- "batch_context": not on the group's pages, but clearly implied by related pages in the batch.
- "administrator": taken from the external context above.
- "not_found": no evidence (value must be null).

Decision:
- "classified" only if every required field has a supported value and the group boundaries are clear.
- otherwise "isolated" with isolation_reason_code one of: missing_required_field, ambiguous_value,
  ambiguous_boundary, unreadable_content, conflicting_metadata, unsupported_content,
  insufficient_context, multiple_possible_classifications, invalid_schema_value,
  and a clear human-readable isolation_reason.
Confidence: high (all values directly evidenced, boundaries unambiguous), medium (some values rely on
batch context or minor ambiguity), low (weak/partly conflicting evidence), insufficient (cannot classify).

PAGE OBSERVATIONS (JSON):
{json.dumps(observations, ensure_ascii=False)}

Return ONLY a JSON object:
{{"page_relationships": [{{"page_index": 0, "relationship": "first_page", "related_page_index": null}}],
 "groups": [{{"page_indices": [0, 1], "decision": "classified", "isolation_reason_code": null,
   "isolation_reason": null, "confidence": "high",
   "relationship_evidence": ["why these pages belong together / where the boundary is"],
   "fields": {{"<schema field>": {{"value": "<value only, e.g. a code without its title>", "source": "page_content", "evidence": "...", "alternatives": []}}}},
   "conflicts": []}}]}}
"""


# ============================================================
# JSON PARSING
# ============================================================

def extract_json_object(text: str) -> Any:
    """Parse the first JSON object in a model reply (tolerates code fences / prose)."""
    if not isinstance(text, str):
        raise ValueError("model reply is not text")
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found in model reply")
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(text[start:])
    return obj


def _content_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content or "")


# ============================================================
# LANGCHAIN-BACKED MODEL
# ============================================================

class LangChainSorterModel:
    """SorterModel implementation on top of a LangChain chat model."""

    def __init__(self, llm, settings: SorterSettings):
        self.llm = llm
        self.settings = settings
        self.calls = 0
        self._vision_supported = settings.vision_enabled
        self._lock = threading.Lock()

    # ---- low-level -------------------------------------------------

    def _invoke(self, messages: List[Any]) -> str:
        with self._lock:  # page-analysis chunks run concurrently
            self.calls += 1
        try:
            reply = self.llm.invoke(messages)
        except Exception as exc:
            raise ModelUnavailableError(
                f"The language model request failed ({type(exc).__name__})."
            ) from exc
        return _content_text(reply)

    def _structured(self, messages: List[Any], schema: type, validate) -> Any:
        """Invoke, parse, validate; ask the model to repair invalid output."""
        from langchain_core.messages import AIMessage, HumanMessage

        attempts = 1 + max(0, self.settings.model_repair_attempts)
        last_error = "unknown"
        for attempt in range(attempts):
            reply = self._invoke(messages)
            try:
                parsed = schema.model_validate(extract_json_object(reply))
                validate(parsed)
                return parsed
            except (ValueError, ValidationError) as exc:
                last_error = _short_error(exc)
                logger.warning("model_output_invalid attempt=%d error=%s", attempt + 1, last_error)
                messages = list(messages) + [
                    AIMessage(content=reply[:8000]),
                    HumanMessage(
                        content=(
                            "Your previous reply was not valid: "
                            f"{last_error}. Reply again with ONLY the corrected JSON object."
                        )
                    ),
                ]
        raise ModelOutputError(f"Model output failed validation: {last_error}")

    # ---- stage A -----------------------------------------------------

    def analyze_pages(self, pages: Sequence[Page], context: SortingContext) -> List[PageAnalysis]:
        from langchain_core.messages import HumanMessage, SystemMessage

        expected = [p.index for p in pages]

        def validate(batch: PageAnalysisBatch):
            got = [a.page_index for a in batch.pages]
            if sorted(got) != sorted(expected):
                raise ValueError(f"expected exactly one entry for page_index {expected}, got {got}")

        def build(include_images: bool):
            content: List[Dict[str, Any]] = []
            for page in pages:
                content.append({"type": "text", "text": _page_block(page)})
                if include_images:
                    for image in page.images:
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/jpeg;base64," + base64.b64encode(image).decode("ascii"),
                                    "detail": "high",
                                },
                            }
                        )
            return [SystemMessage(content=page_analysis_prompt(pages, context)), HumanMessage(content=content)]

        use_images = self._vision_supported and any(p.images for p in pages)
        try:
            batch = self._structured(build(use_images), PageAnalysisBatch, validate)
        except ModelUnavailableError:
            if not use_images:
                raise
            # Some providers/models reject image input; retry text-only once
            # and remember the outcome for the rest of the request.
            logger.warning("vision_request_failed falling back to text-only analysis")
            self._vision_supported = False
            batch = self._structured(build(False), PageAnalysisBatch, validate)

        by_index = {a.page_index: a for a in batch.pages}
        return [by_index[i] for i in expected]

    # ---- stage B -----------------------------------------------------

    def plan_groups(
        self, pages: Sequence[Page], analyses: Sequence[PageAnalysis], context: SortingContext
    ) -> GroupingPlan:
        from langchain_core.messages import HumanMessage, SystemMessage

        valid = {p.index for p in pages}

        def validate(plan: GroupingPlan):
            seen: Dict[int, int] = {}
            for number, group in enumerate(plan.groups):
                if not group.page_indices:
                    raise ValueError(f"group {number} has no page_indices")
                for index in group.page_indices:
                    if index not in valid:
                        raise ValueError(f"page_index {index} does not exist")
                    if index in seen:
                        raise ValueError(f"page_index {index} appears in more than one group")
                    seen[index] = number
            missing = sorted(valid - set(seen))
            if missing:
                raise ValueError(f"page_index values {missing} are not assigned to any group")

        messages = [
            SystemMessage(content=grouping_prompt(pages, analyses, context)),
            HumanMessage(content="Produce the grouping plan JSON now."),
        ]
        return self._structured(messages, GroupingPlan, validate)


def _short_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        problems = [
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()[:5]
        ]
        return "; ".join(problems)
    return str(exc)[:300]


def build_default_model(settings: SorterSettings) -> LangChainSorterModel:
    """Create the sorter's model client from environment configuration."""
    import os

    if not os.environ.get("OPENAI_API_KEY"):
        raise ModelUnavailableError(
            "The document sorter's language model is not configured on the server.",
            http_status=503,
        )
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=settings.model_name,
        base_url=settings.model_base_url,
        temperature=0,
        timeout=settings.model_timeout_seconds,
        max_retries=2,
    )
    return LangChainSorterModel(llm, settings)
