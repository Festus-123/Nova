"""
Document sorting pipeline.

  1  validate request              (models.SortingRequest - done by the caller)
  2  ingest files to temp storage  (ingestion)    - size limits, hashing
  3  validate file types           (file_types)   - content sniffing, rejections
  4  exact-duplicate detection     (SHA-256)
  5  extract / render pages        (extraction)
  6  analyse pages                 (ai, stage A)  - vision + text
  7  relate pages / form groups    (ai, stage B)  - boundaries, continuations
  8  extract metadata per group    (ai, stage B)
  9  validate against schema       (classification)
  10 decide classified / isolated  (classification)
  11 structured response           (models.DocumentSortingResponse)

The temporary workspace is always removed, including on errors.
"""

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Callable, List, Optional, Sequence

from .ai import PageAnalysis, SorterModel, SortingContext, build_default_model
from .classification import annotate_duplicates, build_group, isolated_without_plan
from .config import SorterSettings
from .errors import ModelOutputError, SortingTimeoutError
from .extraction import FileRejected, extract_file
from .file_types import SUPPORTED_MIME_TYPES, cross_check, sniff_mime
from .ingestion import UploadInput, ingest_uploads, temporary_workspace
from .models import (
    DocumentGroup,
    DocumentSortingResponse,
    DuplicateFile,
    GroupStatus,
    IsolationReason,
    RejectedFile,
    RejectionReason,
    SortingRequest,
    SortingSummary,
)
from .pages import IngestedFile, Page

logger = logging.getLogger("nebula.sorter")


class _Deadline:
    def __init__(self, seconds: float, request_id: str):
        self.expires = time.monotonic() + seconds
        self.seconds = seconds
        self.request_id = request_id

    def remaining(self) -> float:
        return self.expires - time.monotonic()

    def check(self, stage: str):
        if time.monotonic() > self.expires:
            logger.error("request_id=%s stage=%s timeout_after_s=%.0f", self.request_id, stage, self.seconds)
            raise SortingTimeoutError(
                f"Document sorting exceeded the {self.seconds:.0f}s processing limit (during {stage}).",
                details={"stage": stage},
            )


class DocumentSortingEngine:
    """
    Generic document understanding / sorting capability.

    The engine is client-agnostic: the schema, allowed values and any
    external context come from each request.
    """

    def __init__(
        self,
        model: Optional[SorterModel] = None,
        settings: Optional[SorterSettings] = None,
        model_factory: Optional[Callable[[SorterSettings], SorterModel]] = None,
    ):
        self.settings = settings or SorterSettings.from_env()
        self._model = model
        self._model_factory = model_factory or build_default_model

    def _get_model(self) -> SorterModel:
        # A fresh client per request keeps per-request state (call counts,
        # vision fallback) isolated; an injected model is reused as-is.
        if self._model is not None:
            return self._model
        return self._model_factory(self.settings)

    @staticmethod
    def new_request_id() -> str:
        return f"nsr_{uuid.uuid4().hex}"

    # --------------------------------------------------------------

    def sort(self, request: SortingRequest, uploads: List[UploadInput]) -> DocumentSortingResponse:
        request_id = request.request_id or self.new_request_id()
        started = time.monotonic()
        deadline = _Deadline(self.settings.request_timeout_seconds, request_id)
        logger.info(
            "request_id=%s stage=start files=%d schema_fields=%d has_context=%s",
            request_id, len(uploads), len(request.schema_fields), bool(request.additional_context),
        )
        with temporary_workspace(request_id) as workspace:
            stage_start = time.monotonic()
            files = ingest_uploads(
                uploads, workspace, self.settings.max_request_bytes, self.settings.max_files
            )
            total_bytes = sum(f.size for f in files)
            self._log_stage(request_id, "ingest", stage_start, files=len(files), bytes=total_bytes)
            response = self._process(request, request_id, files, total_bytes, deadline, started)
        logger.info(
            "request_id=%s stage=done status=%s classified=%d isolated=%d rejected=%d duplicates=%d "
            "model_calls=%d duration_ms=%d",
            request_id, response.status, len(response.groups), len(response.isolated),
            len(response.rejected), len(response.duplicates), response.summary.model_calls,
            response.summary.processing_ms,
        )
        return response

    # --------------------------------------------------------------

    def _process(
        self,
        request: SortingRequest,
        request_id: str,
        files: List[IngestedFile],
        total_bytes: int,
        deadline: _Deadline,
        started: float,
    ) -> DocumentSortingResponse:
        settings = self.settings
        rejected: List[RejectedFile] = []
        duplicates: List[DuplicateFile] = []
        accepted: List[IngestedFile] = []
        file_warnings = {}

        # Stage: validation + exact duplicates
        stage_start = time.monotonic()
        first_by_hash = {}
        for file in files:
            if file.size == 0:
                rejected.append(_reject(file, RejectionReason.empty_file, "The file is empty (0 bytes)."))
                continue
            mime, problem = sniff_mime(file.path)
            if mime is None:
                rejected.append(
                    _reject(file, RejectionReason.unsupported_type, f"Unsupported or unrecognised file: {problem}.")
                )
                continue
            file.mime_type = mime
            file.kind = SUPPORTED_MIME_TYPES[mime]
            file_warnings[file.file_id] = cross_check(file.original_filename, file.declared_content_type, mime)
            if file.sha256 in first_by_hash:
                duplicates.append(
                    DuplicateFile(
                        file_id=file.file_id,
                        original_filename=file.original_filename,
                        duplicate_of_file_id=first_by_hash[file.sha256],
                    )
                )
                continue
            first_by_hash[file.sha256] = file.file_id
            accepted.append(file)
        self._log_stage(request_id, "validate", stage_start, accepted=len(accepted),
                        rejected=len(rejected), duplicates=len(duplicates))

        # Stage: extraction
        stage_start = time.monotonic()
        pages: List[Page] = []
        mime_by_file = {}
        for file in accepted:
            deadline.check("extraction")
            try:
                result = extract_file(file, len(pages), settings)
            except FileRejected as exc:
                rejected.append(_reject(file, exc.code, exc.reason))
                continue
            except Exception as exc:
                logger.error("request_id=%s stage=extract file_id=%s error=%s",
                             request_id, file.file_id, type(exc).__name__)
                rejected.append(_reject(file, RejectionReason.corrupted_file, "The file could not be processed."))
                continue
            if len(pages) + len(result.pages) > settings.max_pages:
                rejected.append(
                    _reject(
                        file,
                        RejectionReason.page_limit_exceeded,
                        f"Processing this file would exceed the limit of {settings.max_pages} pages per request; "
                        "send it in a separate request.",
                    )
                )
                continue
            for page in result.pages:
                page.warnings = file_warnings.get(file.file_id, []) + page.warnings
                if not page.text and not page.images:
                    page.warnings.append("No text or image content could be extracted.")
            mime_by_file[file.file_id] = file.mime_type
            pages.extend(result.pages)
        self._log_stage(request_id, "extract", stage_start, pages=len(pages),
                        rendered=sum(1 for p in pages if p.images),
                        ocr=sum(1 for p in pages if p.text_source == "ocr"))

        context = SortingContext(
            schema_fields=request.schema_fields,
            required_fields=request.required_fields or request.schema_fields,
            field_descriptions=request.field_descriptions,
            allowed_values=request.allowed_values,
            administrator_values=(request.additional_context.field_values if request.additional_context else {}),
            context_source=(request.additional_context.source if request.additional_context else None),
            context_notes=(request.additional_context.notes if request.additional_context else None),
            previous_result=request.previous_result,
        )

        groups: List[DocumentGroup] = []
        model_calls = 0
        if pages:
            model = self._get_model()
            analyses = self._analyze(model, pages, context, deadline, request_id)
            groups, plan_ok = self._group(model, pages, analyses, context, mime_by_file, deadline, request_id)
            model_calls = getattr(model, "calls", 0)

        classified = [g for g in groups if g.status == GroupStatus.classified]
        isolated = [g for g in groups if g.status == GroupStatus.isolated]

        if classified and not isolated and not rejected:
            status = "success"
        elif classified:
            status = "partial"
        else:
            status = "needs_review"

        return DocumentSortingResponse(
            status=status,
            request_id=request_id,
            schema=request.schema_fields,
            groups=classified,
            isolated=isolated,
            rejected=rejected,
            duplicates=duplicates,
            summary=SortingSummary(
                total_files=len(files),
                accepted_files=len(accepted) - len(
                    [r for r in rejected if r.file_id in {f.file_id for f in accepted}]
                ),
                total_bytes=total_bytes,
                total_pages=len(pages),
                classified_groups=len(classified),
                isolated_groups=len(isolated),
                rejected_files=len(rejected),
                duplicate_files=len(duplicates),
                model_calls=model_calls,
                processing_ms=int((time.monotonic() - started) * 1000),
            ),
        )

    # --------------------------------------------------------------

    def _analyze(
        self, model: SorterModel, pages: List[Page], context: SortingContext, deadline: _Deadline, request_id: str
    ) -> List[PageAnalysis]:
        stage_start = time.monotonic()
        size = self.settings.pages_per_analysis_call
        chunks = [pages[offset: offset + size] for offset in range(0, len(pages), size)]
        results: List[Optional[List[PageAnalysis]]] = [None] * len(chunks)
        failed_chunks = 0

        deadline.check("page analysis")
        # Chunks are independent (relationships are reasoned about later), so
        # they are analysed concurrently to keep large batches within budget.
        executor = ThreadPoolExecutor(max_workers=max(1, min(self.settings.analysis_concurrency, len(chunks))))
        try:
            futures = {
                executor.submit(model.analyze_pages, chunk, context): number
                for number, chunk in enumerate(chunks)
            }
            try:
                for future in as_completed(futures, timeout=max(0.0, deadline.remaining())):
                    number = futures[future]
                    try:
                        results[number] = future.result()
                    except ModelOutputError:
                        failed_chunks += 1
            except FuturesTimeout:
                deadline.check("page analysis")
                raise SortingTimeoutError("Document sorting exceeded the processing time limit (during page analysis).")
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        analyses: List[PageAnalysis] = []
        for chunk, result in zip(chunks, results):
            if result is not None:
                analyses.extend(result)
                continue
            for page in chunk:
                page.warnings.append("Automatic page analysis failed validation; page judged from extracted text only.")
                analyses.append(
                    PageAnalysis(
                        page_index=page.index,
                        readable=bool(page.text),
                        notes="Page analysis unavailable (model output invalid).",
                    )
                )
        self._log_stage(request_id, "page_analysis", stage_start, pages=len(pages),
                        chunks=len(chunks), failed_chunks=failed_chunks)
        return analyses

    def _group(
        self,
        model: SorterModel,
        pages: List[Page],
        analyses: Sequence[PageAnalysis],
        context: SortingContext,
        mime_by_file,
        deadline: _Deadline,
        request_id: str,
    ):
        stage_start = time.monotonic()
        deadline.check("grouping")
        by_index = {a.page_index: a for a in analyses}
        page_by_index = {p.index: p for p in pages}
        try:
            plan = model.plan_groups(pages, analyses, context)
        except ModelOutputError:
            logger.error("request_id=%s stage=grouping error=model_output_invalid", request_id)
            groups = []
            for file_id in dict.fromkeys(p.file_id for p in pages):
                file_pages = [p for p in pages if p.file_id == file_id]
                groups.append(
                    isolated_without_plan(
                        f"group_{len(groups) + 1:03d}",
                        file_pages,
                        context,
                        IsolationReason.model_output_invalid,
                        "Automatic grouping could not produce a valid result for this batch; manual review required.",
                    )
                )
            self._log_stage(request_id, "grouping", stage_start, groups=len(groups), plan="invalid")
            return groups, False

        ordered = sorted(plan.groups, key=lambda g: min(g.page_indices))
        groups = [
            build_group(
                f"group_{number:03d}",
                planned,
                [page_by_index[i] for i in planned.page_indices],
                by_index,
                context,
                mime_by_file,
            )
            for number, planned in enumerate(ordered, start=1)
        ]
        annotate_duplicates(groups, pages, plan)
        self._log_stage(request_id, "grouping", stage_start, groups=len(groups),
                        isolated=sum(1 for g in groups if g.status == GroupStatus.isolated))
        return groups, True

    @staticmethod
    def _log_stage(request_id: str, stage: str, stage_start: float, **counts):
        details = " ".join(f"{k}={v}" for k, v in counts.items())
        logger.info(
            "request_id=%s stage=%s duration_ms=%d %s",
            request_id, stage, int((time.monotonic() - stage_start) * 1000), details,
        )


def _reject(file: IngestedFile, code: RejectionReason, reason: str) -> RejectedFile:
    return RejectedFile(
        file_id=file.file_id,
        original_filename=file.original_filename,
        reason_code=code,
        reason=reason,
    )
