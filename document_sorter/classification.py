"""
Deterministic post-processing of the model's grouping plan.

  parse -> validate -> normalize -> check schema -> check allowed values
        -> decide classified / isolated -> structured DocumentGroup

Nothing here knows about any particular client or field name: rules apply to
whatever schema the request supplied.
"""

import re
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

from .ai import GroupingPlan, PageAnalysis, PlannedGroup, SortingContext
from .filenames import propose_filename
from .models import (
    Confidence,
    DocumentGroup,
    FieldSource,
    GroupStatus,
    IsolationReason,
    SourceFileRef,
)
from .pages import Page, hamming_distance

CONFIDENCE_ORDER = [Confidence.insufficient, Confidence.low, Confidence.medium, Confidence.high]


# ============================================================
# ALLOWED VALUE MATCHING
# ============================================================

def _canonical(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold()
    text = re.sub(r"[\s\-_/.,:;()]+", " ", text)
    return text.strip()


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", _canonical(value))


def match_allowed_value(value: str, allowed: Sequence[str]) -> Optional[str]:
    """
    Return the allowed value that `value` denotes, or None.

    Matching is deliberately conservative: case, punctuation and spacing are
    ignored ("abc101" == "ABC 101", "2023-2024" == "2023/2024"), but there is
    no substring or similarity matching, so "Material" never silently becomes
    "Materials" and "100" never matches "1000".
    """
    if value is None:
        return None
    canonical = _canonical(value)
    compact = _compact(value)
    for candidate in allowed:
        if _canonical(candidate) == canonical:
            return candidate
    matches = [c for c in allowed if _compact(c) == compact]
    if len(matches) == 1:
        return matches[0]
    # "ABC 101: Introduction to X" denotes "ABC 101": an allowed value, an
    # explicit separator, then a textual title. Both the separator and letters
    # in the title are required, so "1000" never matches "100" and
    # "2023-2024" never matches "2023".
    parts = re.split(r"\s*[:\-–—(|]\s*", value.strip(), maxsplit=1)
    if len(parts) == 2 and parts[0] and re.search(r"[A-Za-z]{2,}", parts[1]):
        titled = [c for c in allowed if _compact(c) == _compact(parts[0])]
        if len(titled) == 1:
            return titled[0]
    return None


# ============================================================
# GROUP CONSTRUCTION
# ============================================================

def _source_files(pages: Sequence[Page]) -> List[SourceFileRef]:
    refs: Dict[str, SourceFileRef] = {}
    for page in pages:
        ref = refs.setdefault(
            page.file_id,
            SourceFileRef(file_id=page.file_id, original_filename=page.original_filename, page_numbers=[]),
        )
        if page.page_number not in ref.page_numbers:
            ref.page_numbers.append(page.page_number)
    for ref in refs.values():
        ref.page_numbers.sort()
    return list(refs.values())


def _cap(confidence: Confidence, ceiling: Confidence) -> Confidence:
    return min(confidence, ceiling, key=CONFIDENCE_ORDER.index)


def build_group(
    group_id: str,
    planned: PlannedGroup,
    pages: Sequence[Page],
    analyses: Dict[int, PageAnalysis],
    context: SortingContext,
    mime_by_file: Dict[str, str],
    extra_warnings: Optional[List[str]] = None,
) -> DocumentGroup:
    schema = context.schema_fields
    pages = sorted(pages, key=lambda p: p.index)
    warnings: List[str] = list(extra_warnings or [])
    evidence: List[str] = list(planned.relationship_evidence)
    reasons: List[IsolationReason] = []
    reason_texts: List[str] = []

    def isolate(code: IsolationReason, text: str):
        if code not in reasons:
            reasons.append(code)
        if text not in reason_texts:
            reason_texts.append(text)

    unknown_fields = [f for f in planned.fields if f not in schema]
    if unknown_fields:
        warnings.append(f"Ignored model output for fields not in the schema: {unknown_fields}.")

    classification: Dict[str, Optional[str]] = {}
    sources: Dict[str, Optional[FieldSource]] = {}
    confidence = Confidence(planned.confidence)

    for field in schema:
        planned_field = planned.fields.get(field)
        doc_value = planned_field.value if planned_field and planned_field.source != "administrator" else None
        admin_value = context.administrator_values.get(field)

        if admin_value is not None:
            classification[field] = admin_value
            sources[field] = FieldSource.administrator
            evidence.append(
                f"{field}: '{admin_value}' supplied as external context "
                f"({context.context_source or 'administrator'}), not read from the document."
            )
            if doc_value and match_allowed_value(doc_value, [admin_value]) is None:
                warnings.append(
                    f"{field}: document evidence suggests '{doc_value}', but the externally supplied "
                    f"value '{admin_value}' was used."
                )
            continue

        if not doc_value:
            classification[field] = None
            sources[field] = None
            continue

        allowed = context.allowed_values.get(field)
        value = doc_value
        if allowed:
            matched = match_allowed_value(doc_value, allowed)
            if matched is None:
                isolate(
                    IsolationReason.invalid_schema_value,
                    f"The value '{doc_value}' found for '{field}' is not one of the allowed values.",
                )
                classification[field] = None
                sources[field] = None
                evidence.append(f"{field}: document shows '{doc_value}' (not an allowed value).")
                continue
            if matched != doc_value:
                warnings.append(f"{field}: '{doc_value}' normalized to allowed value '{matched}'.")
            value = matched

        source = FieldSource.document if planned_field.source == "page_content" else FieldSource.batch_context
        classification[field] = value
        sources[field] = source
        detail = f" - {planned_field.evidence}" if planned_field.evidence else ""
        evidence.append(f"{field}: '{value}'{detail}")
        if planned_field.alternatives:
            warnings.append(f"{field}: other possible values considered: {planned_field.alternatives}.")
        if source == FieldSource.batch_context:
            confidence = _cap(confidence, Confidence.medium)

    # Required fields
    missing = [f for f in context.required_fields if not classification.get(f)]
    if missing:
        isolate(
            IsolationReason.missing_required_field,
            "Could not determine " + ", ".join(f"'{f}'" for f in missing)
            + " from the document or its neighbouring pages.",
        )

    # Model's own judgement
    if planned.decision == "isolated":
        code = _isolation_code(planned.isolation_reason_code)
        resolved_by_admin = (
            code in (IsolationReason.missing_required_field, IsolationReason.insufficient_context)
            and not missing
            and any(sources.get(f) == FieldSource.administrator for f in schema)
        )
        if not resolved_by_admin:
            isolate(code, planned.isolation_reason or "The model could not classify this group safely.")

    if planned.conflicts:
        warnings.extend(f"Conflict: {c}" for c in planned.conflicts)
        confidence = _cap(confidence, Confidence.medium)
        if confidence in (Confidence.low, Confidence.insufficient):
            isolate(IsolationReason.conflicting_metadata, "Conflicting metadata: " + "; ".join(planned.conflicts))

    if confidence in (Confidence.low, Confidence.insufficient):
        isolate(
            IsolationReason.low_confidence,
            f"Confidence is {confidence.value}; the evidence is not strong enough to classify without review.",
        )

    readable = [analyses[p.index].readable for p in pages if p.index in analyses]
    if readable and not any(readable):
        isolate(IsolationReason.unreadable_content, "The page content could not be read.")
    elif readable and not all(readable):
        warnings.append("Some pages in this group could not be read.")

    if extra_warnings:
        confidence = _cap(confidence, Confidence.medium)

    indices = [p.index for p in pages]
    if indices != list(range(indices[0], indices[0] + len(indices))):
        warnings.append("This group combines pages that were not uploaded consecutively.")

    for page in pages:
        warnings.extend(f"{page.label}: {w}" for w in page.warnings)

    status = GroupStatus.isolated if reasons else GroupStatus.classified
    if status == GroupStatus.isolated and confidence == Confidence.high:
        confidence = Confidence.low

    return DocumentGroup(
        group_id=group_id,
        status=status,
        source_files=_source_files(pages),
        classification=classification,
        field_sources=sources,
        proposed_filename=(
            propose_filename(classification, schema, [mime_by_file[p.file_id] for p in pages])
            if status == GroupStatus.classified
            else None
        ),
        confidence=confidence,
        reason=" ".join(reason_texts) if reason_texts else None,
        isolation_reasons=reasons,
        evidence=_dedupe(evidence),
        warnings=_dedupe(warnings),
    )


def _isolation_code(raw: Optional[str]) -> IsolationReason:
    try:
        return IsolationReason(raw)
    except (ValueError, TypeError):
        return IsolationReason.insufficient_context


def _dedupe(items: List[str]) -> List[str]:
    return list(dict.fromkeys(i for i in items if i))


def isolated_without_plan(
    group_id: str,
    pages: Sequence[Page],
    context: SortingContext,
    reason_code: IsolationReason,
    reason: str,
) -> DocumentGroup:
    """Isolated group used when no trustworthy model plan exists for these pages."""
    return DocumentGroup(
        group_id=group_id,
        status=GroupStatus.isolated,
        source_files=_source_files(pages),
        classification={f: context.administrator_values.get(f) for f in context.schema_fields},
        field_sources={
            f: (FieldSource.administrator if f in context.administrator_values else None)
            for f in context.schema_fields
        },
        proposed_filename=None,
        confidence=Confidence.insufficient,
        reason=reason,
        isolation_reasons=[reason_code],
        evidence=[],
        warnings=[],
    )


# ============================================================
# DUPLICATES (content level)
# ============================================================

def annotate_duplicates(groups: List[DocumentGroup], pages: Sequence[Page], plan: Optional[GroupingPlan]) -> None:
    """Add warnings for duplicate pages and groups. Nothing is removed."""
    page_to_group: Dict[int, str] = {}
    by_file_page: Dict[Tuple[str, int], Page] = {(p.file_id, p.page_number): p for p in pages}
    for group in groups:
        for ref in group.source_files:
            for number in ref.page_numbers:
                page = by_file_page.get((ref.file_id, number))
                if page:
                    page_to_group[page.index] = group.group_id
    group_by_id = {g.group_id: g for g in groups}

    # Identical text or near-identical images between pages.
    for i, page in enumerate(pages):
        for earlier in pages[:i]:
            same_text = page.signals.text_fingerprint and page.signals.text_fingerprint == earlier.signals.text_fingerprint
            same_image = (
                page.signals.image_fingerprint is not None
                and earlier.signals.image_fingerprint is not None
                and hamming_distance(page.signals.image_fingerprint, earlier.signals.image_fingerprint) <= 3
                and not page.text and not earlier.text
            )
            if same_text or same_image:
                group = group_by_id.get(page_to_group.get(page.index))
                if group:
                    group.warnings.append(
                        f"{page.label} appears to duplicate {earlier.label} "
                        f"({page_to_group.get(earlier.index)})."
                    )
                break

    if plan:
        for rel in plan.page_relationships:
            if rel.relationship == "duplicate" and rel.related_page_index is not None:
                group = group_by_id.get(page_to_group.get(rel.page_index))
                other = page_to_group.get(rel.related_page_index)
                if group and other:
                    group.warnings.append(
                        f"Model judged page_index {rel.page_index} a duplicate of page_index "
                        f"{rel.related_page_index} ({other})."
                    )

    # Groups with identical classification.
    seen: Dict[Tuple, str] = {}
    for group in groups:
        if group.status != GroupStatus.classified:
            continue
        key = tuple(sorted((k, (v or "").casefold()) for k, v in group.classification.items()))
        if key in seen:
            group.possible_duplicate_of = seen[key]
            group.warnings.append(f"Possible duplicate of {seen[key]} (identical classification).")
        else:
            seen[key] = group.group_id

    for group in groups:
        group.warnings = _dedupe(group.warnings)
