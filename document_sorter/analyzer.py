import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field

from .models import (
    PageRepresentation,
    DocumentGroup,
    IsolatedGroup,
    SourceFileInfo,
)

logger = logging.getLogger("nebula_analyzer")


# ============================================================
# LLM STRUCTURED OUTPUT SCHEMAS
# ============================================================

class LLMGroupCandidate(BaseModel):
    page_indices: List[int] = Field(
        ...,
        description="Zero-based indices of pages that belong to this document group in chronological order",
    )
    group_type: str = Field(
        default="classified",
        description="'classified' if sufficient evidence, 'isolated' if ambiguous or missing critical information",
    )
    isolation_reason: Optional[str] = Field(
        default=None,
        description="Human-readable reason if marked isolated",
    )
    classification: Dict[str, Optional[str]] = Field(
        default_factory=dict,
        description="Extracted key-value pairs matching the requested schema",
    )
    confidence: str = Field(
        default="high",
        description="'high', 'medium', or 'low'",
    )
    evidence: List[str] = Field(
        default_factory=list,
        description="Evidence points explaining continuation grouping and metadata extraction",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Any warnings or ambiguities observed",
    )


class LLMSortingPlan(BaseModel):
    groups: List[LLMGroupCandidate] = Field(
        default_factory=list,
        description="List of grouped and classified documents",
    )


# ============================================================
# FILENAME SANITIZER & PROPOSAL
# ============================================================

def sanitize_slug(text: str) -> str:
    """Sanitize a text snippet for a clean filename component."""
    clean = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[-\s]+", "_", clean)


def generate_proposed_filename(
    classification: Dict[str, Optional[str]],
    schema: List[str],
    file_type: str,
) -> str:
    """
    Generate a clean proposed filename according to schema order and file extension.
    Example: past_questions-500lvl-science-computer_science-csc_308-2024_2025-first_semester.pdf
    """
    parts = []
    for field in schema:
        val = classification.get(field)
        if val and str(val).strip():
            slug = sanitize_slug(str(val))
            if slug:
                parts.append(slug)

    ext = ".pdf"
    if file_type == "image":
        ext = ".jpg"
    elif file_type == "docx":
        ext = ".docx"

    if not parts:
        return f"unclassified_document{ext}"

    return "-".join(parts) + ext


# ============================================================
# ALLOWED VALUES VALIDATION
# ============================================================

def validate_and_normalize_allowed_values(
    classification: Dict[str, Optional[str]],
    allowed_values: Optional[Dict[str, List[str]]],
) -> Tuple[Dict[str, Optional[str]], List[str], List[str]]:
    """
    Validate and normalize classification values against permitted allowed values.
    Returns (normalized_classification, validation_errors, warnings).
    """
    normalized = dict(classification)
    errors = []
    warnings = []

    if not allowed_values:
        return normalized, errors, warnings

    # Normalize allowed values keys (handle singular/plural e.g. level vs levels)
    lookup_map: Dict[str, List[str]] = {}
    for k, v in allowed_values.items():
        if isinstance(v, list) and v:
            clean_k = k.lower().strip()
            lookup_map[clean_k] = v
            # singular
            if clean_k.endswith("s"):
                lookup_map[clean_k[:-1]] = v

    for field, val in normalized.items():
        if not val:
            continue

        clean_field = field.lower().strip()
        allowed_list = lookup_map.get(clean_field) or lookup_map.get(clean_field + "s")

        if allowed_list:
            # Case-insensitive match or substring match
            matched = None
            val_lower = val.lower().strip()
            for allowed in allowed_list:
                allowed_str = str(allowed).strip()
                if allowed_str.lower() == val_lower:
                    matched = allowed_str
                    break

            if matched:
                normalized[field] = matched
            else:
                # Try fuzzy/contains match
                for allowed in allowed_list:
                    allowed_str = str(allowed).strip()
                    if allowed_str.lower() in val_lower or val_lower in allowed_str.lower():
                        matched = allowed_str
                        break

                if matched:
                    normalized[field] = matched
                    warnings.append(f"Value '{val}' for field '{field}' approximated to allowed value '{matched}'.")
                else:
                    errors.append(
                        f"Value '{val}' for field '{field}' is not in allowed list: {allowed_list}"
                    )

    return normalized, errors, warnings


# ============================================================
# DETERMINISTIC HEURISTIC GROUPER & CLASSIFIER
# ============================================================

def heuristic_group_pages(
    pages: List[PageRepresentation],
    schema: List[str],
    allowed_values: Optional[Dict[str, List[str]]] = None,
    additional_context: Optional[Dict[str, Any]] = None,
) -> Tuple[List[DocumentGroup], List[IsolatedGroup]]:
    """
    Deterministic rule-based grouping and classification fallback.
    Detects boundaries when a new distinct course code / explicit exam header appears.
    Keeps continuation pages with preceding document when question sequence continues or no header exists.
    """
    if not pages:
        return [], []

    # 1. BOUNDARY DETECTION & GROUPING
    raw_groups: List[List[PageRepresentation]] = []
    current_group: List[PageRepresentation] = [pages[0]]

    for i in range(1, len(pages)):
        prev_page = pages[i - 1]
        curr_page = pages[i]

        curr_codes = curr_page.course_codes
        prev_codes = prev_page.course_codes

        # Check if current page is explicitly a new document
        is_new_document = False

        if curr_codes and prev_codes:
            # If both have detected course codes and they are different, definite boundary
            if curr_codes[0] != prev_codes[0]:
                is_new_document = True
        elif curr_page.has_boundary_header and not prev_codes:
            # Explicit exam header appeared
            is_new_document = True
        elif curr_page.question_numbers and prev_page.question_numbers:
            # Check question progression
            curr_first_q = curr_page.question_numbers[0]
            prev_last_q = prev_page.question_numbers[-1]

            # If current page restarts from Question 1 (or <= prev_last_q) and has a distinct header
            if curr_first_q == 1 and prev_last_q > 1 and curr_page.has_boundary_header:
                is_new_document = True

        if is_new_document:
            raw_groups.append(current_group)
            current_group = [curr_page]
        else:
            current_group.append(curr_page)

    if current_group:
        raw_groups.append(current_group)

    # 2. METADATA EXTRACTION FOR EACH GROUP
    classified_groups: List[DocumentGroup] = []
    isolated_groups: List[IsolatedGroup] = []

    for g_idx, grp_pages in enumerate(raw_groups, start=1):
        group_id = f"group_{g_idx:03d}"

        # Combine source file info
        source_files_map: Dict[str, Dict[str, Any]] = {}
        for p in grp_pages:
            if p.file_id not in source_files_map:
                source_files_map[p.file_id] = {
                    "file_id": p.file_id,
                    "original_filename": p.original_filename,
                    "page_numbers": [],
                }
            source_files_map[p.file_id]["page_numbers"].append(p.page_number)

        source_files = [
            SourceFileInfo(**v) for v in source_files_map.values()
        ]

        # Aggregate text and metadata
        all_text = "\n".join(p.text_content for p in grp_pages)
        lower_text = all_text.lower()
        all_codes = []
        for p in grp_pages:
            for c in p.course_codes:
                if c not in all_codes:
                    all_codes.append(c)

        evidence = []
        warnings = []
        classification: Dict[str, Optional[str]] = {k: None for k in schema}

        # Apply additional_context (Admin corrections or external context)
        if additional_context:
            for k, v in additional_context.items():
                if k in classification and v:
                    classification[k] = str(v)
                    evidence.append(f"Admin context applied for '{k}': {v}")

        # Course extraction
        course_val = classification.get("course")
        if not course_val:
            if all_codes:
                course_val = all_codes[0]
                classification["course"] = course_val
                evidence.append(f"Identified course code '{course_val}' from document content")
            else:
                # Check for course code in raw text
                match = re.search(r"\b([A-Z]{2,4}\s*[-–]?\s*\d{3}[A-Za-z]?)\b", all_text)
                if match:
                    course_val = match.group(1).upper()
                    classification["course"] = course_val
                    evidence.append(f"Identified course code '{course_val}' via pattern matching")

        # Material Type extraction
        if "material_type" in schema and not classification.get("material_type"):
            if "past question" in lower_text or "examination" in lower_text or "exam" in lower_text:
                classification["material_type"] = "Past Questions"
                evidence.append("Identified 'Past Questions' from examination headers")
            elif "manual" in lower_text or "lab" in lower_text:
                classification["material_type"] = "Manuals / Lecture Notes"
            elif "lecture note" in lower_text or "note" in lower_text:
                classification["material_type"] = "Manuals / Lecture Notes"
            elif "workbook" in lower_text:
                classification["material_type"] = "Workbooks"
            else:
                classification["material_type"] = "Materials"

        # Level extraction
        if "level" in schema and not classification.get("level"):
            # Check 100-500 level
            level_match = re.search(r"\b([1-5]00)\s*(?:lvl|level)?\b", lower_text)
            if level_match:
                classification["level"] = level_match.group(1)
                evidence.append(f"Identified level '{level_match.group(1)}' from text")
            elif course_val:
                # Infer level from course number (e.g. CSC 308 -> 300)
                c_num_match = re.search(r"(\d)(\d{2})", course_val)
                if c_num_match:
                    inferred_level = f"{c_num_match.group(1)}00"
                    classification["level"] = inferred_level
                    evidence.append(f"Inferred level '{inferred_level}' from course code '{course_val}'")

        # Semester extraction
        if "semester" in schema and not classification.get("semester"):
            if "first semester" in lower_text or "1st semester" in lower_text or "harmattan" in lower_text:
                classification["semester"] = "First Semester"
                evidence.append("Identified First Semester")
            elif "second semester" in lower_text or "2nd semester" in lower_text or "rain semester" in lower_text:
                classification["semester"] = "Second Semester"
                evidence.append("Identified Second Semester")

        # Session extraction
        if "session" in schema and not classification.get("session"):
            session_match = re.search(r"\b(20\d{2}\s*[/–-]\s*20\d{2})\b", all_text)
            if session_match:
                clean_session = re.sub(r"\s*", "", session_match.group(1)).replace("–", "/").replace("-", "/")
                classification["session"] = clean_session
                evidence.append(f"Identified academic session '{clean_session}'")

        # Check continuation relationship evidence
        if len(grp_pages) > 1:
            evidence.append(
                f"Group contains {len(grp_pages)} pages: Continuation identified by question progression and absence of conflicting headers."
            )

        # Allowed values validation
        norm_class, val_errors, val_warns = validate_and_normalize_allowed_values(
            classification, allowed_values
        )
        classification = norm_class
        warnings.extend(val_warns)

        # Determine if isolated (Rule 14: Never guess if critical info is missing)
        primary_field = "course" if "course" in schema else schema[0]
        missing_primary = not classification.get(primary_field)

        if missing_primary or val_errors:
            iso_reason = (
                val_errors[0]
                if val_errors
                else f"Primary identifier '{primary_field}' could not be determined from the document or its neighboring pages."
            )
            isolated_groups.append(
                IsolatedGroup(
                    group_id=group_id,
                    source_files=source_files,
                    partial_classification=classification,
                    proposed_filename=generate_proposed_filename(
                        classification, schema, grp_pages[0].file_type
                    ),
                    status="isolated",
                    reason=iso_reason,
                    evidence=evidence,
                    warnings=warnings,
                )
            )
        else:
            proposed_fname = generate_proposed_filename(
                classification, schema, grp_pages[0].file_type
            )
            classified_groups.append(
                DocumentGroup(
                    group_id=group_id,
                    source_files=source_files,
                    classification=classification,
                    proposed_filename=proposed_fname,
                    status="classified",
                    confidence="high" if len(evidence) >= 2 else "medium",
                    evidence=evidence,
                    warnings=warnings,
                )
            )

    return classified_groups, isolated_groups


# ============================================================
# LLM-ASSISTED GROUPER & CLASSIFIER
# ============================================================

def analyze_and_group_pages(
    llm,
    pages: List[PageRepresentation],
    schema: List[str],
    allowed_values: Optional[Dict[str, List[str]]] = None,
    additional_context: Optional[Dict[str, Any]] = None,
) -> Tuple[List[DocumentGroup], List[IsolatedGroup]]:
    """
    Main entry point for page grouping and classification.
    Uses LLM structured reasoning with heuristic fallback.
    """
    if not pages:
        return [], []

    # Build page summaries for prompt
    page_summaries = []
    for p in pages:
        sample_text = p.text_content[:600].replace("\n", " ").strip()
        page_summaries.append(
            f"Page Index {p.page_index} (File: '{p.original_filename}', Page {p.page_number}):\n"
            f"  - Detected Course Codes: {p.course_codes}\n"
            f"  - Detected Question Numbers: {p.question_numbers}\n"
            f"  - Has Boundary Header: {p.has_boundary_header}\n"
            f"  - Sample Content: {sample_text}\n"
        )

    prompt = f"""You are Nebula's Document Understanding and Sorting Engine.

Your task is to analyze a batch of document pages, determine document boundaries and continuation pages, and extract metadata according to a requested dynamic schema.

============================================================
CRITICAL GROUPING & CONTINUATION RULES:
============================================================
1. DO NOT assume one file or page is one independent document.
2. CONTINUATION PAGES: Academic materials often have continuation pages without a course header.
   If Page A contains questions 1-4 and course EEE 306, and Page B has questions 5-6 with NO course header,
   Page A and Page B belong to the SAME document group (EEE 306).
3. DOCUMENT BOUNDARIES: A new course header or exam title indicates the start of a NEW document group.
4. ISOLATION RULE (DO NOT GUESS):
   If you cannot determine critical fields (e.g. course or document subject) from the page or its context,
   mark the group as isolated with a clear explanation in `isolation_reason`. DO NOT hallucinate.
5. ADMINISTRATOR CONTEXT:
   If additional context is provided (such as admin corrections), incorporate it.

============================================================
REQUESTED SCHEMA:
{schema}

ALLOWED VALUES:
{allowed_values or 'None specified (extract from content)'}

ADDITIONAL CONTEXT (ADMIN INPUT):
{additional_context or 'None'}

============================================================
PAGES IN CONTEXTUAL BATCH:
============================================================
{"".join(page_summaries)}
"""

    try:
        if llm:
            structured_llm = llm.with_structured_output(LLMSortingPlan)
            plan: LLMSortingPlan = structured_llm.invoke(prompt)

            if plan and plan.groups:
                classified_groups: List[DocumentGroup] = []
                isolated_groups: List[IsolatedGroup] = []

                for idx, g in enumerate(plan.groups, start=1):
                    group_id = f"group_{idx:03d}"
                    matching_pages = [p for p in pages if p.page_index in g.page_indices]
                    if not matching_pages:
                        continue

                    # Source files map
                    source_files_map: Dict[str, Dict[str, Any]] = {}
                    for p in matching_pages:
                        if p.file_id not in source_files_map:
                            source_files_map[p.file_id] = {
                                "file_id": p.file_id,
                                "original_filename": p.original_filename,
                                "page_numbers": [],
                            }
                        source_files_map[p.file_id]["page_numbers"].append(p.page_number)

                    source_files = [SourceFileInfo(**v) for v in source_files_map.values()]

                    # Ensure all schema keys exist in classification
                    classification = {k: g.classification.get(k) for k in schema}

                    # Merge admin context if missing
                    if additional_context:
                        for ak, av in additional_context.items():
                            if ak in classification and not classification[ak] and av:
                                classification[ak] = str(av)
                                g.evidence.append(f"Admin correction applied: {ak}={av}")

                    # Validate allowed values
                    norm_class, val_errors, val_warns = validate_and_normalize_allowed_values(
                        classification, allowed_values
                    )
                    classification = norm_class
                    warnings = list(g.warnings) + val_warns

                    is_isolated = g.group_type == "isolated" or bool(val_errors)
                    primary_key = "course" if "course" in schema else schema[0]
                    if not classification.get(primary_key):
                        is_isolated = True

                    proposed_fname = generate_proposed_filename(
                        classification, schema, matching_pages[0].file_type
                    )

                    if is_isolated:
                        iso_reason = (
                            g.isolation_reason
                            or (val_errors[0] if val_errors else f"Primary identifier '{primary_key}' could not be resolved.")
                        )
                        isolated_groups.append(
                            IsolatedGroup(
                                group_id=group_id,
                                source_files=source_files,
                                partial_classification=classification,
                                proposed_filename=proposed_fname,
                                status="isolated",
                                reason=iso_reason,
                                evidence=g.evidence,
                                warnings=warnings,
                            )
                        )
                    else:
                        classified_groups.append(
                            DocumentGroup(
                                group_id=group_id,
                                source_files=source_files,
                                classification=classification,
                                proposed_filename=proposed_fname,
                                status="classified",
                                confidence=g.confidence or "high",
                                evidence=g.evidence,
                                warnings=warnings,
                            )
                        )

                return classified_groups, isolated_groups

    except Exception as e:
        logger.warning(f"LLM page grouping failed or timed out ({e}); falling back to heuristic analysis.")

    # Fallback to deterministic heuristic grouper
    return heuristic_group_pages(
        pages=pages,
        schema=schema,
        allowed_values=allowed_values,
        additional_context=additional_context,
    )
