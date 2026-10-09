import hashlib
import io
import logging
import re
import zipfile
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

from PIL import Image
import pypdf
import docx

from .models import PageRepresentation, RejectedFile


logger = logging.getLogger("nebula_extractor")

# MIME / format definitions
SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
    ".webp": "image",
    ".docx": "docx",
}

# Regex patterns for academic document markers
COURSE_CODE_PATTERN = re.compile(
    r"\b([A-Z]{2,4}\s*[-–]?\s*\d{3}[A-Za-z]?)\b"
)

QUESTION_PATTERNS = [
    re.compile(r"(?:Question|QUESTION|Q\.?)\s*(\d+)", re.IGNORECASE),
    re.compile(r"^\s*(\d+)\s*[\.\)]\s+", re.MULTILINE),
]

HEADER_KEYWORDS = [
    "university", "polytechnic", "college", "faculty", "department",
    "school of", "examination", "semester", "continuous assessment",
    "course code", "course title", "matric no", "time allowed", "instructions"
]


def compute_file_hash(content: bytes) -> str:
    """Compute SHA256 hash of file content for cryptographic duplicate detection."""
    return hashlib.sha256(content).hexdigest()


def detect_file_type(filename: str, content: bytes) -> Tuple[Optional[str], Optional[str]]:
    """
    Validate and detect file format using file extension and magic byte verification.
    Returns (format_type, error_reason).
    """
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return None, f"Unsupported file extension '{ext}'. Supported: PDF, JPG, JPEG, PNG, WebP, DOCX."

    expected_type = SUPPORTED_EXTENSIONS[ext]

    if len(content) == 0:
        return None, "File is empty (0 bytes)."

    # Magic byte verification
    if expected_type == "pdf":
        if not content.startswith(b"%PDF"):
            return None, "File content is not a valid PDF document (missing %PDF magic header)."
    elif expected_type == "image":
        if ext in (".jpg", ".jpeg"):
            if not content.startswith(b"\xff\xd8\xff"):
                return None, "File content is not a valid JPEG image."
        elif ext == ".png":
            if not content.startswith(b"\x89PNG\r\n\x1a\n"):
                return None, "File content is not a valid PNG image."
        elif ext == ".webp":
            if not (content.startswith(b"RIFF") and b"WEBP" in content[:16]):
                return None, "File content is not a valid WebP image."
    elif expected_type == "docx":
        if not content.startswith(b"PK\x03\x04"):
            return None, "File content is not a valid DOCX package (missing ZIP magic header)."
        # Verify it contains word/document.xml
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                if "word/document.xml" not in z.namelist():
                    return None, "File is not a valid Word document (missing word/document.xml)."
        except Exception:
            return None, "Corrupted DOCX archive."

    return expected_type, None


def extract_questions_and_codes(text: str) -> Tuple[List[int], List[str], bool]:
    """
    Extract sequential question numbers, candidate course codes, and header presence.
    """
    # Course codes
    raw_codes = COURSE_CODE_PATTERN.findall(text)
    # Filter false positives like "ISO 9001", "UTF 8", etc.
    course_codes = []
    for code in raw_codes:
        normalized = re.sub(r"\s+", " ", code.strip().upper())
        if normalized not in course_codes:
            course_codes.append(normalized)

    # Questions
    found_questions = []
    for pat in QUESTION_PATTERNS:
        for match in pat.finditer(text):
            try:
                q_num = int(match.group(1))
                if 1 <= q_num <= 50 and q_num not in found_questions:
                    found_questions.append(q_num)
            except (ValueError, IndexError):
                pass
    found_questions.sort()

    # Header presence
    lower_text = text[:800].lower()
    has_header = False
    match_count = sum(1 for kw in HEADER_KEYWORDS if kw in lower_text)
    if match_count >= 2 or (course_codes and match_count >= 1):
        has_header = True

    return found_questions, course_codes, has_header


def run_ocr_on_image(image_bytes: bytes) -> Tuple[str, bool]:
    """
    Run Optical Character Recognition on image bytes using pytesseract.
    Returns (extracted_text, ocr_success).
    """
    import pytesseract
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode not in ("L", "RGB"):
            image = image.convert("RGB")
        text = pytesseract.image_to_string(image).strip()
        return text, True
    except pytesseract.TesseractNotFoundError:
        logger.warning("Tesseract not installed on system; skipping OCR extraction.")
        return "", False
    except Exception as e:
        logger.warning("OCR failed on image: %s", e)
        return "", False


def extract_pages_from_file(
    file_id: str,
    original_filename: str,
    content: bytes,
    start_global_index: int,
) -> Tuple[List[PageRepresentation], Optional[RejectedFile]]:
    """
    Extract page representations from a valid file.
    Returns (pages, rejected_file_if_any).
    """
    file_type, error = detect_file_type(original_filename, content)
    if error:
        return [], RejectedFile(
            file_id=file_id,
            original_filename=original_filename,
            reason=error,
        )

    file_hash = compute_file_hash(content)
    pages: List[PageRepresentation] = []

    try:
        if file_type == "pdf":
            reader = pypdf.PdfReader(io.BytesIO(content))
            num_pages = len(reader.pages)
            if num_pages == 0:
                return [], RejectedFile(
                    file_id=file_id,
                    original_filename=original_filename,
                    reason="PDF contains 0 pages.",
                )

            for p_idx, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                ocr_applied = False
                # If page is empty or too short, attempt OCR if pytesseract is available
                if len(page_text.strip()) < 20:
                    try:
                        # Extract images from page if any
                        for img_obj in page.images:
                            ocr_text, ocr_ok = run_ocr_on_image(img_obj.data)
                            if ocr_ok and ocr_text:
                                page_text += "\n" + ocr_text
                                ocr_applied = True
                    except Exception:
                        pass

                q_nums, c_codes, has_hdr = extract_questions_and_codes(page_text)
                first_lines = [line.strip() for line in page_text.splitlines() if line.strip()][:3]

                pages.append(
                    PageRepresentation(
                        page_index=start_global_index + len(pages),
                        file_id=file_id,
                        original_filename=original_filename,
                        file_type="pdf",
                        page_number=p_idx,
                        text_content=page_text.strip(),
                        ocr_applied=ocr_applied,
                        headers=first_lines,
                        question_numbers=q_nums,
                        course_codes=c_codes,
                        has_boundary_header=has_hdr,
                        file_hash=file_hash,
                    )
                )

        elif file_type == "image":
            ocr_text, ocr_applied = run_ocr_on_image(content)
            q_nums, c_codes, has_hdr = extract_questions_and_codes(ocr_text)
            first_lines = [line.strip() for line in ocr_text.splitlines() if line.strip()][:3]

            pages.append(
                PageRepresentation(
                    page_index=start_global_index + len(pages),
                    file_id=file_id,
                    original_filename=original_filename,
                    file_type="image",
                    page_number=1,
                    text_content=ocr_text.strip(),
                    ocr_applied=ocr_applied,
                    headers=first_lines,
                    question_numbers=q_nums,
                    course_codes=c_codes,
                    has_boundary_header=has_hdr,
                    file_hash=file_hash,
                )
            )

        elif file_type == "docx":
            doc = docx.Document(io.BytesIO(content))
            full_text_parts = []
            for p in doc.paragraphs:
                if p.text.strip():
                    full_text_parts.append(p.text.strip())
            for t in doc.tables:
                for row in t.rows:
                    row_text = " | ".join(c.text.strip() for c in row.cells if c.text.strip())
                    if row_text:
                        full_text_parts.append(row_text)

            docx_text = "\n".join(full_text_parts)
            q_nums, c_codes, has_hdr = extract_questions_and_codes(docx_text)
            first_lines = full_text_parts[:3]

            pages.append(
                PageRepresentation(
                    page_index=start_global_index + len(pages),
                    file_id=file_id,
                    original_filename=original_filename,
                    file_type="docx",
                    page_number=1,
                    text_content=docx_text.strip(),
                    ocr_applied=False,
                    headers=first_lines,
                    question_numbers=q_nums,
                    course_codes=c_codes,
                    has_boundary_header=has_hdr,
                    file_hash=file_hash,
                )
            )

        return pages, None

    except Exception as exc:
        logger.error(f"Failed to process file '{original_filename}': {exc}")
        return [], RejectedFile(
            file_id=file_id,
            original_filename=original_filename,
            reason=f"Processing error: {str(exc)}",
        )
