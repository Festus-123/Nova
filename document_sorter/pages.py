"""
Internal representation of ingested files and extracted pages.

A *page* is the unit the sorter reasons about: one PDF page, one image, or one
page-break-delimited section of a DOCX. Pages carry their original file
identity at all times so no grouping can lose track of where content came from.
"""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class IngestedFile:
    file_id: str
    original_filename: str
    declared_content_type: Optional[str]
    path: Path
    size: int
    sha256: str
    kind: Optional[str] = None          # "pdf" | "image" | "docx"
    mime_type: Optional[str] = None     # detected from content
    extension: str = ""


@dataclass
class PageSignals:
    """Cheap deterministic signals computed from extracted text/images."""

    char_count: int = 0
    question_numbers: List[int] = field(default_factory=list)
    printed_page_number: Optional[int] = None
    text_fingerprint: Optional[str] = None
    image_fingerprint: Optional[int] = None


@dataclass
class Page:
    index: int                      # global, 0-based, stable within one request
    file_id: str
    original_filename: str
    file_kind: str
    page_number: int                # 1-based within its file
    text: str = ""
    text_source: str = "none"       # "text_layer" | "ocr" | "docx" | "none"
    images: List[bytes] = field(default_factory=list)   # JPEG bytes for vision
    file_metadata: dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    signals: PageSignals = field(default_factory=PageSignals)

    @property
    def label(self) -> str:
        return f"page {self.page_number} of '{self.original_filename}'"


# ------------------------------------------------------------
# Deterministic signals
# ------------------------------------------------------------

_QUESTION_PATTERNS = [
    re.compile(r"\b(?:question|q)\s*\.?\s*(\d{1,3})\b", re.IGNORECASE),
    re.compile(r"^\s*(\d{1,3})\s*[\.\)]\s+\S", re.MULTILINE),
]
_PAGE_NUMBER_PATTERNS = [
    re.compile(r"\bpage\s+(\d{1,4})(?:\s+of\s+\d{1,4})?\b", re.IGNORECASE),
    re.compile(r"^\s*-?\s*(\d{1,4})\s*-?\s*$"),
]


def compute_signals(page: Page) -> PageSignals:
    text = page.text or ""
    questions: List[int] = []
    for pattern in _QUESTION_PATTERNS:
        for match in pattern.finditer(text):
            number = int(match.group(1))
            if 0 < number <= 200 and number not in questions:
                questions.append(number)
    questions.sort()

    printed = None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidates = lines[-3:] + lines[:2]
    for line in candidates:
        for pattern in _PAGE_NUMBER_PATTERNS:
            match = pattern.search(line)
            if match:
                printed = int(match.group(1))
                break
        if printed is not None:
            break

    normalized = re.sub(r"\W+", " ", text.lower()).strip()
    text_fp = hashlib.sha256(normalized.encode("utf-8")).hexdigest() if len(normalized) >= 80 else None

    image_fp = None
    if page.images:
        image_fp = image_difference_hash(page.images[0])

    return PageSignals(
        char_count=len(text.strip()),
        question_numbers=questions,
        printed_page_number=printed,
        text_fingerprint=text_fp,
        image_fingerprint=image_fp,
    )


def image_difference_hash(jpeg_bytes: bytes, size: int = 8) -> Optional[int]:
    """64-bit dHash, robust to re-encoding/resizing; used for near-duplicate pages."""
    import io

    from PIL import Image

    try:
        with Image.open(io.BytesIO(jpeg_bytes)) as img:
            gray = img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
            pixels = list(gray.tobytes())
    except Exception:
        return None
    value = 0
    for row in range(size):
        for col in range(size):
            left = pixels[row * (size + 1) + col]
            right = pixels[row * (size + 1) + col + 1]
            value = (value << 1) | (1 if left > right else 0)
    return value


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")
