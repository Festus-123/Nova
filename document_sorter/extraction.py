"""
Content extraction: turn validated files into Page objects.

Strategy per type:
  PDF   - text layer via pypdf; pages with little/no text are rendered to an
          image (pypdfium2) for vision analysis, with optional OCR. Document
          metadata (title/subject) is kept as a weak signal.
  Image - EXIF-orientation-corrected, downscaled JPEG for vision analysis,
          plus optional OCR text.
  DOCX  - paragraphs, tables and header/footer text in body order, split on
          explicit page breaks; embedded images are attached when the text is
          too sparse to understand the document.

Failures specific to one file are reported as rejections, never as crashes.
"""

import io
import logging
import os
import shutil
from dataclasses import dataclass
from typing import List, Optional

from PIL import Image, ImageOps

from .config import SorterSettings
from .models import RejectionReason
from .pages import IngestedFile, Page, compute_signals

logger = logging.getLogger("nebula.sorter.extraction")

# Refuse absurd pixel counts (decompression bombs) instead of exhausting memory.
Image.MAX_IMAGE_PIXELS = 120_000_000


class FileRejected(Exception):
    def __init__(self, code: RejectionReason, reason: str):
        super().__init__(reason)
        self.code = code
        self.reason = reason


@dataclass
class ExtractionResult:
    pages: List[Page]
    warnings: List[str]


# ------------------------------------------------------------
# Image helpers
# ------------------------------------------------------------

def _to_model_jpeg(image: Image.Image, settings: SorterSettings) -> bytes:
    image = ImageOps.exif_transpose(image)
    if image.mode not in ("RGB", "L"):
        background = Image.new("RGB", image.size, "white")
        if image.mode in ("RGBA", "LA", "P"):
            rgba = image.convert("RGBA")
            background.paste(rgba, mask=rgba.split()[-1])
            image = background
        else:
            image = image.convert("RGB")
    image.thumbnail((settings.max_image_side, settings.max_image_side), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=settings.jpeg_quality, optimize=True)
    return buffer.getvalue()


_tesseract_checked: Optional[bool] = None


def ocr_available() -> bool:
    global _tesseract_checked
    if _tesseract_checked is None:
        try:
            import pytesseract

            candidates = [
                os.environ.get("TESSERACT_CMD", ""),
                shutil.which("tesseract") or "",
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            ]
            found = next((c for c in candidates if c and os.path.isfile(c)), None)
            if found:
                pytesseract.pytesseract.tesseract_cmd = found
                pytesseract.get_tesseract_version()
            _tesseract_checked = bool(found)
        except Exception:
            _tesseract_checked = False
    return bool(_tesseract_checked)


def _ocr(jpeg_bytes: bytes) -> str:
    import pytesseract

    try:
        with Image.open(io.BytesIO(jpeg_bytes)) as img:
            return pytesseract.image_to_string(img, timeout=60).strip()
    except Exception as exc:
        logger.warning("ocr_failed error=%s", type(exc).__name__)
        return ""


# ------------------------------------------------------------
# Per-type extraction
# ------------------------------------------------------------

def extract_file(file: IngestedFile, start_index: int, settings: SorterSettings) -> ExtractionResult:
    if file.kind == "pdf":
        pages = _extract_pdf(file, start_index, settings)
    elif file.kind == "image":
        pages = _extract_image(file, start_index, settings)
    elif file.kind == "docx":
        pages = _extract_docx(file, start_index, settings)
    else:  # pragma: no cover - guarded by validation
        raise FileRejected(RejectionReason.unsupported_type, "Unsupported file type.")

    for page in pages:
        page.signals = compute_signals(page)
    return ExtractionResult(pages=pages, warnings=[])


def _extract_image(file: IngestedFile, start_index: int, settings: SorterSettings) -> List[Page]:
    try:
        with Image.open(file.path) as img:
            img.load()
            jpeg = _to_model_jpeg(img, settings)
    except Image.DecompressionBombError:
        raise FileRejected(RejectionReason.corrupted_file, "Image dimensions are too large to process safely.")
    except Exception:
        raise FileRejected(RejectionReason.corrupted_file, "Image could not be decoded (corrupted or truncated).")

    page = Page(
        index=start_index,
        file_id=file.file_id,
        original_filename=file.original_filename,
        file_kind="image",
        page_number=1,
        images=[jpeg],
    )
    if settings.ocr_enabled and ocr_available():
        text = _ocr(jpeg)
        if text:
            page.text, page.text_source = text, "ocr"
    return [page]


def _extract_pdf(file: IngestedFile, start_index: int, settings: SorterSettings) -> List[Page]:
    import pypdf

    try:
        reader = pypdf.PdfReader(str(file.path))
        if reader.is_encrypted:
            try:
                if not reader.decrypt(""):
                    raise FileRejected(RejectionReason.encrypted_file, "PDF is password-protected.")
            except FileRejected:
                raise
            except Exception:
                raise FileRejected(RejectionReason.encrypted_file, "PDF is encrypted and cannot be opened.")
        page_count = len(reader.pages)
    except FileRejected:
        raise
    except Exception:
        raise FileRejected(RejectionReason.corrupted_file, "PDF could not be parsed (corrupted or malformed).")

    if page_count == 0:
        raise FileRejected(RejectionReason.empty_file, "PDF contains no pages.")

    metadata = {}
    try:
        info = reader.metadata or {}
        for key, name in (("/Title", "title"), ("/Subject", "subject")):
            value = info.get(key)
            if value and str(value).strip():
                metadata[name] = str(value).strip()[:200]
    except Exception:
        pass

    renderer = _PdfRenderer(file) if settings.vision_enabled or settings.ocr_enabled else None
    pages: List[Page] = []
    try:
        for number in range(1, page_count + 1):
            page = Page(
                index=start_index + len(pages),
                file_id=file.file_id,
                original_filename=file.original_filename,
                file_kind="pdf",
                page_number=number,
                file_metadata=metadata if number == 1 else {},
            )
            try:
                text = reader.pages[number - 1].extract_text() or ""
            except Exception:
                text = ""
                page.warnings.append("Text layer could not be read.")
            if text.strip():
                page.text, page.text_source = text.strip(), "text_layer"

            if len(page.text) < settings.min_text_chars_for_text_only and renderer:
                image = renderer.render(number - 1, reader, settings)
                if image:
                    if settings.vision_enabled:
                        page.images.append(image)
                    if not page.text and settings.ocr_enabled and ocr_available():
                        ocr_text = _ocr(image)
                        if ocr_text:
                            page.text, page.text_source = ocr_text, "ocr"
                else:
                    page.warnings.append("Page could not be rendered for visual analysis.")
            pages.append(page)
    finally:
        if renderer:
            renderer.close()
    return pages


class _PdfRenderer:
    """Renders PDF pages with pypdfium2; falls back to the largest embedded image."""

    def __init__(self, file: IngestedFile):
        self._doc = None
        try:
            import pypdfium2 as pdfium

            self._doc = pdfium.PdfDocument(str(file.path))
        except Exception as exc:
            logger.info("pdf_renderer_unavailable error=%s", type(exc).__name__)

    def render(self, page_index: int, reader, settings: SorterSettings) -> Optional[bytes]:
        if self._doc is not None:
            try:
                page = self._doc[page_index]
                width, height = page.get_size()
                scale = min(settings.max_image_side / max(width, height, 1), 3.0)
                bitmap = page.render(scale=scale)
                image = bitmap.to_pil()
                return _to_model_jpeg(image, settings)
            except Exception as exc:
                logger.info("pdf_render_failed page=%s error=%s", page_index + 1, type(exc).__name__)
        try:
            images = list(reader.pages[page_index].images)
            if images:
                largest = max(images, key=lambda im: len(im.data))
                with Image.open(io.BytesIO(largest.data)) as img:
                    img.load()
                    return _to_model_jpeg(img, settings)
        except Exception:
            pass
        return None

    def close(self):
        if self._doc is not None:
            try:
                self._doc.close()
            except Exception:
                pass


_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _extract_docx(file: IngestedFile, start_index: int, settings: SorterSettings) -> List[Page]:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        document = docx.Document(str(file.path))
    except Exception:
        raise FileRejected(RejectionReason.corrupted_file, "DOCX document could not be opened (corrupted).")

    # Header/footer text often carries the institution and course information.
    header_lines: List[str] = []
    try:
        for section in document.sections[:1]:
            for part in (section.header, section.first_page_header, section.footer):
                for paragraph in part.paragraphs:
                    if paragraph.text.strip() and paragraph.text.strip() not in header_lines:
                        header_lines.append(paragraph.text.strip())
    except Exception:
        pass

    sections: List[List[str]] = [[]]
    body = document.element.body
    for child in body.iterchildren():
        if child.tag == _W_NS + "p":
            paragraph = Paragraph(child, document)
            breaks_before = child.findall(f".//{_W_NS}br")
            has_page_break = any(br.get(_W_NS + "type") == "page" for br in breaks_before)
            text = paragraph.text.strip()
            if text:
                sections[-1].append(text)
            if has_page_break:
                sections.append([])
        elif child.tag == _W_NS + "tbl":
            table = Table(child, document)
            for row in table.rows:
                cells = []
                for cell in row.cells:
                    value = cell.text.strip()
                    if value and value not in cells:
                        cells.append(value)
                if cells:
                    sections[-1].append(" | ".join(cells))

    sections = [s for s in sections if s] or [[]]

    embedded: List[bytes] = []
    total_chars = sum(len(" ".join(s)) for s in sections)
    if settings.vision_enabled and total_chars < settings.min_text_chars_for_text_only:
        for rel in document.part.rels.values():
            if "image" in rel.reltype and len(embedded) < 3:
                try:
                    with Image.open(io.BytesIO(rel.target_part.blob)) as img:
                        img.load()
                        embedded.append(_to_model_jpeg(img, settings))
                except Exception:
                    continue

    pages: List[Page] = []
    for number, lines in enumerate(sections, start=1):
        text_parts = []
        if header_lines:
            text_parts.append("[Document header/footer] " + " / ".join(header_lines))
        text_parts.extend(lines)
        page = Page(
            index=start_index + len(pages),
            file_id=file.file_id,
            original_filename=file.original_filename,
            file_kind="docx",
            page_number=number,
            text="\n".join(text_parts).strip(),
            text_source="docx",
        )
        if number == 1 and embedded:
            page.images = embedded
        pages.append(page)

    if not any(p.text or p.images for p in pages):
        raise FileRejected(RejectionReason.unreadable_content, "DOCX document contains no text or images.")
    return pages
