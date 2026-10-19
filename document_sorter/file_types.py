"""
Content-based file type detection.

The detected type is decided from the file's bytes (magic numbers and, for
DOCX, the package structure). The filename extension and the declared
Content-Type are only cross-checked and produce warnings on mismatch.
"""

import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

PDF_MIME = "application/pdf"
JPEG_MIME = "image/jpeg"
PNG_MIME = "image/png"
WEBP_MIME = "image/webp"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

SUPPORTED_MIME_TYPES = {
    PDF_MIME: "pdf",
    JPEG_MIME: "image",
    PNG_MIME: "image",
    WEBP_MIME: "image",
    DOCX_MIME: "docx",
}

# Declared content types accepted as equivalent to the detected type.
MIME_ALIASES = {
    "image/jpg": JPEG_MIME,
    "image/pjpeg": JPEG_MIME,
    "application/x-pdf": PDF_MIME,
}

EXTENSION_TO_MIME = {
    ".pdf": PDF_MIME,
    ".jpg": JPEG_MIME,
    ".jpeg": JPEG_MIME,
    ".png": PNG_MIME,
    ".webp": WEBP_MIME,
    ".docx": DOCX_MIME,
}

MIME_TO_EXTENSION = {
    PDF_MIME: ".pdf",
    JPEG_MIME: ".jpg",
    PNG_MIME: ".png",
    WEBP_MIME: ".webp",
    DOCX_MIME: ".docx",
}

# Guard against zip bombs in DOCX packages.
MAX_DOCX_UNCOMPRESSED_BYTES = 300 * 1024 * 1024


def sniff_mime(path: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (mime_type, problem). mime_type is None when the content is not a
    supported type; problem then explains why.
    """
    with open(path, "rb") as handle:
        head = handle.read(2048)

    if not head:
        return None, "empty"

    # PDF headers may be preceded by a little junk; the spec allows 1024 bytes.
    if b"%PDF-" in head[:1024]:
        return PDF_MIME, None
    if head.startswith(b"\xff\xd8\xff"):
        return JPEG_MIME, None
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return PNG_MIME, None
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return WEBP_MIME, None
    if head.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(path) as archive:
                names = set(archive.namelist())
                if "word/document.xml" not in names or "[Content_Types].xml" not in names:
                    return None, "ZIP archive that is not a Word (.docx) document"
                total = sum(info.file_size for info in archive.infolist())
                if total > MAX_DOCX_UNCOMPRESSED_BYTES:
                    return None, "DOCX package expands beyond the permitted size"
        except zipfile.BadZipFile:
            return None, "corrupted ZIP/DOCX package"
        return DOCX_MIME, None
    if head[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        return None, "legacy binary Office format (.doc) is not supported; convert to DOCX or PDF"
    return None, "content does not match any supported type"


def cross_check(original_filename: str, declared_content_type: Optional[str], detected_mime: str) -> List[str]:
    """Warnings when the extension or declared Content-Type disagree with the content."""
    warnings = []
    ext = Path(original_filename or "").suffix.lower()
    ext_mime = EXTENSION_TO_MIME.get(ext)
    if ext and ext_mime and ext_mime != detected_mime:
        warnings.append(
            f"File extension '{ext}' does not match its content ({detected_mime}); content type was used."
        )
    declared = (declared_content_type or "").split(";")[0].strip().lower()
    declared = MIME_ALIASES.get(declared, declared)
    if declared and declared not in ("application/octet-stream", "binary/octet-stream") and declared != detected_mime:
        warnings.append(
            f"Declared Content-Type '{declared}' does not match detected content ({detected_mime})."
        )
    return warnings
