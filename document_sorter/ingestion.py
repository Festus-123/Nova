"""
File ingestion: stream uploads into a per-request temporary workspace while
enforcing the total size limit, hashing as we go.

Nothing is kept after the request: the workspace directory is removed when the
context manager exits, including when processing fails.
"""

import hashlib
import logging
import re
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, List, Optional

from .errors import NoFilesError, RequestTooLargeError, StorageError
from .pages import IngestedFile

logger = logging.getLogger("nebula.sorter.ingestion")

CHUNK_SIZE = 1024 * 1024


@dataclass
class UploadInput:
    """Transport-independent description of one uploaded file."""

    filename: str
    stream: BinaryIO
    content_type: Optional[str] = None


@contextmanager
def temporary_workspace(request_id: str) -> Iterator[Path]:
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", request_id)[:40]
    try:
        path = Path(tempfile.mkdtemp(prefix=f"nebula-sort-{safe}-"))
    except OSError as exc:
        raise StorageError("Could not create temporary storage for processing.") from exc
    try:
        yield path
    finally:
        try:
            shutil.rmtree(path)
        except Exception as exc:  # pragma: no cover - logged, never raised
            logger.error("request_id=%s cleanup_failed path=%s error=%s", request_id, path, type(exc).__name__)


def ingest_uploads(
    uploads: List[UploadInput],
    workspace: Path,
    max_total_bytes: int,
    max_files: int,
) -> List[IngestedFile]:
    if not uploads:
        raise NoFilesError("No files were uploaded. Send one or more files in the 'files' field.")
    if len(uploads) > max_files:
        raise RequestTooLargeError(
            f"Too many files: {len(uploads)} (maximum {max_files}).",
            code="too_many_files",
        )

    ingested: List[IngestedFile] = []
    total = 0
    for position, upload in enumerate(uploads, start=1):
        file_id = f"file_{position:03d}"
        original = _clean_filename(upload.filename) or f"upload_{position}"
        target = workspace / f"{file_id}.bin"
        digest = hashlib.sha256()
        size = 0
        try:
            with open(target, "wb") as out:
                while True:
                    chunk = upload.stream.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    size += len(chunk)
                    total += len(chunk)
                    if total > max_total_bytes:
                        raise RequestTooLargeError(
                            f"Uploaded files exceed the maximum total size of "
                            f"{max_total_bytes // (1024 * 1024)} MB.",
                            details={"max_bytes": max_total_bytes},
                        )
                    digest.update(chunk)
                    out.write(chunk)
        except RequestTooLargeError:
            raise
        except OSError as exc:
            raise StorageError("Could not write an uploaded file to temporary storage.") from exc

        ingested.append(
            IngestedFile(
                file_id=file_id,
                original_filename=original,
                declared_content_type=upload.content_type,
                path=target,
                size=size,
                sha256=digest.hexdigest(),
                extension=Path(original).suffix.lower(),
            )
        )
    return ingested


def _clean_filename(name: Optional[str]) -> str:
    """Keep only the base name a client sent (never a path) for reporting."""
    if not name:
        return ""
    base = re.split(r"[\\/]", name)[-1]
    return base.strip()[:255]
