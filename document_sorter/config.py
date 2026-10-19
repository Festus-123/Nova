"""
Runtime settings for the document sorter, read from environment variables
so they can be tuned on Render without code changes.
"""

import os
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


MB = 1024 * 1024


@dataclass(frozen=True)
class SorterSettings:
    # Hard limit on the summed size of all uploaded files in one request.
    max_request_bytes: int = 100 * MB
    max_files: int = 200
    # Total pages (across all files) analysed in one request.
    max_pages: int = 250
    # Wall-clock budget for one sorting request.
    request_timeout_seconds: float = 300.0
    # Timeout for a single model call.
    model_timeout_seconds: float = 120.0
    # Extra attempts when the model returns malformed / invalid output.
    model_repair_attempts: int = 1
    # Pages sent to the model per page-analysis call.
    pages_per_analysis_call: int = 4
    # Page-analysis calls run in parallel, up to this many at once.
    analysis_concurrency: int = 4
    # Send page images to the model (vision). Disable for text-only models.
    vision_enabled: bool = True
    # Use Tesseract OCR as an additional text signal when it is installed.
    ocr_enabled: bool = True
    # A PDF page whose text layer is shorter than this is rendered to an image.
    min_text_chars_for_text_only: int = 200
    # Longest side, in pixels, of page images sent to the model.
    max_image_side: int = 1600
    jpeg_quality: int = 80
    # Model used for sorting; falls back to the general Nebula model.
    model_name: str = "gpt-5.4-nano"
    model_base_url: str = "https://www.routeapi.ai/v1"

    @classmethod
    def from_env(cls) -> "SorterSettings":
        return cls(
            max_request_bytes=_env_int("NEBULA_SORT_MAX_REQUEST_MB", 100) * MB,
            max_files=_env_int("NEBULA_SORT_MAX_FILES", 200),
            max_pages=_env_int("NEBULA_SORT_MAX_PAGES", 250),
            request_timeout_seconds=_env_float("NEBULA_SORT_TIMEOUT_SECONDS", 300.0),
            model_timeout_seconds=_env_float("NEBULA_SORT_MODEL_TIMEOUT_SECONDS", 120.0),
            model_repair_attempts=_env_int("NEBULA_SORT_MODEL_REPAIR_ATTEMPTS", 1),
            pages_per_analysis_call=max(1, _env_int("NEBULA_SORT_PAGES_PER_CALL", 4)),
            analysis_concurrency=max(1, _env_int("NEBULA_SORT_CONCURRENCY", 4)),
            vision_enabled=_env_bool("NEBULA_SORT_VISION", True),
            ocr_enabled=_env_bool("NEBULA_SORT_OCR", True),
            min_text_chars_for_text_only=_env_int("NEBULA_SORT_MIN_TEXT_CHARS", 200),
            max_image_side=_env_int("NEBULA_SORT_MAX_IMAGE_SIDE", 1600),
            jpeg_quality=_env_int("NEBULA_SORT_JPEG_QUALITY", 80),
            model_name=os.environ.get("NEBULA_SORTER_MODEL")
            or os.environ.get("NEBULA_MODEL")
            or "gpt-5.4-nano",
            model_base_url=os.environ.get("NEBULA_LLM_BASE_URL") or "https://www.routeapi.ai/v1",
        )
