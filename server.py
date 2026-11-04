import hmac
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Any

# The sorting endpoint's multipart field is literally named "schema" (part of
# the client contract); pydantic warns that the name shadows BaseModel.schema.
warnings.filterwarnings("ignore", message=r'Field name "schema', category=UserWarning)

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, ValidationError

from main import run_research
from artist import (
    ArtistEngine,
    DrawRequest,
    DrawResponse,
    DrawingRecord,
    PatternRecord,
    PatternCreateRequest,
    ArtistError,
    DrawingNotFoundError,
    PatternNotFoundError,
    GeometricValidationError,
)
from memory import (
    get_memories,
    save_memory,
    delete_memory_by_id,
    delete_all_memories,
    search_memories,
)
from history import (
    get_user_conversations,
    get_conversation_history,
    delete_conversation,
    delete_user_conversations,
)
from document_sorter import (
    DocumentSortingEngine,
    DocumentSortingResponse,
    ErrorResponse,
    SortingError,
    SortingRequest,
    UploadInput,
)
from document_sorter.config import SorterSettings


logging.basicConfig(
    level=os.environ.get("NEBULA_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("nebula.api")

API_VERSION = "2.0.0"

app = FastAPI(
    title="Nebula API",
    description=(
        "Nebula - general-purpose AI research assistant with long-term memory, "
        "conversation history and a generic document understanding & sorting capability."
    ),
    version=API_VERSION,
)


# ============================================================
# CORS
# ============================================================
# CORS only decides which browser origins may call the API; it is not
# authentication. Server-to-server document sorting is protected by an API
# key. Extra origins can be added with NEBULA_CORS_ORIGINS (comma-separated).

DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://nova-flash-psi.vercel.app",
]


def _cors_origins() -> List[str]:
    extra = [
        origin.strip().rstrip("/")
        for origin in os.environ.get("NEBULA_CORS_ORIGINS", "").split(",")
        if origin.strip()
    ]
    return list(dict.fromkeys(DEFAULT_CORS_ORIGINS + extra))


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# TRUSTED-CLIENT AUTHENTICATION (DOCUMENT SORTING)
# ============================================================

SORTING_PATH = "/sort-documents"
SORTER_SETTINGS = SorterSettings.from_env()
# Allowance for multipart boundaries and JSON form fields on top of file bytes.
MULTIPART_OVERHEAD_BYTES = 4 * 1024 * 1024

_sorting_engine: Optional[DocumentSortingEngine] = None


def get_sorting_engine() -> DocumentSortingEngine:
    global _sorting_engine
    if _sorting_engine is None:
        _sorting_engine = DocumentSortingEngine(settings=SORTER_SETTINGS)
    return _sorting_engine


_artist_engine: Optional[ArtistEngine] = None


def get_artist_engine() -> ArtistEngine:
    global _artist_engine
    if _artist_engine is None:
        _artist_engine = ArtistEngine()
    return _artist_engine


def _presented_api_key(request: Request) -> Optional[str]:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("x-api-key")


def _error_response(
    status_code: int,
    code: str,
    message: str,
    request_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> JSONResponse:
    body = ErrorResponse(
        request_id=request_id,
        error={"code": code, "message": message, "details": details or {}},
    )
    return JSONResponse(status_code=status_code, content=body.model_dump())


@app.middleware("http")
async def guard_document_sorting(request: Request, call_next):
    """Authenticate and size-check sorting requests before the body is read."""
    if request.method == "POST" and request.url.path.rstrip("/") == SORTING_PATH:
        expected = os.environ.get("NEBULA_API_KEY", "")
        if not expected:
            logger.error("Document sorting called but NEBULA_API_KEY is not configured")
            return _error_response(503, "auth_not_configured", "Document sorting is not enabled on this server.")

        presented = _presented_api_key(request) or ""
        if not hmac.compare_digest(presented.encode(), expected.encode()):
            logger.warning("Rejected document-sorting request: invalid or missing API key")
            return _error_response(401, "unauthorized", "Missing or invalid API key.")

        length = request.headers.get("content-length")
        limit = SORTER_SETTINGS.max_request_bytes + MULTIPART_OVERHEAD_BYTES
        if length and length.isdigit() and int(length) > limit:
            return _error_response(
                413,
                "request_too_large",
                f"Request exceeds the maximum size of {SORTER_SETTINGS.max_request_bytes // (1024 * 1024)} MB.",
                details={"max_bytes": SORTER_SETTINGS.max_request_bytes},
            )
    return await call_next(request)


@app.exception_handler(SortingError)
async def sorting_error_handler(request: Request, exc: SortingError):
    request_id = getattr(request.state, "sorting_request_id", None)
    logger.warning("request_id=%s sorting_error code=%s", request_id, exc.code)
    return _error_response(exc.http_status, exc.code, exc.message, request_id, exc.details)


# ============================================================
# REQUEST & RESPONSE SCHEMAS
# ============================================================

class ResearchRequest(BaseModel):
    query: str
    user_id: str = "default"
    session_id: str = "default"


class ResearchResponse(BaseModel):
    response: str
    user_id: str
    session_id: str


class MemoryCreateRequest(BaseModel):
    user_id: str = "default"
    category: str = Field(..., description="e.g. identity, preference, project, skill, instruction, context")
    key: str = Field(..., description="e.g. name, favorite_language, current_project")
    value: str = Field(..., description="The information to remember")
    importance: int = Field(default=8, ge=1, le=10)


# ============================================================
# SYSTEM HEALTH ENDPOINTS
# ============================================================

@app.get("/")
def home():
    return {
        "status": "Nebula Agent Running",
        "service": "nebula",
        "version": API_VERSION,
    }


@app.get("/health")
def health():
    """Liveness only: confirms the API process is up. Makes no model calls."""
    return {
        "status": "ok",
        "service": "nebula",
    }


@app.get("/health/ready")
def readiness():
    """Dependency/configuration readiness. Still makes no model calls."""
    from document_sorter.extraction import ocr_available

    try:
        import pypdfium2  # noqa: F401
        pdf_rendering = True
    except Exception:
        pdf_rendering = False

    checks = {
        "llm_configured": bool(os.environ.get("OPENAI_API_KEY")),
        "document_sorting_auth_configured": bool(os.environ.get("NEBULA_API_KEY")),
        "pdf_rendering": pdf_rendering,
        "ocr": ocr_available(),
        "artist_engine": True,
    }
    return {
        "status": "ready" if checks["llm_configured"] else "degraded",
        "service": "nebula",
        "checks": checks,
    }


# ============================================================
# RESEARCH / AGENT INTERACTION ENDPOINT
# ============================================================

@app.post(
    "/research",
    response_model=ResearchResponse,
)
def research(request: ResearchRequest):
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Research query cannot be empty.",
        )

    clean_user_id = (request.user_id or "default").strip() or "default"
    clean_session_id = (request.session_id or "default").strip() or "default"

    try:
        response = run_research(
            query=request.query.strip(),
            user_id=clean_user_id,
            session_id=clean_session_id,
        )

        return {
            "response": response,
            "user_id": clean_user_id,
            "session_id": clean_session_id,
        }

    except Exception as e:
        logger.exception("Nebula research request failed")
        raise HTTPException(
            status_code=500,
            detail=f"Nebula encountered an error: {type(e).__name__}",
        )


# ============================================================
# LONG-TERM MEMORY ENDPOINTS
# ============================================================

@app.get("/memories")
def list_memories(
    user_id: str = Query("default", description="User identifier"),
    category: Optional[str] = Query(None, description="Filter by category"),
    q: Optional[str] = Query(None, description="Search query"),
):
    """Retrieve all persistent long-term memories for a user."""
    clean_user = (user_id or "default").strip() or "default"

    if q and q.strip():
        memories = search_memories(user_id=clean_user, query=q.strip())
        if category:
            memories = [m for m in memories if m["category"] == category.strip().lower()]
    else:
        memories = get_memories(user_id=clean_user, minimum_importance=1, category=category)

    return {
        "user_id": clean_user,
        "count": len(memories),
        "memories": memories,
    }


@app.post("/memories")
def create_memory(item: MemoryCreateRequest):
    """Explicitly add or update a long-term memory item."""
    clean_user = (item.user_id or "default").strip() or "default"

    try:
        saved = save_memory(
            user_id=clean_user,
            category=item.category,
            key=item.key,
            value=item.value,
            importance=item.importance,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "status": "success",
        "memory": saved,
    }


@app.delete("/memories/{memory_id}")
def delete_single_memory(memory_id: int):
    """Delete a specific long-term memory by ID."""
    deleted = delete_memory_by_id(memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found.")
    return {"status": "success", "deleted_id": memory_id}


@app.delete("/memories")
def clear_all_memories(user_id: str = Query("default")):
    """Delete all long-term memories for a user."""
    clean_user = (user_id or "default").strip() or "default"
    count = delete_all_memories(clean_user)
    return {"status": "success", "deleted_count": count}


# ============================================================
# CONVERSATION HISTORY ENDPOINTS
# ============================================================

@app.get("/conversations")
def list_conversations(user_id: str = Query("default")):
    """List all conversation sessions for a user."""
    clean_user = (user_id or "default").strip() or "default"
    conversations = get_user_conversations(clean_user)
    return {
        "user_id": clean_user,
        "count": len(conversations),
        "conversations": conversations,
    }


@app.get("/conversations/{session_id}/messages")
def get_session_messages(session_id: str, user_id: str = Query("default")):
    """Get message history for a specific conversation session."""
    clean_user = (user_id or "default").strip() or "default"
    messages = get_conversation_history(user_id=clean_user, session_id=session_id, limit=50)
    return {
        "user_id": clean_user,
        "session_id": session_id,
        "count": len(messages),
        "messages": messages,
    }


@app.delete("/conversations/{session_id}")
def delete_session(session_id: str, user_id: str = Query("default")):
    """Delete a conversation session and its messages."""
    clean_user = (user_id or "default").strip() or "default"
    delete_conversation(user_id=clean_user, session_id=session_id)
    return {
        "status": "success",
        "deleted_session_id": session_id,
    }


# ============================================================
# DOCUMENT SORTING ENDPOINT (TRUSTED SERVER-TO-SERVER CLIENTS)
# ============================================================

_ERROR_CODE_BY_FIELD = {
    "schema": "malformed_schema",
    "required_fields": "malformed_schema",
    "field_descriptions": "malformed_schema",
    "allowed_values": "invalid_allowed_values",
    "additional_context": "invalid_additional_context",
    "previous_result": "invalid_previous_result",
    "request_id": "invalid_request_id",
}


def _parse_json_field(name: str, raw: Optional[str], expected_type, required: bool = False):
    if raw is None or not raw.strip():
        if required:
            raise SortingError(
                f"The '{name}' form field is required.",
                code="missing_schema" if name == "schema" else "invalid_request",
                http_status=422,
            )
        return None
    error_code = _ERROR_CODE_BY_FIELD.get(name, "invalid_request")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        raise SortingError(f"The '{name}' form field must be valid JSON.", code=error_code, http_status=422)
    if not isinstance(value, expected_type):
        kind = "array" if expected_type is list else "object"
        raise SortingError(f"The '{name}' form field must be a JSON {kind}.", code=error_code, http_status=422)
    return value


@app.post(
    SORTING_PATH,
    response_model=DocumentSortingResponse,
    response_model_by_alias=True,
    responses={
        code: {"model": ErrorResponse}
        for code in (400, 401, 413, 422, 500, 502, 503, 504)
    },
    tags=["document-sorting"],
)
def sort_documents_endpoint(
    request: Request,
    files: List[UploadFile] = File(default=[]),
    schema_json: Optional[str] = Form(default=None, alias="schema", description='JSON array of field names, e.g. ["category","year"]'),
    allowed_values: Optional[str] = Form(default=None, description="JSON object: field -> permitted values"),
    additional_context: Optional[str] = Form(default=None, description="JSON object of externally supplied values"),
    required_fields: Optional[str] = Form(default=None, description="JSON array; defaults to every schema field"),
    field_descriptions: Optional[str] = Form(default=None, description="JSON object: field -> what the field means"),
    previous_result: Optional[str] = Form(default=None, description="JSON object: earlier result, for reprocessing"),
    request_id: Optional[str] = Form(default=None),
    task: Optional[str] = Form(default="document_sorting"),
):
    """
    Group, classify and validate a batch of documents against a client-supplied
    schema. Authenticate with `Authorization: Bearer <NEBULA_API_KEY>` (or the
    `X-API-Key` header). Full contract: docs/DOCUMENT_SORTING_API.md.
    """
    supplied_id = (request_id or "").strip() or None
    request.state.sorting_request_id = supplied_id

    if task and task.strip() != "document_sorting":
        raise SortingError(
            "Unsupported task; this endpoint only performs 'document_sorting'.",
            code="unsupported_task",
            http_status=422,
        )

    payload = {
        "task": "document_sorting",
        "request_id": supplied_id,
        "schema": _parse_json_field("schema", schema_json, list, required=True),
        "required_fields": _parse_json_field("required_fields", required_fields, list),
        "field_descriptions": _parse_json_field("field_descriptions", field_descriptions, dict) or {},
        "allowed_values": _parse_json_field("allowed_values", allowed_values, dict),
        "additional_context": _parse_json_field("additional_context", additional_context, dict),
        "previous_result": _parse_json_field("previous_result", previous_result, dict),
    }
    try:
        sorting_request = SortingRequest.model_validate(payload)
    except ValidationError as exc:
        errors = exc.errors(include_input=False, include_url=False)
        problems = [
            {"field": ".".join(str(p) for p in err["loc"]) or "request", "message": err["msg"]}
            for err in errors
        ]
        known_codes = set(_ERROR_CODE_BY_FIELD.values())
        code = next((err["type"] for err in errors if err["type"] in known_codes), None)
        if code is None:
            first_field = str(errors[0]["loc"][0]) if errors and errors[0]["loc"] else ""
            code = _ERROR_CODE_BY_FIELD.get(first_field, "invalid_request")
        raise SortingError("The sorting request is invalid.", code=code, http_status=422,
                           details={"errors": problems})

    engine = get_sorting_engine()
    sorting_request.request_id = sorting_request.request_id or engine.new_request_id()
    request.state.sorting_request_id = sorting_request.request_id

    uploads = [
        UploadInput(filename=upload.filename or "", stream=upload.file, content_type=upload.content_type)
        for upload in files
    ]
    try:
        return engine.sort(sorting_request, uploads)
    except SortingError:
        raise
    except Exception:
        logger.exception("request_id=%s sorting_internal_error", sorting_request.request_id)
        raise SortingError("Internal error while processing documents.", code="internal_error", http_status=500)


# ============================================================
# THE ARTIST: GEOMETRIC DRAWING ENGINE ENDPOINTS (ABILITY 0.01)
# ============================================================

@app.post(
    "/artist/draw",
    response_model=DrawResponse,
    tags=["artist"],
    summary="Construct a geometric drawing from instructions",
)
def create_drawing_endpoint(request: DrawRequest):
    """
    Submits a drawing prompt to Nebula Intelligence and The Artist.
    Calculates dynamic geometry, circles, curves, sharp strokes,
    and boolean operations, returning completed drawing metadata and URLs.
    """
    try:
        engine = get_artist_engine()
        return engine.draw(request)
    except GeometricValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.message)
    except ArtistError as exc:
        raise HTTPException(status_code=exc.http_status, detail=exc.message)
    except Exception as exc:
        logger.exception("The Artist failed to construct drawing: %s", exc)
        raise HTTPException(status_code=500, detail=f"Drawing failed: {str(exc)}")


@app.post(
    "/artist/draw-from-image",
    response_model=DrawResponse,
    tags=["artist"],
    summary="Give Nebula a picture and get it back drawn from circles, shapes and booleans (vector)",
)
async def draw_from_image_endpoint(
    image: UploadFile = File(..., description="PNG, JPG, WEBP or GIF"),
    prompt: str = Form("", description="Optional: what the picture is, used as the drawing's name"),
    style: str = Form("line", description="line | solid | color"),
    detail: str = Form("medium", description="low | medium | high"),
    width: int = Form(800, ge=100, le=4096),
    height: int = Form(600, ge=100, le=4096),
    format: str = Form("png"),
    run_async: bool = Form(False, alias="async"),
):
    """
    Nebula runs the picture through its image-processing tool, fits circles, rectangles, triangles and polygons to
    what it finds (holes are subtractions), and returns the result as a clear vector drawing (SVG, with a PNG).
    """
    data = await image.read()
    try:
        return get_artist_engine().draw_from_image(
            data, prompt=prompt, style=style, detail=detail, width=width, height=height, format=format, run_async=run_async,
        )
    except GeometricValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.message)
    except ArtistError as exc:
        raise HTTPException(status_code=exc.http_status, detail=exc.message)
    except Exception as exc:
        logger.exception("The Artist failed to draw an image: %s", exc)
        raise HTTPException(status_code=500, detail=f"Drawing failed: {str(exc)}")


@app.get(
    "/artist/drawings/{drawing_id}/status",
    response_model=DrawResponse,
    tags=["artist"],
    summary="Progress of a drawing started with async=true",
)
def drawing_status_endpoint(drawing_id: str):
    """Where a background drawing is: pending, rendering, completed or failed (with the reason)."""
    try:
        return get_artist_engine().status(drawing_id)
    except DrawingNotFoundError:
        raise HTTPException(status_code=404, detail=f"Drawing '{drawing_id}' not found")


@app.get(
    "/artist/drawings",
    response_model=List[DrawingRecord],
    tags=["artist"],
    summary="List completed drawings",
)
def list_drawings_endpoint(limit: int = Query(50, ge=1, le=200)):
    """Lists recent drawings created by The Artist."""
    engine = get_artist_engine()
    return engine.list_drawings(limit=limit)


@app.get(
    "/artist/drawings/{drawing_id}",
    response_model=DrawingRecord,
    tags=["artist"],
    summary="Get drawing details",
)
def get_drawing_endpoint(drawing_id: str):
    """Retrieves metadata and properties of a specific drawing."""
    engine = get_artist_engine()
    try:
        return engine.get_drawing(drawing_id)
    except DrawingNotFoundError:
        raise HTTPException(status_code=404, detail=f"Drawing '{drawing_id}' not found")


@app.get(
    "/artist/drawings/{drawing_id}/file",
    tags=["artist"],
    summary="Retrieve rendered image file (PNG or SVG)",
)
def get_drawing_file_endpoint(
    drawing_id: str,
    format: Optional[str] = Query(None, description="Preferred format: 'png' or 'svg'"),
):
    """Serves the rendered image file (PNG or SVG) for a drawing."""
    engine = get_artist_engine()
    try:
        record = engine.get_drawing(drawing_id)
    except DrawingNotFoundError:
        raise HTTPException(status_code=404, detail=f"Drawing '{drawing_id}' not found")

    target_format = (format or record.format).lower().strip()
    file_path = Path(record.storage_path)

    if target_format == "svg" and file_path.suffix.lower() != ".svg":
        svg_candidate = file_path.with_suffix(".svg")
        if svg_candidate.is_file():
            file_path = svg_candidate
    elif target_format == "png" and file_path.suffix.lower() != ".png":
        png_candidate = file_path.with_suffix(".png")
        if png_candidate.is_file():
            file_path = png_candidate

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Drawing file not found on disk")

    media_type = "image/svg+xml" if file_path.suffix.lower() == ".svg" else "image/png"
    return FileResponse(path=str(file_path), media_type=media_type, filename=file_path.name)


@app.get(
    "/artist/patterns",
    response_model=List[PatternRecord],
    tags=["artist"],
    summary="List reusable geometric patterns in Artist DB",
)
def list_patterns_endpoint(category: Optional[str] = Query(None)):
    """Lists reusable parameterized patterns stored in the Artist DB."""
    engine = get_artist_engine()
    return engine.list_patterns(category=category)


@app.get(
    "/artist/patterns/{pattern_id_or_name}",
    response_model=PatternRecord,
    tags=["artist"],
    summary="Get pattern definition",
)
def get_pattern_endpoint(pattern_id_or_name: str):
    """Retrieves a specific reusable pattern definition by ID or name."""
    engine = get_artist_engine()
    pattern = engine.get_pattern(pattern_id_or_name)
    if not pattern:
        raise HTTPException(status_code=404, detail=f"Pattern '{pattern_id_or_name}' not found")
    return pattern


@app.post(
    "/artist/patterns",
    response_model=PatternRecord,
    tags=["artist"],
    summary="Store or update a parameterized pattern",
)
def create_pattern_endpoint(request: PatternCreateRequest):
    """Saves a new parameterized pattern or updates existing pattern with incremented version."""
    engine = get_artist_engine()
    return engine.save_pattern(request)

