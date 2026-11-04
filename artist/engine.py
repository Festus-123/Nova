"""
The Artist: Geometric Drawing Engine Orchestrator (Ability 0.01).

Coordinates Nebula Intelligence, Canvas Grid, Geometry & Boolean Engine,
Pattern Database, and Image Renderers.
"""

import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List

from .config import ARTIST_SETTINGS, ArtistSettings
from .errors import DrawingNotFoundError, RenderingError, GeometricValidationError
from .models import (
    DrawRequest,
    DrawResponse,
    DrawingRecord,
    DrawingStatus,
    RenderFormat,
    PatternRecord,
    PatternCreateRequest,
)
from .db import ArtistDB
from .nebula_planner import NebulaPlanner
from .renderer import SVGRenderer, PNGRenderer
from .construct import Constructor
from .llm_planner import ConstructionPlanner, PlanningUnavailableError
from .vector_render import VectorPNG, VectorSVG
from .vision import ImageVectorizer

# Drawings that take a while (a language model is involved) run here when the caller asks for a background job.
_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="artist")


def _style(value: Optional[str]) -> str:
    return value if value in ("line", "solid", "color") else "line"


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_\-]+", "_", text.strip().lower())
    return slug[:40].strip("_") or "drawing"


class ArtistEngine:
    """The central engine for Ability 0.01: The Artist."""

    def __init__(self, settings: Optional[ArtistSettings] = None, db: Optional[ArtistDB] = None):
        self.settings = settings or ARTIST_SETTINGS
        self.db = db or ArtistDB(db_path=self.settings.db_path)
        self.planner = NebulaPlanner(db=self.db)  # the original stored-pattern planner (developer fallback)
        self.constructor_planner = ConstructionPlanner()
        self._ensure_storage()

    def _ensure_storage(self) -> None:
        self.settings.drawings_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------------------------------------------

    def draw(self, request: DrawRequest) -> DrawResponse:
        """
        Draws from a prompt. With `async` set it returns at once (status 'pending') and keeps working in the
        background; poll get_drawing(). Otherwise it returns when the drawing is finished.
        """
        prompt = request.prompt.strip()
        if not prompt:
            raise GeometricValidationError("Drawing prompt cannot be empty")
        job = dict(
            kind="prompt",
            prompt=prompt,
            width=request.width or self.settings.default_width,
            height=request.height or self.settings.default_height,
            format=(request.format or self.settings.default_format).lower().strip(),
            style=_style(request.style),
            quality=(request.quality or "balanced") if (request.quality or "balanced") in ("fast", "balanced", "best") else "balanced",
            parameters=request.parameters,
        )
        return self._submit(job, bool(request.run_async))

    def draw_from_image(
        self,
        data: bytes,
        prompt: str = "",
        style: str = "line",
        detail: str = "medium",
        width: int = 800,
        height: int = 600,
        format: str = "png",
        run_async: bool = False,
    ) -> DrawResponse:
        """Draws a picture Nebula was given: image processing, then circles / shapes / booleans, then vector art."""
        if not data:
            raise GeometricValidationError("No image was received")
        job = dict(
            kind="image",
            prompt=(prompt or "").strip() or "Drawing from an image",
            image=data,
            width=int(width),
            height=int(height),
            format=(format or "png").lower().strip(),
            style=_style(style),
            detail=detail,
        )
        return self._submit(job, run_async)

    def _submit(self, job: Dict[str, Any], run_async: bool) -> DrawResponse:
        drawing_id = f"draw-{uuid.uuid4().hex[:12]}"
        job["drawing_id"] = drawing_id
        slug = _slugify(job["prompt"])
        job["base"] = f"{slug}_{drawing_id[:8]}"
        record = DrawingRecord(
            drawing_id=drawing_id,
            prompt=job["prompt"],
            filename="",
            format=job["format"] if job["format"] in ("png", "svg") else "png",
            width=job["width"],
            height=job["height"],
            storage_path="",
            status=DrawingStatus.PENDING.value,
            metadata={"style": job["style"], "source": job["kind"], "stage": "queued", "timeline": []},
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self.db.save_drawing(record)
        if run_async:
            _POOL.submit(self._run_job, job, record, False)
            return self._response(record)
        return self._run_job(job, record, True)

    def _progress(self, record: DrawingRecord, stage: str, **extra: Any) -> None:
        record.status = DrawingStatus.RENDERING.value if stage in ("rendering", "saving") else DrawingStatus.PENDING.value
        record.metadata = {**record.metadata, "stage": stage, **extra}
        self.db.save_drawing(record)

    def _run_job(self, job: Dict[str, Any], record: DrawingRecord, raise_errors: bool) -> DrawResponse:
        """The whole pipeline for one drawing. Never lets an exception escape in background mode."""
        t0 = time.time()
        try:
            width, height, style = job["width"], job["height"], job["style"]
            meta = record.metadata
            if job["kind"] == "image":
                self._progress(record, "processing the image")
                program = ImageVectorizer(width, height, style, job.get("detail", "medium")).vectorize(job["image"])
                self._progress(record, "constructing the geometry")
                drawing = Constructor(width, height, style, max_steps=4000).run(program)
                meta.update(planner="image-vision", notes=[program.get("notes", "")], components=[], program_steps=len(program["steps"]))
            else:
                planner_name = "nebula-construct"
                self._progress(record, "planning the drawing")
                if self.constructor_planner.available:
                    plan = self.constructor_planner.plan(job["prompt"], width, height, style, job["quality"], on_stage=lambda label: self._progress(record, label))
                    program, drawing = plan["program"], plan["drawing"]
                    comps = [c.get("name") for c in (plan.get("decomposition") or {}).get("components", []) if isinstance(c, dict)]
                    meta.update(planner=planner_name, model=plan.get("model"), timeline=plan["timeline"], components=[c for c in comps if c], title=program.get("title"))
                elif os.environ.get("NEBULA_ARTIST_FALLBACK") == "1":
                    # Developer opt-in only: the older stored-pattern drawings (no language model).
                    return self._run_legacy(job, record)
                else:
                    raise PlanningUnavailableError("Nebula's drawing model is not available right now, so nothing was drawn.")
            meta["warnings"] = drawing.warnings[:8]

            self._progress(record, "rendering", **{k: v for k, v in meta.items() if k not in ("stage",)})
            svg_filename, png_filename = f"{job['base']}.svg", f"{job['base']}.png"
            svg_path = self.settings.drawings_dir / svg_filename
            png_path = self.settings.drawings_dir / png_filename
            svg_path.write_text(VectorSVG.render(drawing, job["prompt"]), encoding="utf-8")
            VectorPNG.render(drawing).save(str(png_path), format="PNG", optimize=True, dpi=(144, 144))

            primary_name, primary_path, out_fmt = (svg_filename, svg_path, "svg") if job["format"] == "svg" else (png_filename, png_path, "png")
            record.filename, record.format, record.storage_path = primary_name, out_fmt, str(primary_path)
            record.primitives_count, record.operations_count = drawing.primitives_count, drawing.operations_count
            record.status = DrawingStatus.COMPLETED.value
            record.metadata = {**record.metadata, "stage": "done", "seconds": round(time.time() - t0, 1)}
            self.db.save_drawing(record)
            return self._response(record)
        except Exception as exc:  # noqa: BLE001
            record.status = DrawingStatus.FAILED.value
            code = getattr(exc, "code", "drawing_failed")
            record.metadata = {**record.metadata, "stage": "failed", "error": getattr(exc, "message", str(exc))[:400], "error_code": code, "seconds": round(time.time() - t0, 1)}
            self.db.save_drawing(record)
            if raise_errors:
                raise
            return self._response(record)

    def _run_legacy(self, job: Dict[str, Any], record: DrawingRecord) -> DrawResponse:
        """The original stored-pattern engine, kept for developers (NEBULA_ARTIST_FALLBACK=1) and its tests."""
        plan = self.planner.plan_drawing(
            prompt=job["prompt"], width=job["width"], height=job["height"],
            grid_resolution=float(self.settings.default_grid_size), custom_params=job.get("parameters"),
        )
        svg_filename, png_filename = f"{job['base']}.svg", f"{job['base']}.png"
        svg_path = self.settings.drawings_dir / svg_filename
        svg_path.write_text(SVGRenderer.render(plan), encoding="utf-8")
        png_path = self.settings.drawings_dir / png_filename
        PNGRenderer.render(plan).save(str(png_path), format="PNG")
        primary_name, primary_path, out_fmt = (svg_filename, svg_path, "svg") if job["format"] == "svg" else (png_filename, png_path, "png")
        record.filename, record.format, record.storage_path = primary_name, out_fmt, str(primary_path)
        record.primitives_count, record.operations_count = len(plan.primitives), len(plan.operations)
        record.status = DrawingStatus.COMPLETED.value
        record.metadata = {**plan.metadata, "planner": "pattern-fallback", "stage": "done", "style": "legacy", "source": "prompt"}
        self.db.save_drawing(record)
        return self._response(record)

    def _response(self, record: DrawingRecord) -> DrawResponse:
        m = record.metadata or {}
        done = record.status == DrawingStatus.COMPLETED.value
        # A job that has been "pending" for a long time lost its worker (the service restarted): say so.
        if record.status in (DrawingStatus.PENDING.value, DrawingStatus.RENDERING.value) and record.created_at:
            try:
                age = (datetime.now(timezone.utc) - datetime.fromisoformat(record.created_at)).total_seconds()
                if age > 600:
                    record.status = DrawingStatus.FAILED.value
                    m = {**m, "error": "This drawing was interrupted (the service restarted).", "stage": "failed"}
            except ValueError:
                pass
        return DrawResponse(
            drawing_id=record.drawing_id,
            prompt=record.prompt,
            filename=record.filename,
            format=record.format,
            width=record.width,
            height=record.height,
            image_url=f"/artist/drawings/{record.drawing_id}/file" if done else "",
            svg_url=f"/artist/drawings/{record.drawing_id}/file?format=svg" if done else None,
            status=record.status,
            primitives_count=record.primitives_count,
            operations_count=record.operations_count,
            storage_path=record.storage_path,
            created_at=record.created_at or "",
            style=m.get("style"),
            planner=m.get("planner"),
            source=m.get("source", "prompt"),
            model=m.get("model"),
            notes=[n for n in (m.get("notes") or []) + (m.get("warnings") or []) if n],
            timeline=m.get("timeline") or [],
            stage=m.get("stage"),
            components=[str(c) for c in (m.get("components") or [])][:40],
            error=m.get("error"),
        )

    def status(self, drawing_id: str) -> DrawResponse:
        """The current state of a drawing (used to poll a background job)."""
        return self._response(self.get_drawing(drawing_id))

    def get_drawing(self, drawing_id: str) -> DrawingRecord:
        record = self.db.get_drawing(drawing_id)
        if not record:
            raise DrawingNotFoundError(drawing_id)
        return record

    def list_drawings(self, limit: int = 50) -> List[DrawingRecord]:
        return self.db.list_drawings(limit=limit)

    def get_pattern(self, pattern_name_or_id: str) -> Optional[PatternRecord]:
        pattern = self.db.get_pattern(pattern_name_or_id)
        if not pattern:
            pattern = self.db.get_pattern_by_name(pattern_name_or_id)
        return pattern

    def list_patterns(self, category: Optional[str] = None) -> List[PatternRecord]:
        return self.db.list_patterns(category=category)

    def save_pattern(self, request: PatternCreateRequest) -> PatternRecord:
        return self.db.save_pattern(
            name=request.name,
            category=request.category,
            parameters=request.parameters,
            primitives=request.primitives,
            operations=request.operations,
            description=request.description or "",
            tags=request.tags,
        )
