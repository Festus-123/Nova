"""
The Artist: Geometric Drawing Engine Orchestrator (Ability 0.01).

Coordinates Nebula Intelligence, Canvas Grid, Geometry & Boolean Engine,
Pattern Database, and Image Renderers.
"""

import os
import re
import uuid
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


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_\-]+", "_", text.strip().lower())
    return slug[:40].strip("_") or "drawing"


class ArtistEngine:
    """The central engine for Ability 0.01: The Artist."""

    def __init__(self, settings: Optional[ArtistSettings] = None, db: Optional[ArtistDB] = None):
        self.settings = settings or ARTIST_SETTINGS
        self.db = db or ArtistDB(db_path=self.settings.db_path)
        self.planner = NebulaPlanner(db=self.db)
        self._ensure_storage()

    def _ensure_storage(self) -> None:
        self.settings.drawings_dir.mkdir(parents=True, exist_ok=True)

    def draw(self, request: DrawRequest) -> DrawResponse:
        """
        Executes complete drawing pipeline from prompt to saved image.
        """
        prompt = request.prompt.strip()
        if not prompt:
            raise GeometricValidationError("Drawing prompt cannot be empty")

        width = request.width or self.settings.default_width
        height = request.height or self.settings.default_height
        requested_format = (request.format or self.settings.default_format).lower().strip()

        drawing_id = f"draw-{uuid.uuid4().hex[:12]}"
        prompt_slug = _slugify(prompt)
        base_filename = f"{prompt_slug}_{drawing_id[:8]}"

        # 1. Plan drawing through Nebula Intelligence
        plan = self.planner.plan_drawing(
            prompt=prompt,
            width=width,
            height=height,
            grid_resolution=float(self.settings.default_grid_size),
            custom_params=request.parameters,
        )

        # 2. Render vector SVG
        svg_content = SVGRenderer.render(plan)
        svg_filename = f"{base_filename}.svg"
        svg_path = self.settings.drawings_dir / svg_filename
        svg_path.write_text(svg_content, encoding="utf-8")

        # 3. Render raster PNG
        png_image = PNGRenderer.render(plan)
        png_filename = f"{base_filename}.png"
        png_path = self.settings.drawings_dir / png_filename
        png_image.save(str(png_path), format="PNG")

        # Determine primary output
        if requested_format == "svg":
            primary_filename = svg_filename
            primary_path = svg_path
            out_format = "svg"
        else:
            primary_filename = png_filename
            primary_path = png_path
            out_format = "png"

        now_str = datetime.now(timezone.utc).isoformat()

        # 4. Record drawing in Artist DB
        record = DrawingRecord(
            drawing_id=drawing_id,
            prompt=prompt,
            filename=primary_filename,
            format=out_format,
            width=width,
            height=height,
            storage_path=str(primary_path),
            status=DrawingStatus.COMPLETED.value,
            primitives_count=len(plan.primitives),
            operations_count=len(plan.operations),
            metadata=plan.metadata,
            created_at=now_str,
        )
        self.db.save_drawing(record)

        return DrawResponse(
            drawing_id=drawing_id,
            prompt=prompt,
            filename=primary_filename,
            format=out_format,
            width=width,
            height=height,
            image_url=f"/artist/drawings/{drawing_id}/file",
            svg_url=f"/artist/drawings/{drawing_id}/file?format=svg",
            status=DrawingStatus.COMPLETED.value,
            primitives_count=len(plan.primitives),
            operations_count=len(plan.operations),
            storage_path=str(primary_path),
            created_at=now_str,
        )

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
