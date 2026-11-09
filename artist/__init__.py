"""
The Artist (Ability 0.01) - Geometric Drawing Engine for Nebula.
"""

from .config import ArtistSettings, ARTIST_SETTINGS
from .errors import (
    ArtistError,
    GeometricValidationError,
    PatternNotFoundError,
    DrawingNotFoundError,
    RenderingError,
    PlanningError,
)
from .models import (
    OperationType,
    PrimitiveType,
    RenderFormat,
    DrawingStatus,
    Point,
    GeometricPrimitive,
    GeometricOperation,
    DrawingPlan,
    PatternRecord,
    DrawingRecord,
    DrawRequest,
    DrawResponse,
    PatternCreateRequest,
)
from .geometry import (
    Vector2D,
    CircleMath,
    StrokeMath,
    BezierMath,
    PolygonMath,
    TransformMath,
)
from .grid import CanvasGrid, GridCell
from .boolean import BooleanEngine
from .db import ArtistDB
from .patterns import PatternEngine
from .renderer import SVGRenderer, PNGRenderer
from .nebula_planner import NebulaPlanner
from .construct import Constructor, ResolvedDrawing
from .llm_planner import ConstructionPlanner, PlanningUnavailableError
from .vector_render import VectorSVG, VectorPNG
from .vision import ImageVectorizer, ImageProblem
from .gstroke import ALGORITHMS, DEFAULT_ALGORITHM, ImageSketcher, Sketch, SketchPlanner, SketchPNG, SketchSVG, resolve_algorithm
from .engine import ArtistEngine
from .tool import artist_draw

__all__ = [
    "ArtistSettings",
    "ARTIST_SETTINGS",
    "ArtistError",
    "GeometricValidationError",
    "PatternNotFoundError",
    "DrawingNotFoundError",
    "RenderingError",
    "PlanningError",
    "OperationType",
    "PrimitiveType",
    "RenderFormat",
    "DrawingStatus",
    "Point",
    "GeometricPrimitive",
    "GeometricOperation",
    "DrawingPlan",
    "PatternRecord",
    "DrawingRecord",
    "DrawRequest",
    "DrawResponse",
    "PatternCreateRequest",
    "Vector2D",
    "CircleMath",
    "StrokeMath",
    "BezierMath",
    "PolygonMath",
    "TransformMath",
    "CanvasGrid",
    "GridCell",
    "BooleanEngine",
    "ArtistDB",
    "PatternEngine",
    "SVGRenderer",
    "PNGRenderer",
    "NebulaPlanner",
    "ArtistEngine",
    "Constructor",
    "ResolvedDrawing",
    "ConstructionPlanner",
    "PlanningUnavailableError",
    "VectorSVG",
    "VectorPNG",
    "ImageVectorizer",
    "ImageProblem",
    "artist_draw",
    "ALGORITHMS",
    "DEFAULT_ALGORITHM",
    "ImageSketcher",
    "Sketch",
    "SketchPlanner",
    "SketchPNG",
    "SketchSVG",
    "resolve_algorithm",
]
