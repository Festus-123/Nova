"""
Data models and schemas for The Artist (Ability 0.01).
"""

from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field


class OperationType(str, Enum):
    UNION = "union"
    INTERSECTION = "intersection"
    SUBTRACTION = "subtraction"
    DRAW = "draw"


class PrimitiveType(str, Enum):
    CIRCLE = "circle"
    ELLIPSE = "ellipse"
    LINE = "line"
    BEZIER_QUAD = "bezier_quad"
    BEZIER_CUBIC = "bezier_cubic"
    POLYGON = "polygon"
    ARC = "arc"


class RenderFormat(str, Enum):
    PNG = "png"
    SVG = "svg"


class DrawingStatus(str, Enum):
    PENDING = "pending"
    RENDERING = "rendering"
    COMPLETED = "completed"
    FAILED = "failed"


class Point(BaseModel):
    x: float
    y: float

    def as_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


class ColorStyle(BaseModel):
    fill: Optional[str] = None
    stroke: Optional[str] = "#000000"
    stroke_width: float = 2.0
    opacity: float = 1.0


class GeometricPrimitive(BaseModel):
    id: str
    type: PrimitiveType
    # Points usage:
    # - CIRCLE: points[0] is center
    # - ELLIPSE: points[0] is center
    # - LINE: points[0]=start, points[1]=end
    # - BEZIER_QUAD: points[0]=p0, points[1]=p1(control), points[2]=p2(end)
    # - BEZIER_CUBIC: points[0]=p0, points[1]=p1, points[2]=p2, points[3]=p3
    # - POLYGON: points are vertices
    # - ARC: points[0] is center
    points: List[Point] = Field(default_factory=list)
    radius: Optional[float] = None
    radius_y: Optional[float] = None
    angles: Optional[Tuple[float, float]] = None  # (start_angle_deg, end_angle_deg)
    fill: Optional[str] = None
    stroke: Optional[str] = "#000000"
    stroke_width: float = 2.0
    opacity: float = 1.0
    layer: int = 0
    label: Optional[str] = None
    closed: bool = True


class GeometricOperation(BaseModel):
    id: str
    operation: OperationType
    target_id: str
    operand_ids: List[str] = Field(default_factory=list)
    result_fill: Optional[str] = None
    result_stroke: Optional[str] = None
    result_stroke_width: Optional[float] = None
    label: Optional[str] = None


class DrawingPlan(BaseModel):
    prompt: str
    width: int = 800
    height: int = 600
    background_color: str = "#FFFFFF"
    primitives: List[GeometricPrimitive] = Field(default_factory=list)
    operations: List[GeometricOperation] = Field(default_factory=list)
    sequence: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PatternRecord(BaseModel):
    pattern_id: str
    name: str
    category: str
    version: int = 1
    description: str = ""
    parameters: Dict[str, Any] = Field(default_factory=dict)
    primitives: List[Dict[str, Any]] = Field(default_factory=list)
    operations: List[Dict[str, Any]] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class DrawingRecord(BaseModel):
    drawing_id: str
    prompt: str
    filename: str
    format: str
    width: int
    height: int
    storage_path: str
    status: str
    primitives_count: int = 0
    operations_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None


class DrawRequest(BaseModel):
    model_config = {"populate_by_name": True}

    prompt: str = Field(..., description="Drawing instruction, e.g. 'Draw a Ferrari'")
    width: Optional[int] = Field(800, ge=100, le=4096)
    height: Optional[int] = Field(600, ge=100, le=4096)
    format: Optional[str] = Field("png", description="Output format: 'png' or 'svg'")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Optional custom styling/proportions")
    style: Optional[str] = Field("line", description="'line' (clean black line art), 'solid' (bold silhouette) or 'color' (flat colours)")
    quality: Optional[str] = Field("balanced", description="'fast' (one planning step), 'balanced' (interpret, then construct) or 'best' (also review and refine)")
    run_async: Optional[bool] = Field(False, alias="async", description="Return at once with status 'pending'; poll GET /artist/drawings/{id}")
    algorithm: Optional[str] = Field("1.0", description="G-Stroke version: '1.0' (geometry, the default) or '2.0' (sketching with brushes, strokes and strands)")
    brush: Optional[str] = Field(None, description="G-Stroke 2.0 only: pencil (default), pen, fineliner, ink, charcoal or marker")


class DrawResponse(BaseModel):
    drawing_id: str
    prompt: str
    filename: str
    format: str
    width: int
    height: int
    image_url: str
    svg_url: Optional[str] = None
    status: str
    primitives_count: int
    operations_count: int
    storage_path: str
    created_at: str
    # Added with the constructive engine; all optional so existing clients keep working.
    style: Optional[str] = None
    planner: Optional[str] = None
    source: Optional[str] = "prompt"
    model: Optional[str] = None
    notes: List[str] = Field(default_factory=list)
    timeline: List[Dict[str, Any]] = Field(default_factory=list)
    stage: Optional[str] = None
    components: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    algorithm: Optional[str] = "1.0"  # which G-Stroke version drew it
    brush: Optional[str] = None  # G-Stroke 2.0 only


class PatternCreateRequest(BaseModel):
    name: str
    category: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    primitives: List[Dict[str, Any]] = Field(default_factory=list)
    operations: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    description: Optional[str] = ""
    tags: Optional[List[str]] = Field(default_factory=list)
