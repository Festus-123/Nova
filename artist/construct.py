"""
Constructive Geometry Engine for The Artist (Ability 0.01).

Nebula decides WHAT to build and WHERE; this module builds it exactly.

A drawing is a *construction program*: an ordered list of steps. A step either creates a shape (circle, ellipse,
rectangle, triangle, polygon, stroke/line, arc, bezier) or combines shapes already made with the three boolean
operations the ability is founded on:

    UNION        merge shapes into one continuous shape
    INTERSECT    keep only the region the shapes share
    SUBTRACT     cut shapes out of another shape (windows, doors, wheel arches, holes)

The booleans are performed on the real vector geometry (not on pixels), so the result is exact: the outline of a
union really is the outline of the merged shape, and a subtraction really leaves a hole. The output is a set of
resolved vector shapes that the renderer writes out as SVG (and PNG).

Coordinates in a program are in a resolution-independent *design grid*: DESIGN_WIDTH units across, with the height
following the canvas aspect ratio. Origin is the top-left, x grows right, y grows DOWN. The grid is what lets the
same program be drawn at any size.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field
from shapely import affinity
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Point as SPoint,
    Polygon,
    box as sbox,
)
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .errors import GeometricValidationError

DESIGN_WIDTH = 100.0
MAX_STEPS = 400
# Smoothness of curves: segments per quarter circle. High enough that a circle stays round at 4x zoom.
CURVE_SEGMENTS = 48

INK = "#111111"
TONES = {"light": "#FFFFFF", "mid": "#C9C9C9", "dark": INK}
PAPER = "#FFFFFF"


class ConstructionError(GeometricValidationError):
    """A construction program that cannot be built (nothing usable in it)."""


# --------------------------------------------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------------------------------------------


class ResolvedItem(BaseModel):
    """One finished shape, ready to draw."""

    model_config = {"arbitrary_types_allowed": True}

    id: str
    geometry: Any  # shapely Polygon / MultiPolygon, in canvas pixels
    fill: Optional[str] = None
    stroke: Optional[str] = None
    stroke_width: float = 0.0
    layer: int = 0
    is_stroke: bool = False
    label: Optional[str] = None


class ResolvedDrawing(BaseModel):
    width: int
    height: int
    background: str = PAPER
    style: str = "line"
    items: List[ResolvedItem] = Field(default_factory=list)
    primitives_count: int = 0
    operations_count: int = 0
    steps_run: int = 0
    warnings: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------------------------------------------
# Lenient reading of what the model wrote
# --------------------------------------------------------------------------------------------------------------

_TYPE_ALIASES = {
    "circle": "circle", "disc": "circle", "disk": "circle", "dot": "circle",
    "ellipse": "ellipse", "oval": "ellipse",
    "rect": "rect", "rectangle": "rect", "square": "rect", "box": "rect", "quad": "rect",
    "triangle": "triangle", "tri": "triangle",
    "polygon": "polygon", "poly": "polygon", "path": "polygon", "shape": "polygon",
    "line": "line", "stroke": "line", "polyline": "line", "segment": "line", "strokes": "line",
    "arc": "arc",
    "bezier": "bezier", "curve": "bezier", "bezier_quad": "bezier", "bezier_cubic": "bezier", "quadratic": "bezier", "cubic": "bezier",
    "ring": "ring", "donut": "ring", "annulus": "ring",
}
_OP_ALIASES = {
    "shape": "shape", "create": "shape", "make": "shape", "primitive": "shape", "draw": "shape", "add": "shape",
    "union": "union", "merge": "union", "combine": "union", "add_shapes": "union",
    "intersect": "intersect", "intersection": "intersect", "and": "intersect", "overlap": "intersect", "clip": "intersect",
    "subtract": "subtract", "subtraction": "subtract", "difference": "subtract", "cut": "subtract", "minus": "subtract", "diff": "subtract",
}


def _num(v: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if v is None or isinstance(v, bool):
            return default
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _pt(v: Any) -> Optional[Tuple[float, float]]:
    """Accepts [x, y], (x, y) or {"x": .., "y": ..}."""
    if isinstance(v, dict):
        x, y = _num(v.get("x")), _num(v.get("y"))
    elif isinstance(v, (list, tuple)) and len(v) >= 2:
        x, y = _num(v[0]), _num(v[1])
    else:
        return None
    return None if x is None or y is None else (x, y)


def _pts(v: Any) -> List[Tuple[float, float]]:
    if not isinstance(v, (list, tuple)):
        return []
    out = [p for p in (_pt(i) for i in v) if p is not None]
    return out


def _ids(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, (list, tuple)):
        return [str(i) for i in v if i is not None]
    return []


def _first(d: Dict[str, Any], *names: str) -> Any:
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return None


def normalize_program(raw: Any, max_steps: int = MAX_STEPS) -> Dict[str, Any]:
    """Finds the steps in whatever wrapper the model used, and tidies the field names."""
    if isinstance(raw, list):
        raw = {"steps": raw}
    if not isinstance(raw, dict):
        raise ConstructionError("The construction program is not a JSON object")
    steps = _first(raw, "steps", "construction", "program", "operations", "plan")
    if not isinstance(steps, list):
        raise ConstructionError("The construction program has no list of steps")
    out_steps = []
    for s in steps[:max_steps]:
        if isinstance(s, dict):
            out_steps.append(s)
    return {
        "title": str(_first(raw, "title", "object", "name") or ""),
        "steps": out_steps,
        "draw": _first(raw, "draw", "output", "outputs", "render", "layers") or [],
        "notes": _first(raw, "notes", "note") or "",
        "background": _first(raw, "background", "paper"),
    }


# --------------------------------------------------------------------------------------------------------------
# Building shapes
# --------------------------------------------------------------------------------------------------------------


def _bezier_points(ctrl: Sequence[Tuple[float, float]], steps: int = 64) -> List[Tuple[float, float]]:
    out = []
    n = len(ctrl) - 1
    for i in range(steps + 1):
        t = i / steps
        # de Casteljau
        pts = list(ctrl)
        for _ in range(n):
            pts = [((1 - t) * pts[j][0] + t * pts[j + 1][0], (1 - t) * pts[j][1] + t * pts[j + 1][1]) for j in range(len(pts) - 1)]
        out.append(pts[0])
    return out


def _stroke(line_pts: Sequence[Tuple[float, float]], width: float, cap: str) -> Optional[BaseGeometry]:
    if len(line_pts) < 2 or width <= 0:
        return None
    cap_style = {"round": 1, "flat": 2, "butt": 2, "square": 3}.get(cap, 1)
    ls = LineString(line_pts)
    if ls.length == 0:
        return None
    return ls.buffer(width / 2.0, quad_segs=CURVE_SEGMENTS // 2, cap_style=cap_style, join_style=1)


def _rotate(g: BaseGeometry, angle: Optional[float], origin: Any = "centroid") -> BaseGeometry:
    return affinity.rotate(g, angle, origin=origin) if angle else g


def build_shape(step: Dict[str, Any]) -> Tuple[Optional[BaseGeometry], bool, Optional[str]]:
    """
    Builds one shape from a step. Returns (geometry, is_stroke, problem).
    is_stroke marks thin line-like shapes so that the renderer fills them solid instead of outlining them.
    """
    raw_type = str(_first(step, "type", "shape", "kind", "primitive") or "").lower().strip()
    t = _TYPE_ALIASES.get(raw_type)
    if t is None:
        return None, False, f"unknown shape type '{raw_type}'"
    angle = _num(_first(step, "angle", "rotate", "rotation"), 0.0)

    try:
        if t == "circle":
            c = _pt(_first(step, "center", "c")) or (_num(step.get("cx")), _num(step.get("cy")))
            r = _num(_first(step, "r", "radius"))
            if c is None or c[0] is None or c[1] is None or not r or r <= 0:
                return None, False, "circle needs cx, cy and a positive r"
            return SPoint(c).buffer(r, quad_segs=CURVE_SEGMENTS), False, None

        if t == "ellipse":
            c = _pt(_first(step, "center", "c")) or (_num(step.get("cx")), _num(step.get("cy")))
            rx = _num(_first(step, "rx", "radius_x", "r"))
            ry = _num(_first(step, "ry", "radius_y"), rx)
            if c is None or c[0] is None or c[1] is None or not rx or not ry or rx <= 0 or ry <= 0:
                return None, False, "ellipse needs cx, cy, rx and ry"
            g = affinity.scale(SPoint(c).buffer(1.0, quad_segs=CURVE_SEGMENTS), rx, ry, origin=c)
            return _rotate(g, angle, c), False, None

        if t == "rect":
            x, y, w, h = (_num(step.get(k)) for k in ("x", "y", "w", "h"))
            if w is None:
                w = _num(step.get("width"))
            if h is None:
                h = _num(step.get("height"))
            if x is None or y is None or not w or not h or w <= 0 or h <= 0:
                return None, False, "rect needs x, y, w and h"
            g: BaseGeometry = sbox(x, y, x + w, y + h)
            rr = _num(_first(step, "radius", "corner", "round"), 0.0) or 0.0
            if rr > 0:
                rr = min(rr, w / 2.0 - 1e-6, h / 2.0 - 1e-6)
                if rr > 0:
                    g = sbox(x + rr, y + rr, x + w - rr, y + h - rr).buffer(rr, quad_segs=CURVE_SEGMENTS // 2)
            return _rotate(g, angle), False, None

        if t in ("triangle", "polygon"):
            pts = _pts(_first(step, "points", "vertices", "p"))
            need = 3
            if len(pts) < need:
                return None, False, f"{t} needs at least {need} points"
            if t == "triangle" and len(pts) > 3:
                pts = pts[:3]
            g = Polygon(pts)
            if not g.is_valid:
                g = g.buffer(0)
            if g.is_empty or g.area <= 0:
                return None, False, f"{t} has no area"
            return _rotate(g, angle), False, None

        if t == "line":
            pts = _pts(_first(step, "points", "p"))
            if len(pts) < 2:
                a, b = _pt(step.get("from")), _pt(step.get("to"))
                pts = [p for p in (a, b) if p]
            width = _num(_first(step, "width", "stroke_width", "thickness", "w"), 1.0)
            cap = str(step.get("cap") or "round").lower()
            g = _stroke(pts, width or 1.0, cap)
            if g is None:
                return None, True, "line needs two points and a positive width"
            return _rotate(g, angle), True, None

        if t == "arc":
            c = _pt(_first(step, "center", "c")) or (_num(step.get("cx")), _num(step.get("cy")))
            r = _num(_first(step, "r", "radius"))
            a0 = _num(_first(step, "start", "start_angle", "from"), 0.0)
            a1 = _num(_first(step, "end", "end_angle", "to"), 180.0)
            width = _num(_first(step, "width", "stroke_width", "thickness"), 1.0)
            if c is None or c[0] is None or c[1] is None or not r or r <= 0:
                return None, True, "arc needs cx, cy and r"
            # Degrees, 0 = east (right), growing clockwise on screen (y points down).
            if a1 < a0:
                a1 += 360.0
            n = max(8, int(abs(a1 - a0) / 360.0 * CURVE_SEGMENTS * 4))
            pts = [(c[0] + r * math.cos(math.radians(a0 + (a1 - a0) * i / n)), c[1] + r * math.sin(math.radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]
            g = _stroke(pts, width or 1.0, str(step.get("cap") or "round").lower())
            if g is None:
                return None, True, "arc could not be built"
            return g, True, None

        if t == "bezier":
            pts = _pts(_first(step, "points", "p"))
            if len(pts) not in (3, 4):
                return None, False, "bezier needs 3 (quadratic) or 4 (cubic) control points"
            curve = _bezier_points(pts)
            if step.get("closed") or step.get("fill") is True:
                g = Polygon(curve)
                g = g if g.is_valid else g.buffer(0)
                if g.is_empty or g.area <= 0:
                    return None, False, "closed bezier has no area"
                return _rotate(g, angle), False, None
            width = _num(_first(step, "width", "stroke_width", "thickness"), 1.0)
            g = _stroke(curve, width or 1.0, str(step.get("cap") or "round").lower())
            if g is None:
                return None, True, "bezier could not be built"
            return g, True, None

        if t == "ring":
            c = _pt(_first(step, "center", "c")) or (_num(step.get("cx")), _num(step.get("cy")))
            r = _num(_first(step, "r", "radius", "outer"))
            ri = _num(_first(step, "r_inner", "inner", "inner_radius"))
            if c is None or c[0] is None or c[1] is None or not r or not ri or r <= ri or ri <= 0:
                return None, False, "ring needs cx, cy, r and a smaller r_inner"
            return SPoint(c).buffer(r, quad_segs=CURVE_SEGMENTS).difference(SPoint(c).buffer(ri, quad_segs=CURVE_SEGMENTS)), False, None

    except Exception as e:  # a bad number must not take the whole drawing down
        return None, False, f"{t} could not be built ({e})"
    return None, False, f"unhandled shape '{t}'"


def _polys(g: BaseGeometry) -> List[Polygon]:
    """The polygons inside any geometry (drops stray points and lines that boolean ops can leave behind)."""
    if g is None or g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    if isinstance(g, MultiPolygon):
        return list(g.geoms)
    if isinstance(g, GeometryCollection):
        out: List[Polygon] = []
        for part in g.geoms:
            out.extend(_polys(part))
        return out
    return []


def _clean(g: Optional[BaseGeometry]) -> Optional[BaseGeometry]:
    if g is None:
        return None
    if not g.is_valid:
        g = g.buffer(0)
    polys = [p for p in _polys(g) if p.area > 1e-6]
    if not polys:
        return None
    return polys[0] if len(polys) == 1 else MultiPolygon(polys)


# --------------------------------------------------------------------------------------------------------------
# The constructor
# --------------------------------------------------------------------------------------------------------------


class Constructor:
    """Runs a construction program and returns resolved vector shapes."""

    def __init__(self, width: int = 800, height: int = 600, style: str = "line", fit: bool = True, max_steps: int = MAX_STEPS):
        self.max_steps = max_steps
        self.width = int(width)
        self.height = int(height)
        self.style = style if style in ("line", "solid", "color") else "line"
        self.fit = fit
        self.units_w = DESIGN_WIDTH
        self.units_h = DESIGN_WIDTH * self.height / self.width
        self.k = self.width / self.units_w  # pixels per design unit

    # -- running -------------------------------------------------------------------------------------------

    def run(self, program: Any) -> ResolvedDrawing:
        prog = normalize_program(program, self.max_steps)
        shapes: Dict[str, BaseGeometry] = {}
        strokes: Dict[str, bool] = {}
        meta: Dict[str, Dict[str, Any]] = {}
        consumed: set = set()
        order: List[str] = []
        warnings: List[str] = []
        prims = ops = 0

        for n, step in enumerate(prog["steps"]):
            sid = str(_first(step, "id", "name") or f"s{n + 1}")
            op = _OP_ALIASES.get(str(_first(step, "op", "operation", "action") or ("shape" if _first(step, "type", "shape") else "")).lower().strip())
            if op is None:
                warnings.append(f"step {sid}: unknown operation")
                continue

            if op == "shape":
                g, is_stroke, problem = build_shape(step)
                if problem or g is None:
                    warnings.append(f"step {sid}: {problem}")
                    continue
                g = _clean(g)
                if g is None:
                    warnings.append(f"step {sid}: shape is empty")
                    continue
                shapes[sid] = g
                strokes[sid] = is_stroke
                meta[sid] = step
                order.append(sid)
                prims += 1
                continue

            # boolean steps
            if op in ("union", "intersect"):
                refs = _ids(_first(step, "of", "ids", "shapes", "operands", "targets"))
                refs = [r for r in refs if r in shapes]
                if len(refs) < 2:
                    warnings.append(f"step {sid}: {op} needs two or more existing shapes")
                    continue
                if op == "union":
                    g = unary_union([shapes[r] for r in refs])
                else:
                    g = shapes[refs[0]]
                    for r in refs[1:]:
                        g = g.intersection(shapes[r])
                src = refs
            else:  # subtract
                base = _first(step, "from", "base", "target", "of")
                base = base if isinstance(base, str) else (_ids(base) or [None])[0]
                rem = _ids(_first(step, "remove", "subtract", "minus", "cut", "operands", "shapes", "with"))
                rem = [r for r in rem if r in shapes and r != base]
                if base not in shapes or not rem:
                    warnings.append(f"step {sid}: subtract needs an existing base and shapes to remove")
                    continue
                g = shapes[base]
                for r in rem:
                    g = g.difference(shapes[r])
                src = [base] + rem

            g = _clean(g)
            if g is None:
                warnings.append(f"step {sid}: {op} left nothing (the shapes do not overlap as intended)")
                continue
            ops += 1
            shapes[sid] = g
            strokes[sid] = all(strokes.get(s, False) for s in src)
            meta[sid] = {**step, "_from": src}
            order.append(sid)
            # Unless asked to keep them, the shapes that went into an operation are not drawn by themselves.
            if not step.get("keep"):
                consumed.update(src)

        if not shapes:
            raise ConstructionError("Nothing could be built from the construction program: " + "; ".join(warnings[:4]) if warnings else "the program has no shapes")

        # What to draw, in what order.
        wanted: List[Dict[str, Any]] = []
        for d in prog["draw"] if isinstance(prog["draw"], list) else []:
            if isinstance(d, str):
                wanted.append({"id": d})
            elif isinstance(d, dict) and _first(d, "id", "shape", "name") is not None:
                wanted.append({**d, "id": str(_first(d, "id", "shape", "name"))})
        wanted = [w for w in wanted if w["id"] in shapes]
        if not wanted:
            wanted = [{"id": s} for s in order if s not in consumed]
            if not wanted:
                wanted = [{"id": s} for s in order]
        elif any(s not in {w["id"] for w in wanted} and s not in consumed for s in order):
            # The model listed some shapes but left others (not consumed by an operation) out: draw those too.
            listed = {w["id"] for w in wanted}
            wanted += [{"id": s} for s in order if s not in listed and s not in consumed]
            warnings.append("some finished shapes were not in the draw list; they were drawn too")

        items = self._style_items(wanted, shapes, strokes, meta)
        drawing = ResolvedDrawing(
            width=self.width, height=self.height, style=self.style, items=items,
            primitives_count=prims, operations_count=ops, steps_run=prims + ops, warnings=warnings,
        )
        if _valid_hex(prog.get("background")):
            drawing.background = prog["background"]
        if self.fit:
            self._fit(drawing)
        return drawing

    # -- styling -------------------------------------------------------------------------------------------

    def _style_items(self, wanted: List[Dict[str, Any]], shapes: Dict[str, BaseGeometry], strokes: Dict[str, bool], meta: Dict[str, Dict[str, Any]]) -> List[ResolvedItem]:
        base_sw = max(1.2, 2.4 * self.width / 800.0)
        items: List[ResolvedItem] = []
        for idx, w in enumerate(wanted):
            sid = w["id"]
            g = affinity.scale(shapes[sid], self.k, self.k, origin=(0, 0))
            tone = str(_first(w, "tone", "shade") or "").lower()
            color = _first(w, "color", "colour", "fill")
            is_stroke = strokes.get(sid, False)
            sw_units = _num(_first(w, "stroke_width"), None)
            sw = (sw_units * self.k) if sw_units else base_sw
            layer = int(_num(_first(w, "layer", "z"), idx) or idx)

            fill: Optional[str]
            stroke: Optional[str]
            if is_stroke:
                # Lines, arcs and curves are already thick shapes: fill them solid.
                fill = INK if self.style != "color" or not _valid_hex(color) else color
                stroke, sw = None, 0.0
            elif self.style == "line":
                fill = TONES.get(tone, PAPER)  # white by default: hides what is behind it, like real line art
                stroke = INK
            elif self.style == "solid":
                fill = TONES.get(tone, INK)
                stroke = None
                sw = 0.0
            else:  # color
                fill = color if _valid_hex(color) else TONES.get(tone, "#E8E8E8")
                stroke = INK
                sw = max(1.0, base_sw * 0.7)
            items.append(ResolvedItem(id=sid, geometry=g, fill=fill, stroke=stroke, stroke_width=sw, layer=layer, is_stroke=is_stroke, label=str(w.get("label") or sid)))
        items.sort(key=lambda i: i.layer)
        return items

    # -- composition ---------------------------------------------------------------------------------------

    def _fit(self, d: ResolvedDrawing) -> None:
        """Centres the drawing and scales it to a comfortable size, whatever coordinates the model chose."""
        geoms = [i.geometry for i in d.items if i.geometry is not None and not i.geometry.is_empty]
        if not geoms:
            return
        minx, miny, maxx, maxy = unary_union(geoms).bounds
        bw, bh = maxx - minx, maxy - miny
        if bw <= 0 or bh <= 0:
            return
        margin = 0.08
        avail_w, avail_h = self.width * (1 - 2 * margin), self.height * (1 - 2 * margin)
        s = min(avail_w / bw, avail_h / bh)
        inside = minx >= 0 and miny >= 0 and maxx <= self.width and maxy <= self.height
        fills = max(bw / self.width, bh / self.height)
        # Leave a good composition alone; fix one that is clipped, off-centre, or tiny.
        centred = abs((minx + maxx) / 2 - self.width / 2) < self.width * 0.1 and abs((miny + maxy) / 2 - self.height / 2) < self.height * 0.12
        if inside and centred and 0.55 <= fills <= 0.95:
            return
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        for i in d.items:
            g = affinity.translate(i.geometry, self.width / 2 - cx, self.height / 2 - cy)
            g = affinity.scale(g, s, s, origin=(self.width / 2, self.height / 2))
            i.geometry = g  # outline weight is left alone on purpose: scaling a drawing must not fatten its lines
        d.warnings.append(f"composition adjusted to fit the canvas (scale {s:.2f})")


def _valid_hex(c: Any) -> bool:
    if not isinstance(c, str) or not c.startswith("#"):
        return False
    h = c[1:]
    return len(h) in (3, 6) and all(ch in "0123456789abcdefABCDEF" for ch in h)
