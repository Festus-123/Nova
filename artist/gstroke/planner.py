"""
Prompt -> sketch for G-Stroke 2.0.

Nebula's language model does not write pixels or SVG. It acts as the artist's hand: it decides where each pen stroke
goes, which brush makes it, how hard it presses, and where bundles of strokes (strands) and hatching add form and
shadow. This module asks for that "sketch program" as JSON, checks it, and builds a `Sketch` from it.

Program format (all coordinates on the design grid: x 0..100, y 0..H, y grows downward):

  {"title": "...",
   "strokes":  [{"brush": "ink", "width": 0.5, "color": "#222222", "points": [[x,y], [x,y,pressure], ...], "closed": false},
                {"shape": "ellipse", "cx":.., "cy":.., "rx":.., "ry":.., "rotate": 0, "start": 0, "end": 360, ...}],
   "strands":  [{"guide": [[x,y], ...], "count": 6, "spread": 0.5, "fan": 0.3, "brush": "pencil", "width": 0.25}],
   "hatches":  [{"region": [[x,y], ...], "angle": 45, "spacing": 0.9, "brush": "pencil", "width": 0.2, "opacity": 0.7}]}
"""

from __future__ import annotations

import json
import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..construct import DESIGN_WIDTH
from ..errors import GeometricValidationError
from ..llm_planner import LLMClient, PlanningUnavailableError, parse_json
from .brushes import BRUSHES, get_brush
from .imaging import v2_style
from .render import SketchPNG
from .sketch import Sketch, Stroke, make_strand

MAX_STROKES = 1600
MAX_POINTS = 400
PAPER = "#FBFAF6"


class SketchProgramError(GeometricValidationError):
    """The model's sketch program could not be used."""


SYSTEM_SKETCH = """You are Nebula's drawing hand, working in G-Stroke 2.0. You draw like a skilled illustrator making a clean sketch: confident pen strokes, light construction, tone built from hatching. You never write SVG or pixels. You decide where each stroke goes, which brush draws it, and how hard it presses.

CANVAS: a design grid. x runs 0 to {W} (left to right), y runs 0 to {H} (top to bottom, y grows DOWNWARD). Keep everything inside it, the subject filling about 80% of the canvas and centred.

BRUSHES: pen (clean even ink), fineliner (thin, precise), ink (calligraphic: tapers hard, swells in the middle), pencil (soft graphite, grainy), charcoal (broad, smudgy), marker (bold, blunt). Default brush for this drawing: {brush}.

OUTPUT (JSON only):
{{
 "title": "...",
 "strokes": [
   {{"brush": "ink", "width": 0.5, "color": "#1C1C1E", "points": [[x,y],[x,y],[x,y],...]}},
   {{"shape": "ellipse", "cx": 50, "cy": 40, "rx": 10, "ry": 8, "rotate": 0, "start": 0, "end": 360, "width": 0.4}}
 ],
 "strands": [{{"guide": [[x,y],[x,y],[x,y]], "count": 7, "spread": 0.45, "fan": 0.3, "brush": "pencil", "width": 0.22}}],
 "hatches": [{{"region": [[x,y],[x,y],[x,y],[x,y]], "angle": 45, "spacing": 0.9, "brush": "pencil", "width": 0.2, "opacity": 0.7}}]
}}

STROKE RULES (this is what makes it look drawn, not plotted):
 1. A stroke is a flowing path. Give 4 to 30 points along a curve; the engine fits a smooth spline through them, so do NOT repeat points to fake curves and do NOT draw polygons with sharp corners unless the object has corners. Use "shape":"ellipse" for circles, ellipses and arcs (start/end are degrees, 0 = right, 90 = DOWN).
 2. Draw an outline as a few long strokes rather than many tiny ones; break a stroke where a real edge ends or where the line would change direction sharply. Overlapping the ends slightly looks natural.
 3. Vary line weight: "width" is in grid units. Silhouettes and the nearest edges 0.5 to 0.9, inner detail 0.25 to 0.45, texture and shading 0.12 to 0.25. Optional third number in a point is pressure 0..1 (harder = thicker).
 4. Build form and shadow with "hatches": a region (polygon) filled with parallel lines at an angle. Lay a lighter pass first, then a second pass at another angle ("angle" + 90, or +45) on the darkest parts to cross-hatch. Hatch spacing 0.6 to 1.4, opacity 0.4 to 0.9.
 5. Use "strands" for anything made of many similar lines: hair, fur, grass, feathers, wood grain, folds, rays. "guide" is the path they follow; "count" lines run side by side "spread" apart, "fan" opens them out along the guide.
 6. Aim for 40 to 160 strokes and a few hatches for a typical subject. Real proportions, correct perspective, every part a viewer needs to recognise the subject.
 7. Colour: {color_note}
Return only the JSON object."""

SYSTEM_REVIEW = """You are Nebula's drawing hand reviewing your own sketch. You will see the sketch as an image together with the sketch program that made it.
Compare it with what was asked. Look for: missing or misplaced parts, wrong proportions, strokes that stray, weak contrast, flat areas that need hatching, and anything that makes the subject hard to recognise.
If it already looks right reply {{"ok": true}}. Otherwise reply with the COMPLETE corrected sketch program (same format; grid x 0..{W}, y 0..{H}, y grows downward). Reply with JSON only."""


def _color_note(style: str) -> str:
    if style == "color":
        return 'colour is wanted. Give every stroke, strand and hatch a "color" (#RRGGBB) that suits the part: dark tints of the part\'s real colour for line work, the real colour for hatching. Keep a small harmonious palette.'
    return 'monochrome. Use near-black "#1C1C1E" for ink and "#3A3A3C" for pencil; no other colours.'


# --------------------------------------------------------------------------------------------------------------
# Program -> Sketch
# --------------------------------------------------------------------------------------------------------------


def _num(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _colour(v: Any, fallback: str) -> str:
    if isinstance(v, str):
        t = v.strip()
        if len(t) in (4, 7) and t.startswith("#") and all(c in "0123456789abcdefABCDEF" for c in t[1:]):
            if len(t) == 4:
                t = "#" + "".join(c * 2 for c in t[1:])
            return t.upper()
    return fallback


def _pts(raw: Any, k: float, grid_h: float, limit: int = MAX_POINTS) -> List[Tuple[float, float, float]]:
    out: List[Tuple[float, float, float]] = []
    if not isinstance(raw, list):
        return out
    for p in raw[:limit]:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            continue
        x = min(max(_num(p[0]), -5.0), DESIGN_WIDTH + 5.0)
        y = min(max(_num(p[1]), -5.0), grid_h + 5.0)
        pr = min(max(_num(p[2], 0.7), 0.0), 1.0) if len(p) > 2 else 0.7
        out.append((x * k, y * k, pr))
    return out


def _ellipse(d: Dict[str, Any], k: float) -> List[Tuple[float, float, float]]:
    cx, cy = _num(d.get("cx"), 50), _num(d.get("cy"), 50)
    rx = max(0.1, _num(d.get("rx", d.get("r")), 5)); ry = max(0.1, _num(d.get("ry", d.get("r")), rx))
    rot = math.radians(_num(d.get("rotate", d.get("angle")), 0))
    a0, a1 = _num(d.get("start"), 0), _num(d.get("end"), 360)
    if a1 <= a0:
        a1 += 360
    full = (a1 - a0) >= 359.0
    n = max(8, int(min(360, a1 - a0) / 12))
    out = []
    for i in range(n + (0 if full else 1)):
        a = math.radians(a0 + (a1 - a0) * i / n)
        x, y = rx * math.cos(a), ry * math.sin(a)
        out.append(((cx + x * math.cos(rot) - y * math.sin(rot)) * k, (cy + x * math.sin(rot) + y * math.cos(rot)) * k, 0.7))
    return out


def program_to_sketch(program: Dict[str, Any], width: int, height: int, style: str = "line", brush: str = "pencil", seed: int = 7) -> Sketch:
    """Validates a sketch program (clamping and dropping anything unusable) and builds the Sketch."""
    if not isinstance(program, dict):
        raise SketchProgramError("The sketch program is not an object.")
    k = width / DESIGN_WIDTH
    grid_h = height / k
    mono = style != "color"
    ink, graphite = "#1C1C1E", "#3A3A3C"
    default_brush = get_brush(brush).name
    strokes: List[Stroke] = []

    def brush_of(d: Dict[str, Any]) -> str:
        return get_brush(d.get("brush"), default_brush).name

    def color_of(d: Dict[str, Any], b: str) -> str:
        fallback = graphite if b in ("pencil", "charcoal") else ink
        return fallback if mono else _colour(d.get("color"), fallback)

    for i, d in enumerate(program.get("strokes") or []):
        if not isinstance(d, dict) or len(strokes) >= MAX_STROKES:
            continue
        pts = _ellipse(d, k) if str(d.get("shape", "")).lower() in ("ellipse", "circle", "arc") else _pts(d.get("points"), k, grid_h)
        if len(pts) < 2:
            continue
        b = brush_of(d)
        w = max(0.2, min(_num(d.get("width"), 0.4), 4.0) * k)
        full = str(d.get("shape", "")).lower() in ("ellipse", "circle") and abs(_num(d.get("end"), 360) - _num(d.get("start"), 0)) >= 359
        strokes.append(Stroke(pts, b, color_of(d, b), w, group=str(d.get("id") or f"stroke-{i}")[:40], closed=bool(d.get("closed")) or full))

    for i, d in enumerate(program.get("strands") or []):
        if not isinstance(d, dict) or len(strokes) >= MAX_STROKES:
            continue
        guide = [(x, y) for x, y, _ in _pts(d.get("guide"), k, grid_h, 120)]
        if len(guide) < 2:
            continue
        b = brush_of(d)
        for st in make_strand(
            guide, int(min(max(_num(d.get("count"), 5), 1), 24)), max(0.05, _num(d.get("spread"), 0.4)) * k, _num(d.get("fan"), 0.0),
            b, color_of(d, b), max(0.2, min(_num(d.get("width"), 0.25), 2.0) * k), 0.25, f"strand-{i}", seed + i * 31,
        ):
            strokes.append(st)

    rng = np.random.default_rng(seed)
    for i, d in enumerate(program.get("hatches") or []):
        if not isinstance(d, dict):
            continue
        region = [(x, y) for x, y, _ in _pts(d.get("region"), k, grid_h, 200)]
        if len(region) < 3:
            continue
        b = brush_of(d)
        strokes.extend(
            _hatch(region, width, height, _num(d.get("angle"), 45), max(0.3, _num(d.get("spacing"), 1.0)) * k,
                   b, color_of(d, b), max(0.2, min(_num(d.get("width"), 0.2), 2.0) * k), min(max(_num(d.get("opacity"), 0.75), 0.15), 1.0), rng, f"hatch-{i}")
        )

    if not strokes:
        raise SketchProgramError("The sketch program had no usable strokes.")
    title = str(program.get("title") or "sketch")[:80]
    return Sketch(width, height, PAPER, strokes[: MAX_STROKES * 8], seed=seed, meta=dict(title=title, style=style, brush=default_brush))


def _hatch(region, width, height, angle, spacing, brush, colour, line_w, opacity, rng, group) -> List[Stroke]:
    """Parallel hand-drawn lines inside a polygon."""
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(np.array(region)).astype(np.int32)], 1)
    if not mask.any():
        return []
    diag = int(math.hypot(width, height)) + 4
    M = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    M[0, 2] += diag / 2.0 - width / 2.0
    M[1, 2] += diag / 2.0 - height / 2.0
    rot = cv2.warpAffine(mask, M, (diag, diag), flags=cv2.INTER_NEAREST)
    inv = cv2.invertAffineTransform(M)
    out: List[Stroke] = []
    y = float(rng.uniform(0, spacing))
    while y < diag - 1 and len(out) < 1500:
        row = rot[int(y)]
        if row.any():
            edge = np.diff(np.concatenate([[0], (row > 0).astype(np.int8), [0]]))
            for x0, x1 in zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)):
                a, b = float(x0) + rng.uniform(0, 2), float(x1) - rng.uniform(0, 2)
                if b - a < spacing * 1.5:
                    continue
                n = max(2, int((b - a) / 20) + 1)
                pts = np.stack([np.linspace(a, b, n), y + rng.normal(0, 0.25, n)], axis=1)
                back = pts @ inv[:, :2].T + inv[:, 2]
                out.append(Stroke([(float(px), float(py), 0.6) for px, py in back], brush, colour, line_w, opacity=opacity, group=group))
        y += spacing * float(rng.uniform(0.9, 1.12))
    return out


# --------------------------------------------------------------------------------------------------------------
# The planner
# --------------------------------------------------------------------------------------------------------------


class SketchPlanner:
    """Turns a request into a sketch via the language model."""

    def __init__(self, client: Optional[LLMClient] = None):
        self.llm = client or LLMClient()

    @property
    def available(self) -> bool:
        return self.llm.available

    def plan(
        self,
        prompt: str,
        width: int = 800,
        height: int = 600,
        style: str = "line",
        quality: str = "balanced",
        brush: str = "pencil",
        budget_s: float = 120.0,
        on_stage: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        style = v2_style(style)
        t0 = time.time()
        W, H = DESIGN_WIDTH, round(DESIGN_WIDTH * height / width, 2)
        timeline: List[Dict[str, Any]] = []
        left = lambda: max(8.0, budget_s - (time.time() - t0))  # noqa: E731
        system = SYSTEM_SKETCH.format(W=W, H=H, brush=get_brush(brush).name, color_note=_color_note(style))

        def stage(label: str, name: str, fn: Callable[[], Any]) -> Any:
            if on_stage:
                try:
                    on_stage(label)
                except Exception:  # progress reporting must never stop a drawing
                    pass
            s = time.time()
            try:
                out = fn()
                timeline.append({"stage": name, "ms": int((time.time() - s) * 1000), "ok": True})
                return out
            except PlanningUnavailableError:
                raise
            except Exception as e:
                timeline.append({"stage": name, "ms": int((time.time() - s) * 1000), "ok": False, "note": str(e)[:160]})
                raise

        program: Optional[Dict[str, Any]] = None
        sketch: Optional[Sketch] = None
        last_error = ""
        for attempt in range(2):
            ask = f"Draw: {prompt}" + (f"\n\nYour previous program could not be used: {last_error}\nReturn a corrected, complete program." if last_error else "")
            raw = stage("drawing with pen and brush" if attempt == 0 else "fixing a problem and trying again", "sketch" if attempt == 0 else "sketch (retry)",
                        lambda: self.llm.complete(system, ask, timeout=min(110, left()), max_tokens=14000))
            try:
                program = parse_json(raw)
                sketch = program_to_sketch(program, width, height, style, brush)
                break
            except (ValueError, GeometricValidationError) as e:
                last_error = str(e)[:300]
                program = sketch = None
            if left() < 25:
                break
        if sketch is None or program is None:
            raise PlanningUnavailableError(f"Nebula could not produce a usable sketch ({last_error or 'no usable program'}).")

        if quality == "best":
            for round_no in range(2):
                if left() < 40:
                    break
                try:
                    revised = stage("looking at the sketch and refining it", f"review {round_no + 1}", lambda: self._review(prompt, program, sketch, W, H, left()))
                except Exception:
                    break
                if revised is None:
                    break
                try:
                    new_sketch = program_to_sketch(revised, width, height, style, brush)
                except GeometricValidationError:
                    break
                program, sketch = revised, new_sketch

        sketch.meta["timeline"] = timeline
        return {"program": program, "sketch": sketch, "timeline": timeline, "model": self.llm.last_model}

    def _review(self, prompt: str, program: Dict[str, Any], sketch: Sketch, W: float, H: float, remaining: float) -> Optional[Dict[str, Any]]:
        import base64
        import io

        img = SketchPNG.render(sketch)
        img.thumbnail((720, 720))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        text = f"Requested drawing: {prompt}\n\nCurrent sketch program:\n{json.dumps(program, separators=(',', ':'))[:16000]}"
        content = [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")}}]
        data = parse_json(self.llm.complete(SYSTEM_REVIEW.format(W=W, H=H), content, timeout=min(90, remaining), max_tokens=14000))
        if isinstance(data, dict) and data.get("ok") is True and "strokes" not in data:
            return None
        return data
