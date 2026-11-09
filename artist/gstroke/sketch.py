"""
The G-Stroke 2.0 drawing model: points -> strokes -> strands -> sketch.

  point   (x, y, pressure)       a sample of the pen's path
  stroke  a path of points drawn with one brush in one colour; it becomes a smooth, variable-width ribbon of ink
  strand  a bundle of near-parallel strokes drawn together (hatching, hair, grass, folds, wood grain)
  sketch  an ordered list of strokes on a sheet of paper, the thing the renderers turn into SVG / PNG

All coordinates here are canvas pixels. A stroke is turned into one or more `Part`s (filled polygons with an opacity):
a single-pass brush such as a pen gives one, a pencil or charcoal gives several, which is what makes them look grainy.
The SVG and PNG renderers both draw the same parts, so the two files match.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .brushes import Brush, get_brush

Point = Tuple[float, float, float]  # x, y, pressure (0..1)

MAX_PARTS = 120_000


@dataclass
class Stroke:
    points: List[Point]
    brush: str = "pencil"
    color: str = "#1E1E1E"
    width: float = 2.0            # nominal width in pixels, before the brush's own scale
    opacity: Optional[float] = None  # multiplies the brush opacity
    group: str = ""               # strand / layer name (becomes an SVG group)
    closed: bool = False


@dataclass
class Part:
    polygon: np.ndarray  # (N, 2) float
    color: str
    opacity: float
    group: str = ""


@dataclass
class Sketch:
    width: int
    height: int
    paper: str = "#FBFAF6"
    strokes: List[Stroke] = field(default_factory=list)
    seed: int = 7
    meta: Dict[str, Any] = field(default_factory=dict)
    _parts: Optional[List[Part]] = field(default=None, repr=False, compare=False)

    @property
    def stroke_count(self) -> int:
        return len(self.strokes)

    def parts(self) -> List[Part]:
        if self._parts is None:
            out: List[Part] = []
            for i, st in enumerate(self.strokes):
                out.extend(stroke_parts(st, self.seed, i))
                if len(out) >= MAX_PARTS:
                    break
            self._parts = out
        return self._parts


# --------------------------------------------------------------------------------------------------------------
# Strands
# --------------------------------------------------------------------------------------------------------------


def make_strand(
    guide: Sequence[Tuple[float, float]],
    count: int,
    spread: float,
    fan: float = 0.0,
    brush: str = "pencil",
    color: str = "#1E1E1E",
    width: float = 1.5,
    variation: float = 0.25,
    group: str = "",
    seed: int = 1,
) -> List[Stroke]:
    """
    A strand is `count` strokes that follow the guide path side by side, `spread` pixels apart. `fan` spreads them
    out towards the end of the guide (hair, grass, a brush's bristles). `variation` makes each member a little
    different in length, offset and weight so it reads as drawn by hand.
    """
    pts = np.asarray([(float(x), float(y)) for x, y in guide], dtype=float)
    if len(pts) < 2 or count < 1:
        return []
    rng = np.random.default_rng(seed & 0xFFFFFFFF)
    centre = resample(pts, 3.0)
    normals = _normals(centre)
    n = len(centre)
    out: List[Stroke] = []
    for k in range(count):
        slot = k - (count - 1) / 2.0  # -(count-1)/2 .. +(count-1)/2 across the strand
        a = int(rng.uniform(0, variation * 0.5) * n)
        b = n - int(rng.uniform(0, variation) * n)
        if b - a < 3:
            a, b = 0, n
        idx = np.arange(a, b)
        offs = slot * spread * (1.0 + fan * idx / max(n - 1, 1)) + rng.normal(0, spread * variation * 0.15)
        line = centre[idx] + normals[idx] * offs[:, None]
        press = 0.55 + 0.45 * np.sin(np.linspace(0, math.pi, len(idx)))
        out.append(
            Stroke(
                points=[(float(x), float(y), float(p)) for (x, y), p in zip(line, press)],
                brush=brush,
                color=color,
                width=max(0.3, width * (1 + rng.uniform(-variation, variation) * 0.5)),
                opacity=float(np.clip(1 + rng.uniform(-variation, 0.0), 0.4, 1.0)),
                group=group,
            )
        )
    return out


# --------------------------------------------------------------------------------------------------------------
# Geometry: smoothing, width profile, ribbon
# --------------------------------------------------------------------------------------------------------------


def _normals(pts: np.ndarray) -> np.ndarray:
    if len(pts) < 2:
        return np.zeros_like(pts)
    d = np.gradient(pts, axis=0)
    ln = np.hypot(d[:, 0], d[:, 1])
    ln[ln < 1e-9] = 1.0
    d = d / ln[:, None]
    return np.stack([-d[:, 1], d[:, 0]], axis=1)


def resample(pts: np.ndarray, step: float) -> np.ndarray:
    """Evenly spaced points along a polyline."""
    if len(pts) < 2:
        return pts
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total < 1e-6:
        return pts[:1]
    n = max(2, int(math.ceil(total / step)) + 1)
    t = np.linspace(0, total, n)
    return np.stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])], axis=1)


def smooth_path(xyp: np.ndarray, closed: bool = False, step: float = 1.6) -> np.ndarray:
    """
    Catmull-Rom spline through (x, y, pressure) control points, resampled every `step` pixels.
    This is what turns a jagged list of points into a flowing pen line.
    """
    P = xyp
    n = len(P)
    if n == 1:
        return np.repeat(P, 2, axis=0)
    if n == 2:
        ext = P
    else:
        if closed:
            ext = np.vstack([P[-1:], P, P[:2]])
        else:
            ext = np.vstack([2 * P[:1] - P[1:2], P, 2 * P[-1:] - P[-2:-1]])
    segs = n if closed else n - 1
    out: List[np.ndarray] = []
    for i in range(segs):
        if n == 2:
            p0, p1, p2, p3 = P[0], P[0], P[1], P[1]
        else:
            p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        length = float(np.hypot(*(p2[:2] - p1[:2])))
        m = max(2, int(math.ceil(length / step)))
        t = np.linspace(0, 1, m, endpoint=False)[:, None]
        t2, t3 = t * t, t * t * t
        seg = 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
        out.append(seg)
    out.append((P[0] if closed else P[-1])[None, :])
    res = np.vstack(out)
    res[:, 2] = np.clip(res[:, 2], 0.0, 1.0)
    return res


def _noise(rng: np.random.Generator, n: int, spacing_px: float, step: float) -> np.ndarray:
    """Smooth 1-D noise in -1..1 with a feature size of about `spacing_px`."""
    knots = max(2, int(n * step / max(spacing_px, 1e-6)) + 2)
    v = rng.uniform(-1, 1, knots)
    return np.interp(np.linspace(0, knots - 1, n), np.arange(knots), v)


def stroke_parts(st: Stroke, seed: int, index: int) -> List[Part]:
    """One stroke -> the filled polygons that make its ink."""
    brush: Brush = get_brush(st.brush)
    if len(st.points) < 1:
        return []
    raw = np.asarray(st.points, dtype=float)
    if raw.shape[1] == 2:
        raw = np.hstack([raw, np.full((len(raw), 1), 0.7)])
    step = 2.2
    path = smooth_path(raw, closed=st.closed and len(raw) > 2, step=step)
    n = len(path)
    xy, pressure = path[:, :2], path[:, 2]
    seglen = np.hypot(*np.diff(xy, axis=0).T) if n > 1 else np.zeros(0)
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    total = float(s[-1])
    if total < 0.2:
        return _dot(st, brush, xy[0])
    nrm = _normals(xy)
    rng0 = np.random.default_rng(((seed * 1_000_003) + index * 7919) & 0xFFFFFFFF)

    base_w = max(0.25, st.width * brush.width_scale)
    # width along the stroke: taper in / out, pressure, a little unevenness
    taper = min(brush.taper_px, total * 0.45)
    ramp = np.ones(n)
    if taper > 0.5 and not (st.closed and len(raw) > 2):
        f_in = np.clip(s / taper, 0, 1)
        f_out = np.clip((total - s) / taper, 0, 1)
        f = np.minimum(f_in, f_out)
        f = f * f * (3 - 2 * f)  # smoothstep
        ramp = brush.min_width + (1 - brush.min_width) * f
    press = 1 - brush.pressure_gain + brush.pressure_gain * pressure

    parts: List[Part] = []
    opacity_base = brush.opacity * (st.opacity if st.opacity is not None else 1.0)
    for li, (off_w, w_mul, o_mul) in enumerate(brush.layers):
        rng = np.random.default_rng(int(rng0.integers(0, 2**31 - 1)) + li)
        w = base_w * w_mul * ramp * press
        if brush.width_noise > 0:
            w = w * (1 + brush.width_noise * (0.7 * _noise(rng, n, 7.0, step) + 0.3 * _noise(rng, n, 2.5, step)))
        w = np.maximum(w, 0.12)
        shift = np.full(n, off_w * base_w)
        if brush.wobble > 0:
            shift = shift + brush.wobble * _noise(rng, n, 28.0, step) * min(1.0, 0.4 + base_w / 3.0)
        centre = xy + nrm * shift[:, None]
        poly = _ribbon(centre, nrm, w, closed=st.closed and len(raw) > 2)
        if poly is not None:
            parts.append(Part(poly, st.color, float(np.clip(opacity_base * o_mul, 0.03, 1.0)), st.group))
    return parts


def _ribbon(c: np.ndarray, nrm: np.ndarray, w: np.ndarray, closed: bool) -> Optional[np.ndarray]:
    if len(c) < 2:
        return None
    half = (w / 2.0)[:, None]
    left = c + nrm * half
    right = c - nrm * half
    if closed:
        # a ring: outer edge then the inner edge backwards (drawn as one even-odd-safe polygon via a thin seam)
        return np.vstack([left, left[:1], right[:1], right[::-1]])
    cap = _cap(c[-1], nrm[-1], float(half[-1, 0]))
    cap0 = _cap(c[0], -nrm[0], float(half[0, 0]))
    return np.vstack([left, cap, right[::-1], cap0])


def _cap(p: np.ndarray, normal: np.ndarray, r: float) -> np.ndarray:
    """A rounded end: a half circle running from the stroke's left edge round to its right edge."""
    if r < 0.35:
        return np.zeros((0, 2))
    t = np.array([normal[1], -normal[0]])  # the direction the stroke is heading
    a = np.linspace(0, math.pi, 5)[1:-1]
    return np.array([p + r * (normal * math.cos(x) + t * math.sin(x)) for x in a])


def _dot(st: Stroke, brush: Brush, p: np.ndarray) -> List[Part]:
    r = max(0.3, st.width * brush.width_scale * 0.5)
    a = np.linspace(0, 2 * math.pi, 10, endpoint=False)
    poly = np.stack([p[0] + r * np.cos(a), p[1] + r * np.sin(a)], axis=1)
    return [Part(poly, st.color, float(np.clip(brush.opacity * (st.opacity or 1.0), 0.03, 1.0)), st.group)]
