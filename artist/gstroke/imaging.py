"""
Image -> sketch for G-Stroke 2.0.

Where G-Stroke 1.0 fits circles and polygons to the regions of a picture, 2.0 draws it the way a person sketches:

  1. CONTOURS   Two scales of edge detection find the structure (big outlines) and the detail (small marks). The
                edge pixels are walked into long pen paths, smoothed, and drawn as strokes whose weight follows how
                strong the edge is: heavy where the picture has a clear edge, light where it fades.
  2. SHADING    The picture's tones are cut into levels. Each darker level is covered with hatching at a new angle
                (cross-hatching), drawn as short hand-length strokes with ragged ends, grouped into strands.
  3. COLOUR     In `color` style each stroke picks up the colour of the picture where it lies.

Same picture + same settings always gives the same sketch (a fixed seed drives every bit of "hand" randomness).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from ..vision import ImageProblem, load_image
from .brushes import get_brush
from .sketch import Sketch, Stroke

# processing size, max contour strokes, max hatch strokes, shortest contour kept (processing px), hatch spacing divisor
DETAIL: Dict[str, Dict[str, Any]] = {
    "low": dict(side=560, contours=1400, hatches=2500, min_len=16, spacing=95),
    "medium": dict(side=800, contours=3200, hatches=6000, min_len=11, spacing=125),
    "high": dict(side=1100, contours=6000, hatches=11000, min_len=8, spacing=165),
}

# darkness thresholds (tone below this gets this pass) and the angle of each pass, lightest pass first
LEVELS = {
    "line": [(0.20, 45)],
    "shaded": [(0.66, 45), (0.48, 135), (0.32, 12), (0.17, 98)],
    "color": [(0.84, 45), (0.64, 135), (0.44, 12), (0.26, 98)],
}
LEVEL_OPACITY = [0.55, 0.7, 0.82, 0.95]

STYLE_ALIASES = {"line": "line", "solid": "shaded", "shaded": "shaded", "shade": "shaded", "color": "color", "colour": "color"}


def v2_style(value: Optional[str]) -> str:
    return STYLE_ALIASES.get((value or "line").strip().lower(), "line")


def _hex(rgb) -> str:
    r, g, b = (int(max(0, min(255, round(float(v))))) for v in rgb[:3])
    return f"#{r:02X}{g:02X}{b:02X}"


class ImageSketcher:
    def __init__(self, width: int = 800, height: int = 600, style: str = "line", detail: str = "medium", brush: str = "pencil", seed: int = 7):
        self.width, self.height = width, height
        self.style = v2_style(style)
        self.detail = detail if detail in DETAIL else "medium"
        self.cfg = DETAIL[self.detail]
        self.brush_name = get_brush(brush).name
        self.seed = seed
        self.ink = "#1C1C1E"

    # ---------------------------------------------------------------------------------------------------------

    def sketch(self, data: bytes) -> Sketch:
        img = load_image(data)
        src_w, src_h = img.size
        side = self.cfg["side"]
        img.thumbnail((side, side), Image.Resampling.LANCZOS)
        rgb = np.array(img)
        ph, pw = rgb.shape[:2]
        if pw < 16 or ph < 16:
            raise ImageProblem("The image is too small to sketch.")

        self.scale = min(self.width * 0.92 / pw, self.height * 0.92 / ph)  # processing px -> canvas px
        self.ox = (self.width - pw * self.scale) / 2.0
        self.oy = (self.height - ph * self.scale) / 2.0
        self.rgb = rgb
        self.rng = np.random.default_rng(self.seed)
        diag = math.hypot(self.width, self.height)
        self.line_w = max(1.0, diag / 480.0)  # a pen line, in canvas px

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        strokes: List[Stroke] = []
        contour_strokes = self._contours(gray)
        strokes.extend(self._hatching(gray, contour_strokes))
        strokes.extend(contour_strokes)  # outlines are inked last, over the shading

        if not strokes:
            raise ImageProblem("Nebula could not find anything to sketch in that image. Try a clearer picture or a higher detail level.")

        paper = "#FBFAF6"
        n_contour, n_hatch = len(contour_strokes), len(strokes) - len(contour_strokes)
        return Sketch(
            self.width, self.height, paper, strokes, seed=self.seed,
            meta=dict(
                source=f"image {src_w}x{src_h}", detail=self.detail, style=self.style, brush=self.brush_name,
                contour_strokes=n_contour, hatch_strokes=n_hatch,
            ),
        )

    # -- coordinates -------------------------------------------------------------------------------------------

    def _to_canvas(self, xy: np.ndarray) -> np.ndarray:
        return xy * self.scale + np.array([self.ox, self.oy])

    # -- 1. contours -------------------------------------------------------------------------------------------

    def _contours(self, gray: np.ndarray) -> List[Stroke]:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g = clahe.apply(cv2.bilateralFilter(gray, 7, 40, 7))
        coarse = self._edges(g, sigma=2.2, pct=93, floor=45)
        fine = self._edges(g, sigma=1.1, pct=90, floor=65)
        # drop fine edges that just repeat a coarse one (they would draw every outline twice)
        near = cv2.dilate(coarse[0], np.ones((5, 5), np.uint8))
        fine_mask = (fine[0] > 0) & (near == 0)
        paths: List[Tuple[np.ndarray, float, float]] = []  # (pixels, strength 0..1 per point, weight)
        for mask, mag, hi, weight in ((coarse[0] > 0, coarse[1], coarse[2], 1.25), (fine_mask, fine[1], fine[2], 0.8)):
            for pts in _trace(mask):
                if len(pts) < 3:
                    continue
                paths.append((pts, np.clip(mag[pts[:, 1], pts[:, 0]] / (hi * 1.6), 0.25, 1.0), weight))

        strokes: List[Stroke] = []
        ranked = sorted(paths, key=lambda p: -len(p[0]) * p[2])
        for pts, strength, weight in ranked:
            length = _length(pts)
            if length < self.cfg["min_len"] * (1.0 if weight > 1 else 1.4):
                continue
            pts = _smooth(pts.astype(float))
            strength = _box(strength, 9)
            if len(strength) != len(pts):
                strength = np.interp(np.linspace(0, 1, len(pts)), np.linspace(0, 1, len(strength)), strength)
            simp = cv2.approxPolyDP(pts.astype(np.float32).reshape(-1, 1, 2), 0.9, False).reshape(-1, 2)
            if len(simp) < 2:
                continue
            idx = [int(np.argmin(np.hypot(*(pts - p).T))) for p in simp]
            canvas = self._to_canvas(simp)
            colour = self.ink
            if self.style == "color":
                mid = pts[len(pts) // 2].astype(int)
                colour = self._tint(mid, 0.42)
            strokes.append(
                Stroke(
                    points=[(float(x), float(y), float(strength[i])) for (x, y), i in zip(canvas, idx)],
                    brush=self.brush_name, color=colour, width=self.line_w * weight, group="contour",
                    opacity=1.0 if weight > 1 else 0.85,
                )
            )
            if len(strokes) >= self.cfg["contours"]:
                break
        return strokes

    @staticmethod
    def _edges(g: np.ndarray, sigma: float, pct: float, floor: float) -> Tuple[np.ndarray, np.ndarray, float]:
        b = cv2.GaussianBlur(g, (0, 0), sigma)
        gx, gy = cv2.Sobel(b, cv2.CV_32F, 1, 0), cv2.Sobel(b, cv2.CV_32F, 0, 1)
        mag = np.hypot(gx, gy)
        hi = max(float(np.percentile(mag, pct)), floor)  # Sobel units: a clean 30-level step is about 100
        edges = cv2.Canny(b, hi * 0.4, hi, L2gradient=True)
        return edges, mag, hi

    # -- 2. shading --------------------------------------------------------------------------------------------

    def _hatching(self, gray: np.ndarray, contours: List[Stroke]) -> List[Stroke]:
        levels = LEVELS[self.style]
        h, w = gray.shape
        g = cv2.GaussianBlur(gray, (0, 0), 2.0).astype(np.float32)
        lo, hi = np.percentile(g, 1), np.percentile(g, 99)
        if hi - lo < 25:  # a flat picture: nothing to shade
            return []
        tone = np.clip((g - lo) / (hi - lo), 0, 1)
        colour_img = cv2.GaussianBlur(self.rgb, (0, 0), 3.0) if self.style == "color" else None
        spacing = max(2.6, math.hypot(w, h) / self.cfg["spacing"])
        brush = get_brush(self.brush_name)
        out: List[Stroke] = []
        for li, (thr, angle) in enumerate(levels):
            mask = (tone < thr).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            keep = np.zeros(n, dtype=bool)
            keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(30, 0.0006 * h * w)
            mask = keep[lab].astype(np.uint8)
            if not mask.any():
                continue
            # darker passes sit a little tighter
            sp = spacing * (1.0 - 0.07 * li)
            opacity = LEVEL_OPACITY[min(li, len(LEVEL_OPACITY) - 1)]
            for run in self._hatch_runs(mask, angle + float(self.rng.uniform(-4, 4)), sp):
                col = self.ink
                if colour_img is not None:
                    mx, my = (int(np.clip(v, 0, 1000000)) for v in run[len(run) // 2])
                    col = self._tint((min(mx, w - 1), min(my, h - 1)), 0.78 - 0.1 * li, colour_img)
                out.append(
                    Stroke(
                        points=[(float(x), float(y), 0.6) for x, y in self._to_canvas(run)],
                        brush=self.brush_name, color=col, width=self.line_w * (0.6 if brush.width_scale <= 1.2 else 0.5),
                        opacity=opacity, group=f"hatch-{li + 1}",
                    )
                )
        cap = self.cfg["hatches"]
        if len(out) > cap:  # too dense: thin it evenly rather than cutting off the bottom of the picture
            keep_idx = np.sort(self.rng.choice(len(out), cap, replace=False))
            out = [out[i] for i in keep_idx]
        return out

    def _hatch_runs(self, mask: np.ndarray, angle: float, spacing: float) -> List[np.ndarray]:
        """Parallel hatch lines across `mask`, cut into hand-length pieces with ragged ends."""
        h, w = mask.shape
        diag = int(math.hypot(w, h)) + 4
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        M[0, 2] += diag / 2.0 - w / 2.0
        M[1, 2] += diag / 2.0 - h / 2.0
        rot = cv2.warpAffine(mask, M, (diag, diag), flags=cv2.INTER_NEAREST)
        inv = cv2.invertAffineTransform(M)
        runs: List[np.ndarray] = []
        piece_min = spacing * 2.0
        y = float(self.rng.uniform(0, spacing))
        while y < diag - 1:
            row = rot[int(y)]
            if row.any():
                edge = np.diff(np.concatenate([[0], (row > 0).astype(np.int8), [0]]))
                for x0, x1 in zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)):
                    x = float(x0)
                    while x < x1:
                        seg = float(self.rng.uniform(34, 95))
                        a, b = x, min(x + seg, float(x1))
                        x = b + float(self.rng.uniform(0, 3))
                        a += float(self.rng.uniform(0, 2.2))
                        b -= float(self.rng.uniform(0, 2.2))
                        if b - a < piece_min:
                            continue
                        n = max(2, int((b - a) / 18) + 1)
                        xs = np.linspace(a, b, n)
                        ys = y + self.rng.normal(0, 0.25, n)
                        pts = np.stack([xs, ys], axis=1)
                        back = pts @ inv[:, :2].T + inv[:, 2]
                        runs.append(back)
            y += spacing * float(self.rng.uniform(0.88, 1.14))
        return runs

    # -- colour ------------------------------------------------------------------------------------------------

    def _tint(self, pt, darken: float, img: Optional[np.ndarray] = None) -> str:
        src = self.rgb if img is None else img
        x = int(np.clip(pt[0], 0, src.shape[1] - 1))
        y = int(np.clip(pt[1], 0, src.shape[0] - 1))
        return _hex(src[y, x].astype(float) * darken)


# --------------------------------------------------------------------------------------------------------------
# Tracing edge pixels into pen paths
# --------------------------------------------------------------------------------------------------------------

_N8 = [(-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (1, -1), (1, 1), (-1, 1)]  # the four straight neighbours first


def _trace(mask: np.ndarray) -> List[np.ndarray]:
    """Walks a thin edge mask into ordered pixel paths (x, y). Follows the straightest continuation at forks."""
    h, w = mask.shape
    grid = [bytearray(row.tobytes()) for row in np.pad(mask.astype(np.uint8), 1)]
    cnt = cv2.filter2D(mask.astype(np.uint8), -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT)
    ends = np.argwhere(mask & (cnt <= 2))  # a pixel plus at most one neighbour: the end of a line
    others = np.argwhere(mask)

    def walk(x: int, y: int) -> List[Tuple[int, int]]:
        path = [(x, y)]
        grid[y + 1][x + 1] = 0
        while True:
            best, best_score = None, -9.0
            dx0 = dy0 = 0.0
            if len(path) >= 4:
                dx0, dy0 = x - path[-4][0], y - path[-4][1]
                n0 = math.hypot(dx0, dy0) or 1.0
                dx0, dy0 = dx0 / n0, dy0 / n0
            for k, (dx, dy) in enumerate(_N8):
                if grid[y + 1 + dy][x + 1 + dx]:
                    score = (dx * dx0 + dy * dy0) / math.hypot(dx, dy) - k * 0.001
                    if score > best_score:
                        best, best_score = (dx, dy), score
            if best is None:
                return path
            x, y = x + best[0], y + best[1]
            grid[y + 1][x + 1] = 0
            path.append((x, y))

    out: List[np.ndarray] = []
    for (y, x) in list(ends) + list(others):
        if not grid[y + 1][x + 1]:
            continue
        fwd = walk(int(x), int(y))
        # started mid-line: also walk the other way and join, so a line is not cut in two
        back = _walk_back(grid, fwd[0])
        pts = back[::-1] + fwd if back else fwd
        out.append(np.array(pts, dtype=np.int32))
    return out


def _walk_back(grid, start: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Continues a path from its first pixel in the opposite direction, using whatever is still unvisited."""
    x, y = start
    path: List[Tuple[int, int]] = []
    while True:
        nxt = None
        for dx, dy in _N8:
            if grid[y + 1 + dy][x + 1 + dx]:
                nxt = (dx, dy)
                break
        if nxt is None:
            return path
        x, y = x + nxt[0], y + nxt[1]
        grid[y + 1][x + 1] = 0
        path.append((x, y))


def _length(pts: np.ndarray) -> float:
    d = np.diff(pts.astype(float), axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def _box(v: np.ndarray, k: int) -> np.ndarray:
    if len(v) < 3:
        return v
    k = min(k, len(v) // 2 * 2 + 1)
    pad = np.pad(v, (k // 2, k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")[: len(v)]


def _smooth(pts: np.ndarray, k: int = 5) -> np.ndarray:
    """Moving average that keeps the two ends where they are, so lines still meet."""
    if len(pts) < k + 2:
        return pts
    pad = np.pad(pts, ((k // 2, k // 2), (0, 0)), mode="edge")
    ker = np.ones(k) / k
    out = np.stack([np.convolve(pad[:, 0], ker, mode="valid"), np.convolve(pad[:, 1], ker, mode="valid")], axis=1)[: len(pts)]
    out[0], out[-1] = pts[0], pts[-1]
    return out
