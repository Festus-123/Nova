"""
Image input for The Artist (Ability 0.01).

Nebula receives a picture, runs it through image processing, and DRAWS it with the same geometry the text prompts
use: circles, rectangles, triangles, polygons and strokes joined by union and subtraction, written out as clear
vector art (SVG, plus a smooth PNG).

How it works (all classical image processing, with OpenCV):
  1. Load and normalise the picture (orientation, transparency, size).
  2. Segment it into a handful of tones or colours (k-means) and clean the regions (median filter, morphology).
  3. Trace each region's boundary. A region with holes is an outer shape SUBTRACT its holes.
  4. FIT geometry to each region: an exact circle if it is round, a triangle or rectangle if it is one, otherwise a
     simplified polygon (Douglas-Peucker). Tiny specks are dropped.
  5. Hand the result to the Constructor as a construction program, exactly like a prompt drawing.
"""

from __future__ import annotations

import io
import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

from .errors import GeometricValidationError

MAX_BYTES = 12 * 1024 * 1024
MAX_PIXELS = 60_000_000

# per detail level: processing size, clusters (tones for line/solid, colours for color), simplification, minimum region size
DETAIL = {
    "low": dict(side=360, tones=3, colors=5, eps=0.014, min_area=0.0045, regions=60),
    "medium": dict(side=520, tones=4, colors=8, eps=0.009, min_area=0.0015, regions=140),
    "high": dict(side=720, tones=6, colors=12, eps=0.006, min_area=0.0005, regions=260),
}


class ImageProblem(GeometricValidationError):
    """The uploaded file is not a picture Nebula can use."""


def load_image(data: bytes) -> Image.Image:
    if not data:
        raise ImageProblem("The image is empty.")
    if len(data) > MAX_BYTES:
        raise ImageProblem(f"The image is too large (limit {MAX_BYTES // (1024 * 1024)} MB).")
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
        img = Image.open(io.BytesIO(data))
    except Exception:
        raise ImageProblem("That file is not a readable image (use PNG, JPG, WEBP or GIF).")
    if img.width * img.height > MAX_PIXELS:
        raise ImageProblem("The image has too many pixels.")
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, rgba)
    return img.convert("RGB")


def _hex(rgb: np.ndarray) -> str:
    r, g, b = (int(max(0, min(255, round(v)))) for v in rgb[:3])
    return f"#{r:02X}{g:02X}{b:02X}"


def _tone(gray: float) -> str:
    return "dark" if gray < 85 else ("mid" if gray < 170 else "light")


class ImageVectorizer:
    """Turns a picture into a construction program (circles, rectangles, triangles, polygons, subtractions)."""

    def __init__(self, width: int = 800, height: int = 600, style: str = "line", detail: str = "medium"):
        self.width = width
        self.height = height
        self.style = style if style in ("line", "solid", "color") else "line"
        self.detail = detail if detail in DETAIL else "medium"
        self.cfg = DETAIL[self.detail]
        self.k = width / 100.0  # canvas pixels per design unit

    def vectorize(self, data: bytes) -> Dict[str, Any]:
        img = load_image(data)
        src_w, src_h = img.size
        side = self.cfg["side"]
        img.thumbnail((side, side), Image.Resampling.LANCZOS)
        rgb = np.array(img)
        ph, pw = rgb.shape[:2]
        if pw < 8 or ph < 8:
            raise ImageProblem("The image is too small to draw.")

        # The picture, placed in the canvas with a margin. (x, y) in picture pixels -> design units.
        s = min(self.width * 0.9 / pw, self.height * 0.9 / ph)
        ox, oy = (self.width - pw * s) / 2.0, (self.height - ph * s) / 2.0
        to_units = lambda x, y: (round((x * s + ox) / self.k, 2), round((y * s + oy) / self.k, 2))  # noqa: E731

        labels, centres, bg_label = self._segment(rgb)
        regions = self._regions(labels, centres, bg_label, pw, ph)

        steps: List[Dict[str, Any]] = []
        draw: List[Dict[str, Any]] = []
        fitted = {"circle": 0, "rect": 0, "triangle": 0, "polygon": 0}
        for n, reg in enumerate(regions):
            outer, holes, kind, gray, colour = reg["outer"], reg["holes"], reg["kind"], reg["gray"], reg["colour"]
            rid = f"r{n}"
            steps.append(self._shape_step(rid, kind, outer, to_units, s))
            fitted[kind] += 1
            result = rid
            hole_ids = []
            for h_i, hole in enumerate(holes):
                hid = f"{rid}h{h_i}"
                steps.append(self._shape_step(hid, "polygon", hole, to_units, s))
                hole_ids.append(hid)
            if hole_ids:
                result = f"{rid}c"
                steps.append({"id": result, "op": "subtract", "from": rid, "remove": hole_ids})
            item: Dict[str, Any] = {"id": result, "tone": _tone(gray)}
            if self.style == "color":
                item["color"] = colour
            draw.append(item)

        if not draw:
            raise ImageProblem("Nebula could not find any shapes in that image. Try a clearer picture or a higher detail level.")

        bg = _hex(centres[bg_label]) if self.style == "color" else "#FFFFFF"
        return {
            "title": "image",
            "steps": steps,
            "draw": draw,
            "background": bg,
            "notes": f"image {src_w}x{src_h}, detail {self.detail}, fitted {fitted}",
            "fitted": fitted,
        }

    # -- 1. segmentation -------------------------------------------------------------------------------------

    def _segment(self, rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        h, w = rgb.shape[:2]
        smooth = cv2.bilateralFilter(rgb, 7, 40, 7)
        # Always segment by colour: two objects of the same brightness but different colours (a yellow sun on a pale
        # sky) must stay separate. Line and solid styles use fewer clusters, then show them as tones.
        samples = smooth.reshape(-1, 3).astype(np.float32)
        k = self.cfg["colors"] if self.style == "color" else self.cfg["tones"] + 2
        k = max(2, min(k, len(np.unique(samples, axis=0))))
        cv2.setRNGSeed(7)  # the same picture always gives the same drawing
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.5)
        _, lab, centres = cv2.kmeans(samples, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        labels = lab.reshape(h, w).astype(np.uint8)
        labels = cv2.medianBlur(labels, 5)  # removes speckle between neighbouring regions
        # The background is the cluster most common around the picture's edge.
        edge = np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
        bg_label = int(np.bincount(edge, minlength=len(centres)).argmax())
        return labels, centres, bg_label

    # -- 2. regions ------------------------------------------------------------------------------------------

    def _regions(self, labels: np.ndarray, centres: np.ndarray, bg_label: int, pw: int, ph: int) -> List[Dict[str, Any]]:
        total = float(pw * ph)
        min_area = self.cfg["min_area"] * total
        kernel = np.ones((3, 3), np.uint8)
        out: List[Dict[str, Any]] = []
        for lab in range(len(centres)):
            if lab == bg_label:
                continue  # the background is the paper
            mask = (labels == lab).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            if hier is None:
                continue
            hier = hier[0]
            for i, c in enumerate(contours):
                if hier[i][3] != -1:
                    continue  # a hole: handled with its parent
                area = cv2.contourArea(c)
                if area < min_area:
                    continue
                holes = []
                child = hier[i][2]
                while child != -1:
                    if cv2.contourArea(contours[child]) >= min_area * 0.5:
                        holes.append(self._simplify(contours[child]))
                    child = hier[child][0]
                kind, pts = self._fit(c, area)
                gray = float(0.299 * centres[lab][0] + 0.587 * centres[lab][1] + 0.114 * centres[lab][2])
                out.append(dict(outer=pts, holes=[h for h in holes if len(h) >= 3], kind=kind, area=area, gray=gray, colour=_hex(centres[lab])))
        out.sort(key=lambda r: -r["area"])  # big regions underneath, small details on top
        return out[: self.cfg["regions"]]

    def _simplify(self, contour: np.ndarray) -> np.ndarray:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, max(0.8, self.cfg["eps"] * peri), True)
        return approx.reshape(-1, 2).astype(float)

    # -- 3. fit geometry -------------------------------------------------------------------------------------

    def _fit(self, contour: np.ndarray, area: float) -> Tuple[str, Any]:
        """Snaps a region to the simplest primitive that really fits it."""
        peri = cv2.arcLength(contour, True)
        circularity = 4.0 * math.pi * area / (peri * peri) if peri > 0 else 0.0
        poly = self._simplify(contour)
        if circularity > 0.86 and area > 30:
            (cx, cy), r = cv2.minEnclosingCircle(contour)
            return "circle", (cx, cy, math.sqrt(area / math.pi))
        if len(poly) == 3:
            return "triangle", poly
        rect = cv2.minAreaRect(contour)
        rw, rh = rect[1]
        if rw > 0 and rh > 0 and area / (rw * rh) > 0.93 and len(poly) <= 6:
            return "rect", cv2.boxPoints(rect).astype(float)
        return "polygon", poly

    def _shape_step(self, sid: str, kind: str, geom: Any, to_units, s: float) -> Dict[str, Any]:
        if kind == "circle":
            cx, cy, r = geom
            x, y = to_units(cx, cy)
            return {"id": sid, "type": "circle", "cx": x, "cy": y, "r": round(r * s / self.k, 3)}
        pts = [list(to_units(x, y)) for x, y in geom]
        return {"id": sid, "type": "triangle" if kind == "triangle" else "polygon", "points": pts}
