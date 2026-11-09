"""
Renderers for G-Stroke 2.0. Both draw the same ink parts, so the SVG and the PNG match.

  SketchSVG  scalable vector: every stroke is a group of filled, variable-width ribbons (the master copy)
  SketchPNG  the same ink composited with real transparency, at 2x and scaled down so edges are smooth
"""

from __future__ import annotations

from typing import Dict, List
from xml.sax.saxutils import escape

import cv2
import numpy as np
from PIL import Image, ImageColor

from .sketch import Part, Sketch


def _d(poly: np.ndarray) -> str:
    pts = poly
    return "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts) + "Z"


class SketchSVG:
    @staticmethod
    def render(sk: Sketch, title: str = "") -> str:
        out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {sk.width} {sk.height}" width="{sk.width}" height="{sk.height}">']
        if title:
            out.append(f"  <title>{escape(title)}</title>")
        out.append("  <desc>Drawn with Nebula G-Stroke 2.0 (brushes, strokes and strands).</desc>")
        out.append(f'  <rect width="100%" height="100%" fill="{sk.paper}"/>')
        out.append('  <g stroke="none">')
        # group consecutive parts of the same layer / colour / opacity so the file stays small
        run: List[Part] = []

        def flush() -> None:
            if not run:
                return
            head = run[0]
            d = " ".join(_d(p.polygon) for p in run)
            op = f' fill-opacity="{head.opacity:.2f}"' if head.opacity < 0.995 else ""
            out.append(f'    <path fill="{head.color}"{op} d="{d}"/>')
            run.clear()

        for part in sk.parts():
            if len(part.polygon) < 3:
                continue
            if run and (part.color != run[0].color or abs(part.opacity - run[0].opacity) > 1e-6 or len(run) >= 40):
                flush()
            run.append(part)
        flush()
        out.append("  </g>")
        out.append("</svg>")
        return "\n".join(out)


class SketchPNG:
    SUPERSAMPLE = 2
    MAX_SIDE = 6000

    @classmethod
    def render(cls, sk: Sketch) -> Image.Image:
        s = cls.SUPERSAMPLE
        while s > 1 and max(sk.width, sk.height) * s > cls.MAX_SIDE:
            s -= 1
        W, H = sk.width * s, sk.height * s
        paper = np.array(ImageColor.getrgb(sk.paper), dtype=np.float32)
        canvas = np.empty((H, W, 3), dtype=np.float32)
        canvas[:] = paper
        # a barely visible paper tooth, so graphite and charcoal have something to catch on
        rng = np.random.default_rng(sk.seed)
        tooth = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 0.9 * s)
        tooth = tooth / (float(tooth.std()) + 1e-6)
        canvas += (tooth * 1.4)[..., None]

        colors: Dict[str, np.ndarray] = {}
        shift = 4
        for part in sk.parts():
            poly = part.polygon
            if len(poly) < 3:
                continue
            x0, y0 = np.floor(poly.min(axis=0) * s).astype(int) - 2
            x1, y1 = np.ceil(poly.max(axis=0) * s).astype(int) + 3
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            pts = np.round((poly * s - np.array([x0, y0])) * (1 << shift)).astype(np.int32)
            cv2.fillPoly(mask, [pts], 255, lineType=cv2.LINE_AA, shift=shift)
            col = colors.get(part.color)
            if col is None:
                col = colors[part.color] = np.array(ImageColor.getrgb(part.color), dtype=np.float32)
            a = (mask.astype(np.float32) / 255.0 * part.opacity)[..., None]
            roi = canvas[y0:y1, x0:x1]
            roi *= 1 - a
            roi += a * col
        img = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), "RGB")
        if s > 1:
            img = img.resize((sk.width, sk.height), Image.Resampling.LANCZOS)
        return img
