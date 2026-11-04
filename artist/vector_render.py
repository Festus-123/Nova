"""
Vector renderer for The Artist (Ability 0.01).

Writes a ResolvedDrawing (exact boolean-resolved geometry) as:
  - SVG: clean, scalable paths: this is the master copy, and stays sharp at any size
  - PNG: the same geometry rasterised at 3x and scaled down, so edges are smooth (anti-aliased)
"""

from __future__ import annotations

from typing import List
from xml.sax.saxutils import escape

from PIL import Image, ImageColor, ImageDraw
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from .construct import ResolvedDrawing, ResolvedItem


def _polygons(g: BaseGeometry) -> List[Polygon]:
    if g is None or g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    if isinstance(g, MultiPolygon):
        return list(g.geoms)
    return [p for part in getattr(g, "geoms", []) for p in _polygons(part)]


def _ring_path(coords) -> str:
    pts = list(coords)
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 3:
        return ""
    return "M" + " L".join(f"{x:.2f} {y:.2f}" for x, y in pts) + " Z"


def _path_data(g: BaseGeometry) -> str:
    parts = []
    for poly in _polygons(g):
        parts.append(_ring_path(poly.exterior.coords))
        for hole in poly.interiors:
            parts.append(_ring_path(hole.coords))
    return " ".join(p for p in parts if p)


class VectorSVG:
    @staticmethod
    def render(d: ResolvedDrawing, title: str = "") -> str:
        out = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {d.width} {d.height}" width="{d.width}" height="{d.height}">',
        ]
        if title:
            out.append(f"  <title>{escape(title)}</title>")
        out.append(f'  <rect width="100%" height="100%" fill="{d.background}"/>')
        out.append('  <g stroke-linejoin="round" stroke-linecap="round">')
        for item in d.items:
            path = _path_data(item.geometry)
            if not path:
                continue
            attrs = [f'id="{escape(item.id)}"', f'd="{path}"', 'fill-rule="evenodd"']
            attrs.append(f'fill="{item.fill}"' if item.fill else 'fill="none"')
            if item.stroke and item.stroke_width > 0:
                attrs.append(f'stroke="{item.stroke}" stroke-width="{item.stroke_width:.2f}"')
            out.append(f"    <path {' '.join(attrs)}/>")
        out.append("  </g>")
        out.append("</svg>")
        return "\n".join(out)


class VectorPNG:
    SUPERSAMPLE = 3
    MAX_SIDE = 7000

    @classmethod
    def render(cls, d: ResolvedDrawing) -> Image.Image:
        s = cls.SUPERSAMPLE
        while s > 1 and max(d.width, d.height) * s > cls.MAX_SIDE:
            s -= 1
        W, H = d.width * s, d.height * s
        canvas = Image.new("RGB", (W, H), ImageColor.getrgb(d.background))

        for item in d.items:
            polys = _polygons(item.geometry)
            if not polys:
                continue
            if item.fill:
                canvas.paste(ImageColor.getrgb(item.fill), mask=cls._mask(polys, W, H, s))
            if item.stroke and item.stroke_width > 0:
                # The outline is drawn as a thin band along the shape's boundary, so it joins up smoothly.
                band = [item.geometry.boundary.buffer(item.stroke_width / 2.0, quad_segs=8)]
                canvas.paste(ImageColor.getrgb(item.stroke), mask=cls._mask([p for b in band for p in _polygons(b)], W, H, s))

        if s > 1:
            canvas = canvas.resize((d.width, d.height), Image.Resampling.LANCZOS)
        return canvas

    @staticmethod
    def _mask(polys: List[Polygon], W: int, H: int, s: int) -> Image.Image:
        mask = Image.new("L", (W, H), 0)
        draw = ImageDraw.Draw(mask)
        for poly in polys:
            draw.polygon([(x * s, y * s) for x, y in poly.exterior.coords], fill=255)
            for hole in poly.interiors:
                draw.polygon([(x * s, y * s) for x, y in hole.coords], fill=0)
        return mask
