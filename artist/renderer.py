"""
Renderer Engine for The Artist (Ability 0.01).

Generates high-precision vector SVG and anti-aliased raster PNG images
from geometric compositions and boolean operations.
"""

import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageColor

from .models import (
    DrawingPlan,
    GeometricPrimitive,
    GeometricOperation,
    PrimitiveType,
    OperationType,
    Point,
)
from .geometry import BezierMath
from .boolean import BooleanEngine


class SVGRenderer:
    """Renders DrawingPlan to vector SVG format."""

    @classmethod
    def render(cls, plan: DrawingPlan) -> str:
        w, h = plan.width, plan.height
        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
            f'  <rect width="100%" height="100%" fill="{plan.background_color}"/>',
        ]

        # Identify operations and their masks/clips
        prim_map: Dict[str, GeometricPrimitive] = {p.id: p for p in plan.primitives}
        defs_parts: List[str] = []
        op_target_map: Dict[str, GeometricOperation] = {}
        for op in plan.operations:
            op_target_map[op.target_id] = op

            # Build SVG mask or clipPath in defs
            if op.operation == OperationType.SUBTRACTION:
                mask_id = f"mask_{op.id}"
                defs_parts.append(f'    <mask id="{mask_id}">')
                defs_parts.append(f'      <rect width="100%" height="100%" fill="white"/>')
                defs_parts.append('      <g fill="black">')
                for op_id in op.operand_ids:
                    if op_id in prim_map:
                        defs_parts.append(cls._render_primitive_svg(prim_map[op_id], override_fill="black"))
                defs_parts.append('      </g>')
                defs_parts.append('    </mask>')

            elif op.operation == OperationType.INTERSECTION and op.operand_ids:
                clip_id = f"clip_{op.id}"
                defs_parts.append(f'    <clipPath id="{clip_id}">')
                for op_id in op.operand_ids:
                    if op_id in prim_map:
                        defs_parts.append(cls._render_primitive_svg(prim_map[op_id]))
                defs_parts.append('    </clipPath>')

        if defs_parts:
            svg_parts.append('  <defs>')
            svg_parts.extend(defs_parts)
            svg_parts.append('  </defs>')

        # Exclude operand primitives that are pure subtractors/cutouts from standalone drawing
        subtractor_ids = set()
        for op in plan.operations:
            if op.operation in (OperationType.SUBTRACTION, OperationType.INTERSECTION):
                subtractor_ids.update(op.operand_ids)

        # Sort primitives by layer
        sorted_prims = sorted(plan.primitives, key=lambda p: p.layer)

        for prim in sorted_prims:
            if prim.id in subtractor_ids:
                continue

            mask_attr = ""
            clip_attr = ""
            if prim.id in op_target_map:
                op = op_target_map[prim.id]
                if op.operation == OperationType.SUBTRACTION:
                    mask_attr = f' mask="url(#mask_{op.id})"'
                elif op.operation == OperationType.INTERSECTION:
                    clip_attr = f' clip-path="url(#clip_{op.id})"'

            elem = cls._render_primitive_svg(prim, extra_attrs=f"{mask_attr}{clip_attr}")
            svg_parts.append(f"  {elem}")

        svg_parts.append('</svg>')
        return "\n".join(svg_parts)

    @classmethod
    def _render_primitive_svg(
        cls,
        prim: GeometricPrimitive,
        override_fill: Optional[str] = None,
        extra_attrs: str = "",
    ) -> str:
        fill = override_fill or prim.fill or "none"
        stroke = "none" if override_fill else (prim.stroke or "none")
        sw = prim.stroke_width
        opacity = prim.opacity

        style = f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"{extra_attrs}'

        ptype = prim.type
        if ptype == PrimitiveType.CIRCLE and prim.radius:
            p = prim.points[0]
            return f'<circle cx="{p.x:.2f}" cy="{p.y:.2f}" r="{prim.radius:.2f}" {style}/>'

        elif ptype == PrimitiveType.ELLIPSE and prim.radius:
            p = prim.points[0]
            rx = prim.radius
            ry = prim.radius_y or prim.radius
            return f'<ellipse cx="{p.x:.2f}" cy="{p.y:.2f}" rx="{rx:.2f}" ry="{ry:.2f}" {style}/>'

        elif ptype == PrimitiveType.LINE and len(prim.points) >= 2:
            p1, p2 = prim.points[0], prim.points[1]
            return f'<line x1="{p1.x:.2f}" y1="{p1.y:.2f}" x2="{p2.x:.2f}" y2="{p2.y:.2f}" {style}/>'

        elif ptype == PrimitiveType.POLYGON and prim.points:
            pts_str = " ".join(f"{p.x:.2f},{p.y:.2f}" for p in prim.points)
            return f'<polygon points="{pts_str}" {style}/>'

        elif ptype == PrimitiveType.BEZIER_QUAD and len(prim.points) >= 3:
            p0, p1, p2 = prim.points[0], prim.points[1], prim.points[2]
            d = f'M {p0.x:.2f} {p0.y:.2f} Q {p1.x:.2f} {p1.y:.2f} {p2.x:.2f} {p2.y:.2f}'
            if prim.closed:
                d += ' Z'
            return f'<path d="{d}" {style}/>'

        elif ptype == PrimitiveType.BEZIER_CUBIC and len(prim.points) >= 4:
            p0, p1, p2, p3 = prim.points[0], prim.points[1], prim.points[2], prim.points[3]
            d = f'M {p0.x:.2f} {p0.y:.2f} C {p1.x:.2f} {p1.y:.2f} {p2.x:.2f} {p2.y:.2f} {p3.x:.2f} {p3.y:.2f}'
            if prim.closed:
                d += ' Z'
            return f'<path d="{d}" {style}/>'

        elif ptype == PrimitiveType.ARC and prim.radius and prim.angles:
            p = prim.points[0]
            start_deg, end_deg = prim.angles
            # Approximate arc as SVG path or stroke
            cm = BezierMath
            return f'<circle cx="{p.x:.2f}" cy="{p.y:.2f}" r="{prim.radius:.2f}" {style}/>'

        return ""


class PNGRenderer:
    """Renders DrawingPlan to raster PNG format using Pillow."""

    @classmethod
    def render(cls, plan: DrawingPlan, supersample: int = 2) -> Image.Image:
        w, h = plan.width, plan.height
        canvas = Image.new("RGBA", (w, h), ImageColor.getrgb(plan.background_color) + (255,))

        prim_map = {p.id: p for p in plan.primitives}
        op_map = {op.target_id: op for op in plan.operations}

        # Subtractor operands are not drawn as independent positive shapes
        subtractor_ids = set()
        for op in plan.operations:
            if op.operation in (OperationType.SUBTRACTION, OperationType.INTERSECTION):
                subtractor_ids.update(op.operand_ids)

        sorted_prims = sorted(plan.primitives, key=lambda p: p.layer)

        for prim in sorted_prims:
            if prim.id in subtractor_ids:
                continue

            # Create base mask for primitive
            target_mask = BooleanEngine.create_primitive_mask(prim, w, h, supersample=supersample)

            # Apply boolean operations if target of an operation
            if prim.id in op_map:
                op = op_map[prim.id]
                op_masks = [
                    BooleanEngine.create_primitive_mask(prim_map[op_id], w, h, supersample=supersample)
                    for op_id in op.operand_ids
                    if op_id in prim_map
                ]
                final_mask = BooleanEngine.apply_operation(target_mask, op_masks, op.operation)
            else:
                final_mask = target_mask

            # Composite fill color
            fill_color = prim.fill or "#000000"
            rgb = ImageColor.getrgb(fill_color)
            alpha = int(prim.opacity * 255)
            color_layer = Image.new("RGBA", (w, h), rgb + (alpha,))

            # Paste onto canvas using final_mask as alpha
            canvas.paste(color_layer, (0, 0), final_mask)

            # Draw sharp strokes / borders if specified
            if prim.stroke and prim.stroke != "none" and prim.stroke_width > 0:
                cls._draw_stroke(canvas, prim, supersample=supersample)

        return canvas

    @classmethod
    def _draw_stroke(cls, canvas: Image.Image, prim: GeometricPrimitive, supersample: int = 2) -> None:
        """Draws crisp stroke outline on canvas."""
        draw = ImageDraw.Draw(canvas)
        stroke_color = ImageColor.getrgb(prim.stroke or "#000000") + (int(prim.opacity * 255),)
        sw = max(1, int(round(prim.stroke_width)))

        ptype = prim.type
        if ptype == PrimitiveType.CIRCLE and prim.radius:
            p = prim.points[0]
            r = prim.radius
            draw.ellipse((p.x - r, p.y - r, p.x + r, p.y + r), outline=stroke_color, width=sw)

        elif ptype == PrimitiveType.ELLIPSE and prim.radius:
            p = prim.points[0]
            rx = prim.radius
            ry = prim.radius_y or prim.radius
            draw.ellipse((p.x - rx, p.y - ry, p.x + rx, p.y + ry), outline=stroke_color, width=sw)

        elif ptype == PrimitiveType.LINE and len(prim.points) >= 2:
            p1, p2 = prim.points[0], prim.points[1]
            draw.line([(p1.x, p1.y), (p2.x, p2.y)], fill=stroke_color, width=sw)

        elif ptype == PrimitiveType.POLYGON and prim.points:
            coords = [(p.x, p.y) for p in prim.points]
            if len(coords) >= 3:
                draw.polygon(coords, outline=stroke_color, width=sw)

        elif ptype in (PrimitiveType.BEZIER_QUAD, PrimitiveType.BEZIER_CUBIC) and prim.points:
            is_cubic = ptype == PrimitiveType.BEZIER_CUBIC
            sampled = BezierMath.sample_curve(prim.points, is_cubic=is_cubic, steps=40)
            coords = [(p.x, p.y) for p in sampled]
            for i in range(len(coords) - 1):
                draw.line([coords[i], coords[i + 1]], fill=stroke_color, width=sw)

        elif ptype == PrimitiveType.ARC and prim.radius and prim.angles:
            p = prim.points[0]
            r = prim.radius
            start_deg, end_deg = prim.angles
            draw.arc((p.x - r, p.y - r, p.x + r, p.y + r), start=start_deg, end=end_deg, fill=stroke_color, width=sw)
