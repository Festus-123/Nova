"""
Boolean Engine for The Artist (Ability 0.01).

Implements UNION, INTERSECTION, and SUBTRACTION geometric operations.
Provides mask composition for raster rendering and vector clip/mask
specifications for SVG rendering.
"""

from typing import List, Tuple, Dict, Any, Optional
from PIL import Image, ImageChops, ImageDraw
from .models import GeometricPrimitive, OperationType, PrimitiveType, Point
from .geometry import BezierMath


class BooleanEngine:
    """
    Coordinates geometric UNION, INTERSECTION, and SUBTRACTION operations.
    """

    @staticmethod
    def create_primitive_mask(
        primitive: GeometricPrimitive,
        width: int,
        height: int,
        supersample: int = 2,
    ) -> Image.Image:
        """
        Creates an 8-bit alpha mask ('L') for a geometric primitive.
        Uses supersampling for anti-aliasing.
        """
        w, h = width * supersample, height * supersample
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)

        # Helper to scale points
        def s_pt(p: Point) -> Tuple[float, float]:
            return (p.x * supersample, p.y * supersample)

        ptype = primitive.type

        if ptype == PrimitiveType.CIRCLE and primitive.radius:
            cx, cy = s_pt(primitive.points[0])
            r = primitive.radius * supersample
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)

        elif ptype == PrimitiveType.ELLIPSE and primitive.radius:
            cx, cy = s_pt(primitive.points[0])
            rx = primitive.radius * supersample
            ry = (primitive.radius_y or primitive.radius) * supersample
            draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=255)

        elif ptype == PrimitiveType.POLYGON and primitive.points:
            coords = [s_pt(p) for p in primitive.points]
            if len(coords) >= 3:
                draw.polygon(coords, fill=255)

        elif ptype in (PrimitiveType.BEZIER_QUAD, PrimitiveType.BEZIER_CUBIC) and primitive.points:
            is_cubic = ptype == PrimitiveType.BEZIER_CUBIC
            sampled = BezierMath.sample_curve(primitive.points, is_cubic=is_cubic, steps=40)
            coords = [s_pt(p) for p in sampled]
            if primitive.closed and len(coords) >= 3:
                draw.polygon(coords, fill=255)
            else:
                sw = max(1, int(primitive.stroke_width * supersample))
                for i in range(len(coords) - 1):
                    draw.line([coords[i], coords[i + 1]], fill=255, width=sw)

        elif ptype == PrimitiveType.LINE and len(primitive.points) >= 2:
            p1 = s_pt(primitive.points[0])
            p2 = s_pt(primitive.points[1])
            sw = max(1, int(primitive.stroke_width * supersample))
            draw.line([p1, p2], fill=255, width=sw)

        elif ptype == PrimitiveType.ARC and primitive.radius and primitive.angles:
            cx, cy = s_pt(primitive.points[0])
            r = primitive.radius * supersample
            start_deg, end_deg = primitive.angles
            sw = max(1, int(primitive.stroke_width * supersample))
            draw.arc((cx - r, cy - r, cx + r, cy + r), start=start_deg, end=end_deg, fill=255, width=sw)

        if supersample > 1:
            mask = mask.resize((width, height), Image.Resampling.LANCZOS)

        return mask

    @staticmethod
    def apply_operation(
        target_mask: Image.Image,
        operand_masks: List[Image.Image],
        operation: OperationType,
    ) -> Image.Image:
        """
        Applies UNION, INTERSECTION, or SUBTRACTION on alpha masks.
        """
        result = target_mask.copy()

        for op_mask in operand_masks:
            if operation == OperationType.UNION:
                # Combine shapes: lighter (max)
                result = ImageChops.lighter(result, op_mask)
            elif operation == OperationType.INTERSECTION:
                # Overlapping region only: darker (min)
                result = ImageChops.darker(result, op_mask)
            elif operation == OperationType.SUBTRACTION:
                # Cut out operand shape from target: subtract
                result = ImageChops.subtract(result, op_mask)

        return result
