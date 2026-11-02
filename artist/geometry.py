"""
Geometry Engine for The Artist (Ability 0.01).

Implements mathematical calculations for circles, curves, strokes,
intersections, tangents, and coordinate transformations.
"""

import math
from typing import List, Tuple, Optional, Dict, Any
from .models import Point


class Vector2D:
    """2D Vector with arithmetic operations."""

    def __init__(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)

    def __add__(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vector2D":
        return Vector2D(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: float) -> "Vector2D":
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide vector by zero")
        return Vector2D(self.x / scalar, self.y / scalar)

    @property
    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def normalized(self) -> "Vector2D":
        l = self.length
        if l == 0:
            return Vector2D(0.0, 0.0)
        return Vector2D(self.x / l, self.y / l)

    def perpendicular(self) -> "Vector2D":
        """Returns a 90-degree counterclockwise perpendicular vector."""
        return Vector2D(-self.y, self.x)

    def dot(self, other: "Vector2D") -> float:
        return self.x * other.x + self.y * other.y

    def to_point(self) -> Point:
        return Point(x=self.x, y=self.y)

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


class CircleMath:
    """Mathematical properties and spatial relations for circles."""

    def __init__(self, center: Point, radius: float):
        if radius <= 0:
            raise ValueError(f"Radius must be strictly positive, got {radius}")
        self.center = Point(x=float(center.x), y=float(center.y))
        self.radius = float(radius)

    @property
    def diameter(self) -> float:
        return self.radius * 2.0

    @property
    def circumference(self) -> float:
        return 2.0 * math.pi * self.radius

    @property
    def area(self) -> float:
        return math.pi * (self.radius ** 2)

    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        """Returns (x_min, y_min, x_max, y_max)."""
        return (
            self.center.x - self.radius,
            self.center.y - self.radius,
            self.center.x + self.radius,
            self.center.y + self.radius,
        )

    def distance_to(self, other: "CircleMath") -> float:
        return math.hypot(self.center.x - other.center.x, self.center.y - other.center.y)

    def relationship_with(self, other: "CircleMath") -> Dict[str, Any]:
        """
        Calculates relationship with a neighbouring circle:
        overlap distance, relation type ('disjoint', 'tangent_external',
        'intersecting', 'tangent_internal', 'contained', 'concentric').
        """
        d = self.distance_to(other)
        r1, r2 = self.radius, other.radius
        overlap = (r1 + r2) - d

        if d < 1e-9:
            relation = "concentric"
        elif abs(d - (r1 + r2)) < 1e-4:
            relation = "tangent_external"
        elif abs(d - abs(r1 - r2)) < 1e-4:
            relation = "tangent_internal"
        elif d > (r1 + r2):
            relation = "disjoint"
        elif d < abs(r1 - r2):
            relation = "contained"
        else:
            relation = "intersecting"

        return {
            "distance": d,
            "overlap": max(0.0, overlap),
            "relation": relation,
            "center_offset": (other.center.x - self.center.x, other.center.y - self.center.y),
        }

    def intersection_points(self, other: "CircleMath") -> List[Point]:
        """Calculates exact 2D intersection points between two circles."""
        x0, y0 = self.center.x, self.center.y
        x1, y1 = other.center.x, other.center.y
        r0, r1 = self.radius, other.radius

        d = math.hypot(x1 - x0, y1 - y0)

        # No intersection: separate or one inside the other or concentric
        if d > (r0 + r1) or d < abs(r0 - r1) or d == 0:
            return []

        # Distance from center0 to chord line
        a = (r0 ** 2 - r1 ** 2 + d ** 2) / (2.0 * d)
        h_squared = r0 ** 2 - a ** 2
        h = math.sqrt(max(0.0, h_squared))

        # Base point on line between centers
        x2 = x0 + a * (x1 - x0) / d
        y2 = y0 + a * (y1 - y0) / d

        if h < 1e-9:
            # Single tangent intersection point
            return [Point(x=x2, y=y2)]

        # Two intersection points
        rx = -(y1 - y0) * (h / d)
        ry = (x1 - x0) * (h / d)

        return [
            Point(x=x2 + rx, y=y2 + ry),
            Point(x=x2 - rx, y=y2 - ry),
        ]

    def point_at_angle(self, angle_deg: float) -> Point:
        rad = math.radians(angle_deg)
        return Point(
            x=self.center.x + self.radius * math.cos(rad),
            y=self.center.y + self.radius * math.sin(rad),
        )


class StrokeMath:
    """Calculations for straight lines, sharp strokes, and outlines."""

    @staticmethod
    def length(p1: Point, p2: Point) -> float:
        return math.hypot(p2.x - p1.x, p2.y - p1.y)

    @staticmethod
    def angle_deg(p1: Point, p2: Point) -> float:
        return math.degrees(math.atan2(p2.y - p1.y, p2.x - p1.x))

    @staticmethod
    def line_intersection(p1: Point, p2: Point, p3: Point, p4: Point) -> Optional[Point]:
        """Computes intersection of two line segments (p1-p2) and (p3-p4)."""
        det = (p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x)
        if abs(det) < 1e-9:
            return None  # Parallel or collinear

        t = ((p1.x - p3.x) * (p3.y - p4.y) - (p1.y - p3.y) * (p3.x - p4.x)) / det
        u = -((p1.x - p2.x) * (p1.y - p3.y) - (p1.y - p2.y) * (p1.x - p3.x)) / det

        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
            return Point(
                x=p1.x + t * (p2.x - p1.x),
                y=p1.y + t * (p2.y - p1.y),
            )
        return None

    @staticmethod
    def offset_line(p1: Point, p2: Point, offset: float) -> Tuple[Point, Point]:
        """Creates a parallel line offset by `offset` distance."""
        vec = Vector2D(p2.x - p1.x, p2.y - p1.y).normalized()
        perp = vec.perpendicular() * offset
        return (
            Point(x=p1.x + perp.x, y=p1.y + perp.y),
            Point(x=p2.x + perp.x, y=p2.y + perp.y),
        )


class BezierMath:
    """Calculations for Quadratic and Cubic Bezier curves."""

    @staticmethod
    def eval_quadratic(p0: Point, p1: Point, p2: Point, t: float) -> Point:
        """Quadratic Bezier: B(t) = (1-t)^2 P0 + 2(1-t)t P1 + t^2 P2."""
        t = max(0.0, min(1.0, t))
        inv_t = 1.0 - t
        x = (inv_t ** 2) * p0.x + 2.0 * inv_t * t * p1.x + (t ** 2) * p2.x
        y = (inv_t ** 2) * p0.y + 2.0 * inv_t * t * p1.y + (t ** 2) * p2.y
        return Point(x=x, y=y)

    @staticmethod
    def eval_cubic(p0: Point, p1: Point, p2: Point, p3: Point, t: float) -> Point:
        """Cubic Bezier: B(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t)t^2 P2 + t^3 P3."""
        t = max(0.0, min(1.0, t))
        inv_t = 1.0 - t
        x = (
            (inv_t ** 3) * p0.x
            + 3.0 * (inv_t ** 2) * t * p1.x
            + 3.0 * inv_t * (t ** 2) * p2.x
            + (t ** 3) * p3.x
        )
        y = (
            (inv_t ** 3) * p0.y
            + 3.0 * (inv_t ** 2) * t * p1.y
            + 3.0 * inv_t * (t ** 2) * p2.y
            + (t ** 3) * p3.y
        )
        return Point(x=x, y=y)

    @classmethod
    def sample_curve(cls, points: List[Point], is_cubic: bool = True, steps: int = 30) -> List[Point]:
        """Discretizes a Bezier curve into points for rasterization or analysis."""
        sampled = []
        for i in range(steps + 1):
            t = i / float(steps)
            if is_cubic and len(points) >= 4:
                pt = cls.eval_cubic(points[0], points[1], points[2], points[3], t)
            elif len(points) >= 3:
                pt = cls.eval_quadratic(points[0], points[1], points[2], t)
            else:
                pt = points[0]
            sampled.append(pt)
        return sampled


class PolygonMath:
    """Calculations for Polygons."""

    @staticmethod
    def centroid(points: List[Point]) -> Point:
        if not points:
            return Point(x=0.0, y=0.0)
        cx = sum(p.x for p in points) / len(points)
        cy = sum(p.y for p in points) / len(points)
        return Point(x=cx, y=cy)

    @staticmethod
    def bounding_box(points: List[Point]) -> Tuple[float, float, float, float]:
        if not points:
            return (0.0, 0.0, 0.0, 0.0)
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        return (min(xs), min(ys), max(xs), max(ys))

    @staticmethod
    def area(points: List[Point]) -> float:
        """Shoelace formula."""
        n = len(points)
        if n < 3:
            return 0.0
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += points[i].x * points[j].y
            area -= points[j].x * points[i].y
        return abs(area) / 2.0


class TransformMath:
    """Affine 2D transformations: translation, scaling, rotation."""

    @staticmethod
    def translate_point(p: Point, dx: float, dy: float) -> Point:
        return Point(x=p.x + dx, y=p.y + dy)

    @staticmethod
    def scale_point(p: Point, origin: Point, sx: float, sy: float) -> Point:
        return Point(
            x=origin.x + (p.x - origin.x) * sx,
            y=origin.y + (p.y - origin.y) * sy,
        )

    @staticmethod
    def rotate_point(p: Point, origin: Point, angle_deg: float) -> Point:
        rad = math.radians(angle_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        dx = p.x - origin.x
        dy = p.y - origin.y
        return Point(
            x=origin.x + dx * cos_a - dy * sin_a,
            y=origin.y + dx * sin_a + dy * cos_a,
        )
