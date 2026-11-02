"""
Canvas and Grid System for The Artist (Ability 0.01).

Provides coordinate mapping, cell subdivisions, spatial partitioning,
and precision alignment.
"""

import math
from typing import List, Tuple, Optional, Set
from .models import Point
from .geometry import CircleMath


class GridCell:
    """A discrete coordinate cell on the canvas."""

    def __init__(self, col: int, row: int, x_min: float, y_min: float, x_max: float, y_max: float):
        self.col = col
        self.row = row
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

    @property
    def center(self) -> Point:
        return Point(
            x=(self.x_min + self.x_max) / 2.0,
            y=(self.y_min + self.y_max) / 2.0,
        )

    def contains_point(self, p: Point) -> bool:
        return self.x_min <= p.x <= self.x_max and self.y_min <= p.y <= self.y_max

    def intersects_box(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        return not (self.x_max < x0 or self.x_min > x1 or self.y_max < y0 or self.y_min > y1)


class CanvasGrid:
    """
    Mathematically organized canvas grid.
    Divides canvas into cells for spatial indexing and coordinate alignment.
    """

    def __init__(self, width: float, height: float, resolution: float = 20.0):
        if width <= 0 or height <= 0:
            raise ValueError(f"Canvas dimensions must be positive, got ({width}, {height})")
        if resolution <= 0:
            raise ValueError(f"Grid resolution must be positive, got {resolution}")

        self.width = float(width)
        self.height = float(height)
        self.resolution = float(resolution)

        self.cols = max(1, int(math.ceil(self.width / self.resolution)))
        self.rows = max(1, int(math.ceil(self.height / self.resolution)))

    def coord_to_cell_indices(self, x: float, y: float) -> Tuple[int, int]:
        """Maps continuous canvas coordinates to grid column and row indices."""
        col = max(0, min(self.cols - 1, int(x // self.resolution)))
        row = max(0, min(self.rows - 1, int(y // self.resolution)))
        return (col, row)

    def get_cell(self, col: int, row: int) -> GridCell:
        """Returns cell bounds for specific column and row."""
        c = max(0, min(self.cols - 1, col))
        r = max(0, min(self.rows - 1, row))
        x_min = c * self.resolution
        y_min = r * self.resolution
        x_max = min(self.width, x_min + self.resolution)
        y_max = min(self.height, y_min + self.resolution)
        return GridCell(c, r, x_min, y_min, x_max, y_max)

    def snap_point(self, p: Point, snap_step: Optional[float] = None) -> Point:
        """Snaps a point to the nearest grid intersection."""
        step = snap_step or self.resolution
        return Point(
            x=round(p.x / step) * step,
            y=round(p.y / step) * step,
        )

    def is_within_bounds(self, p: Point) -> bool:
        return 0.0 <= p.x <= self.width and 0.0 <= p.y <= self.height

    def clamp_point(self, p: Point) -> Point:
        return Point(
            x=max(0.0, min(self.width, p.x)),
            y=max(0.0, min(self.height, p.y)),
        )

    # ============================================================
    # SPATIAL PARTITIONING & RELEVANT CELLS
    # ============================================================

    def cells_for_bounding_box(self, x_min: float, y_min: float, x_max: float, y_max: float) -> List[GridCell]:
        """Identifies only the grid cells relevant to a bounding box."""
        col_start, row_start = self.coord_to_cell_indices(max(0.0, x_min), max(0.0, y_min))
        col_end, row_end = self.coord_to_cell_indices(min(self.width, x_max), min(self.height, y_max))

        cells = []
        for r in range(row_start, row_end + 1):
            for c in range(col_start, col_end + 1):
                cells.append(self.get_cell(c, r))
        return cells

    def cells_for_circle(self, center: Point, radius: float) -> List[GridCell]:
        """Identifies cells intersected by a circle."""
        cm = CircleMath(center, radius)
        x0, y0, x1, y1 = cm.bounding_box
        candidate_cells = self.cells_for_bounding_box(x0, y0, x1, y1)

        # Precise cell test: does cell intersect circle?
        relevant = []
        for cell in candidate_cells:
            # Find closest point on cell to circle center
            closest_x = max(cell.x_min, min(center.x, cell.x_max))
            closest_y = max(cell.y_min, min(center.y, cell.y_max))
            dist_sq = (center.x - closest_x) ** 2 + (center.y - closest_y) ** 2
            if dist_sq <= (radius ** 2):
                relevant.append(cell)
        return relevant

    def cells_for_stroke(self, p1: Point, p2: Point) -> List[GridCell]:
        """Identifies cells intersected by a line stroke using Bresenham / grid walk."""
        x0, y0, x1, y1 = (
            min(p1.x, p2.x),
            min(p1.y, p2.y),
            max(p1.x, p2.x),
            max(p1.y, p2.y),
        )
        return self.cells_for_bounding_box(x0, y0, x1, y1)
