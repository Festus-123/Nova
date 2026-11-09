"""
G-Stroke: the drawing algorithm of Nebula's Artist (Ability 0.01).

  G-Stroke 1.0  geometry: circles, shapes and strokes joined by union / intersection / subtraction (the original,
                and still the default). Lives in the rest of the `artist` package and is not changed by 2.0.
  G-Stroke 2.0  sketching: brushes, points, strokes and strands. Pen paths with pressure and taper, hatching and
                cross-hatching for tone, optional colour; clean sketches of art rather than plain geometry.

Callers choose with `algorithm` ("1.0" or "2.0"). Nothing that does not ask for 2.0 ever reaches it.
"""

from ..errors import GeometricValidationError

DEFAULT_ALGORITHM = "1.0"
ALGORITHMS = {
    "1.0": "Geometry: circles, shapes and strokes with union / intersection / subtraction.",
    "2.0": "Sketching: brushes, points, strokes and strands; hatching, pressure, taper and colour.",
}

_ALIASES = {"1": "1.0", "1.0": "1.0", "v1": "1.0", "g-stroke-1.0": "1.0", "gstroke-1.0": "1.0", "gstroke1": "1.0",
            "2": "2.0", "2.0": "2.0", "v2": "2.0", "g-stroke-2.0": "2.0", "gstroke-2.0": "2.0", "gstroke2": "2.0"}


def resolve_algorithm(value) -> str:
    """'2', '2.0', 'v2', 'g-stroke-2.0' -> '2.0'. Empty / None -> the default, '1.0'. Anything else is a 422."""
    if value is None or str(value).strip() == "":
        return DEFAULT_ALGORITHM
    key = str(value).strip().lower().replace(" ", "-").replace("_", "-")
    if key not in _ALIASES:
        raise GeometricValidationError(f"Unknown drawing algorithm '{value}'. Available: {', '.join(ALGORITHMS)}.")
    return _ALIASES[key]


from .brushes import BRUSHES, Brush, brush_names, get_brush  # noqa: E402
from .sketch import Part, Sketch, Stroke, make_strand  # noqa: E402
from .render import SketchPNG, SketchSVG  # noqa: E402
from .imaging import ImageSketcher, v2_style  # noqa: E402
from .planner import SketchPlanner, SketchProgramError, program_to_sketch  # noqa: E402

__all__ = [
    "DEFAULT_ALGORITHM", "ALGORITHMS", "resolve_algorithm",
    "BRUSHES", "Brush", "brush_names", "get_brush",
    "Part", "Sketch", "Stroke", "make_strand",
    "SketchPNG", "SketchSVG", "ImageSketcher", "v2_style",
    "SketchPlanner", "SketchProgramError", "program_to_sketch",
]
