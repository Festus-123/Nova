"""
Brushes for G-Stroke 2.0.

A brush decides how one stroke (a path of points with pressure) becomes ink on paper:
  - how the width swells and tapers along the stroke (pressure, entry / exit taper),
  - how much the "hand" wobbles and how uneven the line is,
  - how many overlapping passes ("layers") make the mark, which is what gives pencil and charcoal their grain.

Everything is deterministic: the same stroke with the same seed always gives the same mark.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Brush:
    name: str
    description: str
    width_scale: float = 1.0   # multiplies the stroke's nominal width
    taper_px: float = 8.0      # length over which the stroke grows from / shrinks to its minimum width
    min_width: float = 0.3     # width at the very tip, as a fraction of full width
    pressure_gain: float = 0.4  # 0 = ignore pressure, 1 = width follows pressure fully
    width_noise: float = 0.05  # unevenness of the line (fraction of width)
    wobble: float = 0.3        # sideways drift of the hand, in pixels
    opacity: float = 1.0
    # (offset in widths, width multiplier, opacity multiplier): one entry per pass of the brush
    layers: Tuple[Tuple[float, float, float], ...] = ((0.0, 1.0, 1.0),)


BRUSHES: Dict[str, Brush] = {
    "pen": Brush("pen", "Clean, even ink line with a light taper.", 1.0, 6, 0.35, 0.35, 0.04, 0.25, 1.0),
    "fineliner": Brush("fineliner", "Thin, precise technical line.", 0.7, 3, 0.6, 0.1, 0.0, 0.1, 1.0),
    "ink": Brush("ink", "Brush pen: strong taper, the line swells with pressure.", 1.15, 20, 0.06, 0.85, 0.07, 0.3, 0.96),
    "pencil": Brush(
        "pencil", "Graphite: several soft, slightly offset passes with grain.", 1.0, 12, 0.18, 0.6, 0.22, 0.55, 0.85,
        ((-0.30, 0.70, 0.70), (0.28, 0.62, 0.62), (0.0, 0.90, 0.50)),
    ),
    "charcoal": Brush(
        "charcoal", "Wide, smudgy, broken line built from four loose passes.", 2.2, 16, 0.2, 0.7, 0.45, 0.9, 0.62,
        ((-0.40, 0.55, 0.55), (0.35, 0.60, 0.50), (-0.05, 0.85, 0.40), (0.10, 0.45, 0.60)),
    ),
    "marker": Brush("marker", "Flat, bold, even mark with blunt ends.", 2.4, 0, 1.0, 0.0, 0.02, 0.2, 0.88),
}

DEFAULT_BRUSH = "pencil"


def get_brush(name: str | None, fallback: str = DEFAULT_BRUSH) -> Brush:
    key = (name or "").strip().lower()
    aliases = {"graphite": "pencil", "pen": "pen", "biro": "pen", "brush": "ink", "calligraphy": "ink", "felt": "marker", "liner": "fineliner"}
    key = aliases.get(key, key)
    return BRUSHES.get(key) or BRUSHES[fallback if fallback in BRUSHES else DEFAULT_BRUSH]


def brush_names() -> List[str]:
    return list(BRUSHES)
