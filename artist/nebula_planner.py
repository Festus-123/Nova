"""
Nebula Intelligence Drawing Planner for The Artist (Ability 0.01).

Interprets natural-language drawing instructions, determines dimensions
and proportions, decomposes objects into geometric components, coordinates
circles, strokes, curves, and boolean operations.
"""

import json
import logging
import math
import os
import re
from typing import Optional, Dict, Any, List

from .models import (
    DrawingPlan,
    GeometricPrimitive,
    GeometricOperation,
    Point,
    PrimitiveType,
    OperationType,
)
from .geometry import CircleMath, BezierMath, PolygonMath, TransformMath
from .grid import CanvasGrid
from .patterns import PatternEngine
from .db import ArtistDB

logger = logging.getLogger("nebula.artist.planner")


class NebulaPlanner:
    """Nebula intelligence responsible for geometric drawing planning."""

    def __init__(self, db: Optional[ArtistDB] = None):
        self.db = db or ArtistDB()
        self.pattern_engine = PatternEngine(self.db)

    def plan_drawing(
        self,
        prompt: str,
        width: int = 800,
        height: int = 600,
        grid_resolution: float = 20.0,
        custom_params: Optional[Dict[str, Any]] = None,
    ) -> DrawingPlan:
        """
        Interprets prompt and constructs a structured DrawingPlan.
        """
        grid = CanvasGrid(width=width, height=height, resolution=grid_resolution)
        lower_prompt = prompt.lower().strip()

        # Check if LLM planning should be attempted for novel/complex prompts
        if self._should_use_llm(lower_prompt):
            try:
                llm_plan = self._plan_with_llm(prompt, width, height, grid)
                if llm_plan and len(llm_plan.primitives) > 0:
                    return llm_plan
            except Exception as e:
                logger.warning("LLM planning fallback to algorithmic geometric engine: %s", e)

        # Algorithmic geometric synthesis
        if any(w in lower_prompt for w in ["ferrari", "sports car", "supercar", "car", "vehicle", "automobile"]):
            return self._plan_ferrari_or_car(prompt, width, height, grid, custom_params)

        elif any(w in lower_prompt for w in ["clock", "watch", "time", "dial"]):
            return self._plan_clock(prompt, width, height, grid, custom_params)

        elif any(w in lower_prompt for w in ["gear", "cog", "mechanical"]):
            return self._plan_gear(prompt, width, height, grid, custom_params)

        elif any(w in lower_prompt for w in ["flower", "mandala", "petal", "rose"]):
            return self._plan_geometric_flower(prompt, width, height, grid, custom_params)

        elif any(w in lower_prompt for w in ["sun", "solar", "moon", "planet", "celestial", "space"]):
            return self._plan_celestial(prompt, width, height, grid, custom_params)

        else:
            return self._plan_geometric_composition(prompt, width, height, grid, custom_params)

    # ============================================================
    # FERRARI / SPORTS CAR GEOMETRIC SYNTHESIS
    # ============================================================

    def _plan_ferrari_or_car(
        self,
        prompt: str,
        width: int,
        height: int,
        grid: CanvasGrid,
        params: Optional[Dict[str, Any]] = None,
    ) -> DrawingPlan:
        """
        Mathematically constructs a Ferrari / sports car drawing:
        - Sleek aerodynamic chassis with Bezier body contours
        - Wheel arches cut out via SUBTRACTION operations
        - Front and rear multi-layer geometric wheels (tire, alloy rim, hub)
        - Cockpit windows and aerodynamic spoiler
        - Dynamic Ferrari red (#E60000) styling and prancing shield badge
        """
        params = params or {}
        body_color = params.get("body_color", "#D40000")  # Ferrari Rosso Corsa
        accent_color = params.get("accent_color", "#111111")
        glass_color = params.get("glass_color", "#1E293B")

        # Proportions based on canvas size
        car_w = min(width * 0.85, 700.0)
        car_h = car_w * 0.28
        car_x = (width - car_w) / 2.0
        car_y = height * 0.52

        wheel_r = car_h * 0.38
        wheel_arch_r = wheel_r * 1.18

        front_wheel_cx = car_x + car_w * 0.22
        rear_wheel_cx = car_x + car_w * 0.78
        ground_y = car_y + car_h
        wheel_cy = ground_y - wheel_r * 0.65

        primitives: List[GeometricPrimitive] = []
        operations: List[GeometricOperation] = []

        # 1. Ground shadow / road baseline stroke
        shadow_p1 = Point(x=car_x - 30.0, y=ground_y + 4.0)
        shadow_p2 = Point(x=car_x + car_w + 30.0, y=ground_y + 4.0)
        primitives.append(
            GeometricPrimitive(
                id="road_shadow",
                type=PrimitiveType.LINE,
                points=[shadow_p1, shadow_p2],
                stroke="#6B7280",
                stroke_width=3.0,
                layer=0,
                label="Road baseline",
            )
        )

        # 2. Main aerodynamic body chassis (Bezier contours + sharp edges)
        nose_pt = Point(x=car_x, y=ground_y - car_h * 0.3)
        hood_front = Point(x=car_x + car_w * 0.08, y=ground_y - car_h * 0.55)
        hood_back = Point(x=car_x + car_w * 0.30, y=ground_y - car_h * 0.62)
        windshield_top = Point(x=car_x + car_w * 0.45, y=ground_y - car_h * 1.05)
        roof_back = Point(x=car_x + car_w * 0.65, y=ground_y - car_h * 1.0)
        engine_deck = Point(x=car_x + car_w * 0.82, y=ground_y - car_h * 0.65)
        spoiler_top = Point(x=car_x + car_w * 0.98, y=ground_y - car_h * 0.72)
        tail_bottom = Point(x=car_x + car_w * 0.96, y=ground_y - car_h * 0.25)
        diffuser_pt = Point(x=car_x + car_w * 0.88, y=ground_y - car_h * 0.15)
        front_splitter = Point(x=car_x + car_w * 0.05, y=ground_y - car_h * 0.15)

        body_poly = [
            nose_pt,
            hood_front,
            hood_back,
            windshield_top,
            roof_back,
            engine_deck,
            spoiler_top,
            tail_bottom,
            diffuser_pt,
            Point(x=rear_wheel_cx + wheel_arch_r, y=ground_y - car_h * 0.15),
            Point(x=rear_wheel_cx - wheel_arch_r, y=ground_y - car_h * 0.15),
            Point(x=front_wheel_cx + wheel_arch_r, y=ground_y - car_h * 0.15),
            Point(x=front_wheel_cx - wheel_arch_r, y=ground_y - car_h * 0.15),
            front_splitter,
        ]

        primitives.append(
            GeometricPrimitive(
                id="ferrari_chassis",
                type=PrimitiveType.POLYGON,
                points=body_poly,
                fill=body_color,
                stroke="#8B0000",
                stroke_width=2.5,
                layer=2,
                label="Ferrari aerodynamic chassis",
            )
        )

        # 3. Aerodynamic roofline contour curve (Cubic Bezier)
        primitives.append(
            GeometricPrimitive(
                id="roof_curve",
                type=PrimitiveType.BEZIER_CUBIC,
                points=[
                    hood_back,
                    Point(x=car_x + car_w * 0.38, y=ground_y - car_h * 0.95),
                    windshield_top,
                    roof_back,
                ],
                stroke="#660000",
                stroke_width=2.0,
                closed=False,
                layer=3,
                label="Curved roofline",
            )
        )

        # 4. Cockpit window polygon
        cockpit_pts = [
            Point(x=car_x + car_w * 0.33, y=ground_y - car_h * 0.65),
            Point(x=car_x + car_w * 0.46, y=ground_y - car_h * 0.98),
            Point(x=car_x + car_w * 0.63, y=ground_y - car_h * 0.94),
            Point(x=car_x + car_w * 0.68, y=ground_y - car_h * 0.66),
        ]
        primitives.append(
            GeometricPrimitive(
                id="cockpit_glass",
                type=PrimitiveType.POLYGON,
                points=cockpit_pts,
                fill=glass_color,
                stroke="#0F172A",
                stroke_width=2.0,
                layer=4,
                label="Tinted glass",
            )
        )

        # 5. Wheel Arch Cutouts (SUBTRACTION from chassis)
        f_arch = GeometricPrimitive(
            id="front_arch_cutout",
            type=PrimitiveType.CIRCLE,
            points=[Point(x=front_wheel_cx, y=wheel_cy)],
            radius=wheel_arch_r,
            layer=1,
            label="Front wheel arch cutout",
        )
        r_arch = GeometricPrimitive(
            id="rear_arch_cutout",
            type=PrimitiveType.CIRCLE,
            points=[Point(x=rear_wheel_cx, y=wheel_cy)],
            radius=wheel_arch_r,
            layer=1,
            label="Rear wheel arch cutout",
        )
        primitives.extend([f_arch, r_arch])

        operations.append(
            GeometricOperation(
                id="op_chassis_wheel_cutouts",
                operation=OperationType.SUBTRACTION,
                target_id="ferrari_chassis",
                operand_ids=["front_arch_cutout", "rear_arch_cutout"],
                label="Wheel wells subtracted from chassis",
            )
        )

        # 6. Instantiate Wheels using reusable pattern memory from Artist DB
        f_wheel_prims, f_wheel_ops = self.pattern_engine.instantiate_pattern(
            pattern_name_or_id="wheel",
            instance_id="f_wheel",
            params={"center_x": front_wheel_cx, "center_y": wheel_cy, "radius": wheel_r},
        )
        r_wheel_prims, r_wheel_ops = self.pattern_engine.instantiate_pattern(
            pattern_name_or_id="wheel",
            instance_id="r_wheel",
            params={"center_x": rear_wheel_cx, "center_y": wheel_cy, "radius": wheel_r},
        )

        for p in f_wheel_prims + r_wheel_prims:
            p.layer = 5  # Wheels sit in front of wheel well cutouts
        primitives.extend(f_wheel_prims)
        primitives.extend(r_wheel_prims)
        operations.extend(f_wheel_ops)
        operations.extend(r_wheel_ops)

        # 7. Sleek side air intake duct (SUBTRACTION / sharp contour)
        intake_pts = [
            Point(x=car_x + car_w * 0.52, y=ground_y - car_h * 0.45),
            Point(x=car_x + car_w * 0.65, y=ground_y - car_h * 0.50),
            Point(x=car_x + car_w * 0.68, y=ground_y - car_h * 0.35),
            Point(x=car_x + car_w * 0.54, y=ground_y - car_h * 0.32),
        ]
        primitives.append(
            GeometricPrimitive(
                id="side_air_intake",
                type=PrimitiveType.POLYGON,
                points=intake_pts,
                fill="#1A1A1A",
                stroke="#111111",
                stroke_width=1.5,
                layer=6,
                label="Aero side intake",
            )
        )

        # 8. Front Headlight Sharp Contour
        headlight_pts = [
            Point(x=car_x + car_w * 0.06, y=ground_y - car_h * 0.46),
            Point(x=car_x + car_w * 0.15, y=ground_y - car_h * 0.52),
            Point(x=car_x + car_w * 0.12, y=ground_y - car_h * 0.42),
        ]
        primitives.append(
            GeometricPrimitive(
                id="headlight",
                type=PrimitiveType.POLYGON,
                points=headlight_pts,
                fill="#FEF08A",  # Xenon / LED glow
                stroke="#CA8A04",
                stroke_width=1.5,
                layer=6,
                label="LED headlight",
            )
        )

        # 9. Scuderia Ferrari Racing Shield Badge
        shield_cx = car_x + car_w * 0.28
        shield_cy = ground_y - car_h * 0.50
        primitives.append(
            GeometricPrimitive(
                id="ferrari_shield_badge",
                type=PrimitiveType.CIRCLE,
                points=[Point(x=shield_cx, y=shield_cy)],
                radius=car_h * 0.08,
                fill="#FACC15",  # Ferrari Modena Yellow
                stroke="#1E293B",
                stroke_width=1.2,
                layer=7,
                label="Modena yellow shield badge",
            )
        )

        return DrawingPlan(
            prompt=prompt,
            width=width,
            height=height,
            background_color="#F8FAFC",
            primitives=primitives,
            operations=operations,
            metadata={
                "theme": "ferrari_sports_car",
                "car_width": car_w,
                "car_height": car_h,
                "wheelbase": rear_wheel_cx - front_wheel_cx,
                "wheel_radius": wheel_r,
            },
        )

    # ============================================================
    # CLOCK / MECHANICAL TIMEPIECE SYNTHESIS
    # ============================================================

    def _plan_clock(
        self,
        prompt: str,
        width: int,
        height: int,
        grid: CanvasGrid,
        params: Optional[Dict[str, Any]] = None,
    ) -> DrawingPlan:
        """Constructs geometric clock with concentric circles, tick marks, and gears."""
        cx = width / 2.0
        cy = height / 2.0
        r_outer = min(width, height) * 0.38
        r_inner = r_outer * 0.88

        primitives: List[GeometricPrimitive] = []
        operations: List[GeometricOperation] = []

        # Outer bezel
        primitives.append(
            GeometricPrimitive(
                id="clock_bezel",
                type=PrimitiveType.CIRCLE,
                points=[Point(x=cx, y=cy)],
                radius=r_outer,
                fill="#334155",
                stroke="#1E293B",
                stroke_width=3.0,
                layer=1,
            )
        )

        # Dial face
        primitives.append(
            GeometricPrimitive(
                id="clock_face",
                type=PrimitiveType.CIRCLE,
                points=[Point(x=cx, y=cy)],
                radius=r_inner,
                fill="#F8FAFC",
                stroke="#0F172A",
                stroke_width=2.0,
                layer=2,
            )
        )

        # Hour tick marks (12 ticks around circumference)
        clock_cm = CircleMath(Point(x=cx, y=cy), r_inner)
        for hour in range(12):
            angle = hour * 30.0
            p_outer = clock_cm.point_at_angle(angle)
            tick_r = r_inner * (0.82 if hour % 3 == 0 else 0.88)
            p_inner = CircleMath(Point(x=cx, y=cy), tick_r).point_at_angle(angle)
            primitives.append(
                GeometricPrimitive(
                    id=f"tick_{hour}",
                    type=PrimitiveType.LINE,
                    points=[p_inner, p_outer],
                    stroke="#0F172A",
                    stroke_width=3.5 if hour % 3 == 0 else 1.5,
                    layer=3,
                )
            )

        # Hour and Minute hands
        hour_hand_end = CircleMath(Point(x=cx, y=cy), r_inner * 0.5).point_at_angle(-60.0)
        min_hand_end = CircleMath(Point(x=cx, y=cy), r_inner * 0.75).point_at_angle(30.0)

        primitives.append(
            GeometricPrimitive(
                id="hour_hand",
                type=PrimitiveType.LINE,
                points=[Point(x=cx, y=cy), hour_hand_end],
                stroke="#0F172A",
                stroke_width=4.0,
                layer=4,
            )
        )
        primitives.append(
            GeometricPrimitive(
                id="minute_hand",
                type=PrimitiveType.LINE,
                points=[Point(x=cx, y=cy), min_hand_end],
                stroke="#2563EB",
                stroke_width=2.5,
                layer=4,
            )
        )

        # Center pin
        primitives.append(
            GeometricPrimitive(
                id="center_pin",
                type=PrimitiveType.CIRCLE,
                points=[Point(x=cx, y=cy)],
                radius=r_inner * 0.06,
                fill="#EF4444",
                stroke="#991B1B",
                stroke_width=1.5,
                layer=5,
            )
        )

        return DrawingPlan(
            prompt=prompt,
            width=width,
            height=height,
            background_color="#FFFFFF",
            primitives=primitives,
            operations=operations,
            metadata={"theme": "clock", "radius": r_outer},
        )

    # ============================================================
    # GEAR / MECHANICAL SYNTHESIS
    # ============================================================

    def _plan_gear(
        self,
        prompt: str,
        width: int,
        height: int,
        grid: CanvasGrid,
        params: Optional[Dict[str, Any]] = None,
    ) -> DrawingPlan:
        """Constructs interlocking mechanical gears using pattern DB and SUBTRACTION."""
        cx = width / 2.0
        cy = height / 2.0
        gear_r = min(width, height) * 0.35

        primitives, operations = self.pattern_engine.instantiate_pattern(
            pattern_name_or_id="gear",
            instance_id="main_gear",
            params={"center_x": cx, "center_y": cy, "radius": gear_r},
        )

        return DrawingPlan(
            prompt=prompt,
            width=width,
            height=height,
            background_color="#F1F5F9",
            primitives=primitives,
            operations=operations,
            metadata={"theme": "mechanical_gear", "radius": gear_r},
        )

    # ============================================================
    # GEOMETRIC FLOWER / MANDALA SYNTHESIS
    # ============================================================

    def _plan_geometric_flower(
        self,
        prompt: str,
        width: int,
        height: int,
        grid: CanvasGrid,
        params: Optional[Dict[str, Any]] = None,
    ) -> DrawingPlan:
        """Constructs geometric mandala/flower through rotating overlapping circles."""
        cx = width / 2.0
        cy = height / 2.0
        base_r = min(width, height) * 0.22
        petals = 8

        primitives: List[GeometricPrimitive] = []
        operations: List[GeometricOperation] = []

        # Center core
        primitives.append(
            GeometricPrimitive(
                id="flower_core",
                type=PrimitiveType.CIRCLE,
                points=[Point(x=cx, y=cy)],
                radius=base_r * 0.6,
                fill="#FBBF24",
                stroke="#D97706",
                stroke_width=2.0,
                layer=10,
            )
        )

        # Overlapping circles forming petal geometry
        for i in range(petals):
            angle = i * (360.0 / petals)
            p_center = CircleMath(Point(x=cx, y=cy), base_r).point_at_angle(angle)
            primitives.append(
                GeometricPrimitive(
                    id=f"petal_circle_{i}",
                    type=PrimitiveType.CIRCLE,
                    points=[p_center],
                    radius=base_r,
                    fill="#EC4899",
                    stroke="#BE185D",
                    stroke_width=1.5,
                    opacity=0.6,
                    layer=2,
                )
            )

        return DrawingPlan(
            prompt=prompt,
            width=width,
            height=height,
            background_color="#FAF5FF",
            primitives=primitives,
            operations=operations,
            metadata={"theme": "geometric_flower", "petals": petals},
        )

    # ============================================================
    # CELESTIAL / PLANETARY SYNTHESIS
    # ============================================================

    def _plan_celestial(
        self,
        prompt: str,
        width: int,
        height: int,
        grid: CanvasGrid,
        params: Optional[Dict[str, Any]] = None,
    ) -> DrawingPlan:
        """Constructs celestial compositions (Sun, Moon, Orbiting bodies)."""
        cx = width / 2.0
        cy = height / 2.0
        sun_r = min(width, height) * 0.24

        primitives: List[GeometricPrimitive] = []
        operations: List[GeometricOperation] = []

        # Central Sun/Star
        primitives.append(
            GeometricPrimitive(
                id="celestial_sun",
                type=PrimitiveType.CIRCLE,
                points=[Point(x=cx, y=cy)],
                radius=sun_r,
                fill="#F59E0B",
                stroke="#B45309",
                stroke_width=2.5,
                layer=3,
            )
        )

        # Orbital rings (ellipses)
        primitives.append(
            GeometricPrimitive(
                id="orbit_ring_1",
                type=PrimitiveType.ELLIPSE,
                points=[Point(x=cx, y=cy)],
                radius=sun_r * 1.8,
                radius_y=sun_r * 0.9,
                fill="none",
                stroke="#94A3B8",
                stroke_width=1.5,
                layer=1,
            )
        )

        # Orbiting planet
        orbit_p = Point(x=cx + sun_r * 1.8, y=cy)
        primitives.append(
            GeometricPrimitive(
                id="planet_1",
                type=PrimitiveType.CIRCLE,
                points=[orbit_p],
                radius=sun_r * 0.25,
                fill="#3B82F6",
                stroke="#1D4ED8",
                stroke_width=1.5,
                layer=4,
            )
        )

        return DrawingPlan(
            prompt=prompt,
            width=width,
            height=height,
            background_color="#0F172A",
            primitives=primitives,
            operations=operations,
            metadata={"theme": "celestial"},
        )

    # ============================================================
    # GENERIC GEOMETRIC COMPOSITION
    # ============================================================

    def _plan_geometric_composition(
        self,
        prompt: str,
        width: int,
        height: int,
        grid: CanvasGrid,
        params: Optional[Dict[str, Any]] = None,
    ) -> DrawingPlan:
        """Default composition combining circles, sharp strokes, and boolean subtraction."""
        cx = width / 2.0
        cy = height / 2.0
        r_outer = min(width, height) * 0.35
        r_inner = r_outer * 0.55

        primitives = [
            GeometricPrimitive(
                id="base_circle",
                type=PrimitiveType.CIRCLE,
                points=[Point(x=cx, y=cy)],
                radius=r_outer,
                fill="#4F46E5",
                stroke="#3730A3",
                stroke_width=2.5,
                layer=1,
            ),
            GeometricPrimitive(
                id="cutout_circle",
                type=PrimitiveType.CIRCLE,
                points=[Point(x=cx + r_outer * 0.4, y=cy)],
                radius=r_inner,
                fill="#FFFFFF",
                layer=2,
            ),
            GeometricPrimitive(
                id="accent_stroke",
                type=PrimitiveType.LINE,
                points=[Point(x=cx - r_outer * 1.2, y=cy), Point(x=cx + r_outer * 1.2, y=cy)],
                stroke="#F43F5E",
                stroke_width=3.0,
                layer=3,
            ),
        ]

        operations = [
            GeometricOperation(
                id="op_base_cutout",
                operation=OperationType.SUBTRACTION,
                target_id="base_circle",
                operand_ids=["cutout_circle"],
                label="Crescent geometry subtraction",
            )
        ]

        return DrawingPlan(
            prompt=prompt,
            width=width,
            height=height,
            background_color="#F8FAFC",
            primitives=primitives,
            operations=operations,
            metadata={"theme": "geometric_composition"},
        )

    # ============================================================
    # LLM-ASSISTED PLANNING
    # ============================================================

    def _should_use_llm(self, prompt: str) -> bool:
        # If explicitly disabled in env or if key is missing, skip
        if os.environ.get("NEBULA_ARTIST_LLM") == "0":
            return False
        return bool(os.environ.get("OPENAI_API_KEY"))

    def _plan_with_llm(self, prompt: str, width: int, height: int, grid: CanvasGrid) -> Optional[DrawingPlan]:
        """Optionally uses LLM to synthesize geometric primitives for novel requests."""
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import SystemMessage, HumanMessage

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                return None

            model_name = os.environ.get("NEBULA_MODEL", "gpt-5.4-nano")
            base_url = os.environ.get("NEBULA_LLM_BASE_URL", "https://www.routeapi.ai/v1")

            llm = ChatOpenAI(
                model=model_name,
                base_url=base_url,
                api_key=api_key,
                temperature=0.2,
                timeout=10,
            )

            system_instruction = f"""You are Nebula's Geometric Drawing Engine (The Artist).
Translate the user's drawing request into a structured JSON geometric plan.
Canvas dimensions: {width} x {height}.
Available primitives:
- circle: points=[{{"x": cx, "y": cy}}], radius=r
- line: points=[{{"x": x1, "y": y1}}, {{"x": x2, "y": y2}}]
- polygon: points=[{{"x": x1, "y": y1}}, ...]
- bezier_cubic: points=[p0, p1, p2, p3]
Operations:
- union, intersection, subtraction

Return ONLY valid JSON matching this schema:
{{
  "prompt": "{prompt}",
  "width": {width},
  "height": {height},
  "background_color": "#FFFFFF",
  "primitives": [
    {{"id": "id1", "type": "circle", "points": [{{"x": 400, "y": 300}}], "radius": 80, "fill": "#E60000", "stroke": "#000000", "stroke_width": 2, "layer": 1}}
  ],
  "operations": []
}}"""

            response = llm.invoke([
                SystemMessage(content=system_instruction),
                HumanMessage(content=f"Decompose into geometry: {prompt}"),
            ])

            content = response.content.strip()
            # Clean possible markdown block
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            parsed = json.loads(content.strip())
            return DrawingPlan.model_validate(parsed)
        except Exception as e:
            logger.debug("LLM planning omitted: %s", e)
            return None
