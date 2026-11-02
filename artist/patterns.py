"""
Pattern Engine for The Artist (Ability 0.01).

Implements pattern retrieval, parameterization, transformations,
and instantiation for reusable geometric components.
"""

import copy
from typing import Dict, Any, List, Optional, Tuple
from .models import (
    PatternRecord,
    GeometricPrimitive,
    GeometricOperation,
    Point,
    PrimitiveType,
    OperationType,
)
from .geometry import TransformMath
from .db import ArtistDB


class PatternEngine:
    """Manages geometric pattern retrieval, parameterization, and reuse."""

    def __init__(self, db: Optional[ArtistDB] = None):
        self.db = db or ArtistDB()

    def instantiate_pattern(
        self,
        pattern_name_or_id: str,
        instance_id: str,
        params: Optional[Dict[str, Any]] = None,
        translation: Tuple[float, float] = (0.0, 0.0),
        scale: float = 1.0,
        rotation_deg: float = 0.0,
    ) -> Tuple[List[GeometricPrimitive], List[GeometricOperation]]:
        """
        Instantiates a parameterized pattern with specific transforms and parameters.
        Returns newly generated primitives and operations with unique namespaced IDs.
        """
        pattern = self.db.get_pattern_by_name(pattern_name_or_id)
        if not pattern:
            pattern = self.db.get_pattern(pattern_name_or_id)
        if not pattern:
            raise ValueError(f"Pattern '{pattern_name_or_id}' not found in Artist DB")

        merged_params = copy.deepcopy(pattern.parameters)
        if params:
            merged_params.update(params)

        dx, dy = translation
        # If parameters explicitly specify center or offset, factor those in
        if "center_x" in merged_params and "center_y" in merged_params:
            dx += float(merged_params["center_x"])
            dy += float(merged_params["center_y"])
        elif "x" in merged_params and "y" in merged_params:
            dx += float(merged_params["x"])
            dy += float(merged_params["y"])

        if "scale" in merged_params:
            scale *= float(merged_params["scale"])
        elif "radius" in merged_params and "radius" in pattern.parameters and pattern.parameters["radius"]:
            scale *= float(merged_params["radius"]) / float(pattern.parameters["radius"])

        origin = Point(x=dx, y=dy)
        id_prefix = f"{instance_id}_"

        # Transform primitives
        new_primitives: List[GeometricPrimitive] = []
        for raw_prim in pattern.primitives:
            p_data = copy.deepcopy(raw_prim)
            new_id = f"{id_prefix}{p_data['id']}"

            # Transform points
            transformed_points: List[Point] = []
            for pt_dict in p_data.get("points", []):
                pt = Point(x=pt_dict["x"] * scale, y=pt_dict["y"] * scale)
                if rotation_deg != 0.0:
                    pt = TransformMath.rotate_point(pt, Point(x=0.0, y=0.0), rotation_deg)
                pt = TransformMath.translate_point(pt, dx, dy)
                transformed_points.append(pt)

            # Transform radii
            radius = p_data.get("radius")
            if radius is not None:
                radius = float(radius) * scale

            radius_y = p_data.get("radius_y")
            if radius_y is not None:
                radius_y = float(radius_y) * scale

            # Apply parameter color overrides if specified
            fill = p_data.get("fill")
            if "fill" in merged_params:
                fill = merged_params["fill"]
            for param_key, param_val in merged_params.items():
                if param_key.endswith("_color") and param_key.split("_")[0] in p_data["id"]:
                    fill = str(param_val)

            new_prim = GeometricPrimitive(
                id=new_id,
                type=PrimitiveType(p_data["type"]),
                points=transformed_points,
                radius=radius,
                radius_y=radius_y,
                angles=p_data.get("angles"),
                fill=fill,
                stroke=p_data.get("stroke", "#000000"),
                stroke_width=float(p_data.get("stroke_width", 2.0)) * max(1.0, scale),
                opacity=float(p_data.get("opacity", 1.0)),
                layer=int(p_data.get("layer", 0)),
                label=p_data.get("label"),
                closed=bool(p_data.get("closed", True)),
            )
            new_primitives.append(new_prim)

        # Transform operations
        new_operations: List[GeometricOperation] = []
        for raw_op in pattern.operations:
            op_data = copy.deepcopy(raw_op)
            new_op_id = f"{id_prefix}{op_data['id']}"
            target_id = f"{id_prefix}{op_data['target_id']}"
            operand_ids = [f"{id_prefix}{op_id}" for op_id in op_data.get("operand_ids", [])]

            new_op = GeometricOperation(
                id=new_op_id,
                operation=OperationType(op_data["operation"]),
                target_id=target_id,
                operand_ids=operand_ids,
                result_fill=op_data.get("result_fill"),
                result_stroke=op_data.get("result_stroke"),
                result_stroke_width=op_data.get("result_stroke_width"),
                label=op_data.get("label"),
            )
            new_operations.append(new_op)

        return new_primitives, new_operations
