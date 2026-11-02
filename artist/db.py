"""
Artist DB: Persistent Database for Patterns and Drawings (Ability 0.01).

Implements SQLite storage for reusable parameterized patterns, pattern
versioning, and completed drawing records.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any

from .config import ARTIST_SETTINGS
from .models import PatternRecord, DrawingRecord


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ArtistDB:
    """Manages persistent SQLite storage for The Artist."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or ARTIST_SETTINGS.db_path
        self._ensure_db()

    def _get_connection(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_db(self) -> None:
        """Creates tables and seeds starter patterns if needed."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS artist_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    name TEXT UNIQUE,
                    category TEXT,
                    version INTEGER DEFAULT 1,
                    description TEXT,
                    parameters_json TEXT,
                    primitives_json TEXT,
                    operations_json TEXT,
                    tags_json TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS artist_drawings (
                    drawing_id TEXT PRIMARY KEY,
                    prompt TEXT,
                    filename TEXT,
                    format TEXT,
                    width INTEGER,
                    height INTEGER,
                    storage_path TEXT,
                    status TEXT,
                    primitives_count INTEGER,
                    operations_count INTEGER,
                    metadata_json TEXT,
                    created_at TEXT
                )
                """
            )
            conn.commit()

        self._seed_default_patterns()

    def _seed_default_patterns(self) -> None:
        """Seeds foundational reusable geometric patterns (e.g. wheel, gear, eye)."""
        starter_patterns = [
            {
                "pattern_id": "pat-wheel-01",
                "name": "wheel",
                "category": "automotive",
                "description": "Geometric automotive wheel with tire, rim, hub, and bolt circles.",
                "parameters": {
                    "center_x": 0.0,
                    "center_y": 0.0,
                    "radius": 40.0,
                    "tire_color": "#1C1C1E",
                    "rim_color": "#D1D1D6",
                    "hub_color": "#8E8E93",
                },
                "primitives": [
                    {
                        "id": "tire_outer",
                        "type": "circle",
                        "points": [{"x": 0.0, "y": 0.0}],
                        "radius": 40.0,
                        "fill": "#1C1C1E",
                        "stroke": "#000000",
                        "stroke_width": 2.0,
                        "layer": 1,
                        "label": "Tire rubber",
                    },
                    {
                        "id": "rim_outer",
                        "type": "circle",
                        "points": [{"x": 0.0, "y": 0.0}],
                        "radius": 28.0,
                        "fill": "#D1D1D6",
                        "stroke": "#48484A",
                        "stroke_width": 1.5,
                        "layer": 2,
                        "label": "Alloy rim",
                    },
                    {
                        "id": "rim_inner",
                        "type": "circle",
                        "points": [{"x": 0.0, "y": 0.0}],
                        "radius": 14.0,
                        "fill": "#8E8E93",
                        "stroke": "#3A3A3C",
                        "stroke_width": 1.5,
                        "layer": 3,
                        "label": "Inner rim recess",
                    },
                    {
                        "id": "center_hub",
                        "type": "circle",
                        "points": [{"x": 0.0, "y": 0.0}],
                        "radius": 6.0,
                        "fill": "#E5E5EA",
                        "stroke": "#1C1C1E",
                        "stroke_width": 1.0,
                        "layer": 4,
                        "label": "Center hub cap",
                    },
                ],
                "operations": [
                    {
                        "id": "op_wheel_rim_cut",
                        "operation": "union",
                        "target_id": "tire_outer",
                        "operand_ids": ["rim_outer"],
                        "label": "Combine wheel assembly",
                    }
                ],
                "tags": ["wheel", "tire", "car", "vehicle", "round"],
            },
            {
                "pattern_id": "pat-gear-01",
                "name": "gear",
                "category": "mechanical",
                "description": "Mechanical cog gear with axle hole subtracted from center.",
                "parameters": {
                    "center_x": 0.0,
                    "center_y": 0.0,
                    "radius": 50.0,
                    "gear_color": "#708090",
                },
                "primitives": [
                    {
                        "id": "gear_body",
                        "type": "circle",
                        "points": [{"x": 0.0, "y": 0.0}],
                        "radius": 50.0,
                        "fill": "#708090",
                        "stroke": "#2F4F4F",
                        "stroke_width": 2.0,
                        "layer": 1,
                    },
                    {
                        "id": "gear_hole",
                        "type": "circle",
                        "points": [{"x": 0.0, "y": 0.0}],
                        "radius": 18.0,
                        "fill": "#FFFFFF",
                        "stroke": "#2F4F4F",
                        "stroke_width": 2.0,
                        "layer": 2,
                    },
                ],
                "operations": [
                    {
                        "id": "op_gear_hole_sub",
                        "operation": "subtraction",
                        "target_id": "gear_body",
                        "operand_ids": ["gear_hole"],
                        "label": "Axle hole subtraction",
                    }
                ],
                "tags": ["gear", "cog", "mechanical", "clock"],
            },
            {
                "pattern_id": "pat-chassis-01",
                "name": "sports_chassis",
                "category": "automotive",
                "description": "Aerodynamic sports car chassis silhouette with wheel arch cutouts.",
                "parameters": {
                    "x": 0.0,
                    "y": 0.0,
                    "width": 320.0,
                    "height": 70.0,
                    "body_color": "#E60000",
                },
                "primitives": [
                    {
                        "id": "car_body_polygon",
                        "type": "polygon",
                        "points": [
                            {"x": 10.0, "y": 60.0},
                            {"x": 30.0, "y": 35.0},
                            {"x": 80.0, "y": 30.0},
                            {"x": 140.0, "y": 8.0},
                            {"x": 220.0, "y": 8.0},
                            {"x": 280.0, "y": 35.0},
                            {"x": 310.0, "y": 45.0},
                            {"x": 310.0, "y": 65.0},
                            {"x": 270.0, "y": 65.0},
                            {"x": 230.0, "y": 65.0},
                            {"x": 90.0, "y": 65.0},
                            {"x": 50.0, "y": 65.0},
                            {"x": 10.0, "y": 65.0},
                        ],
                        "fill": "#E60000",
                        "stroke": "#8B0000",
                        "stroke_width": 2.0,
                        "layer": 1,
                        "label": "Chassis shell",
                    },
                    {
                        "id": "cockpit_glass",
                        "type": "polygon",
                        "points": [
                            {"x": 105.0, "y": 30.0},
                            {"x": 148.0, "y": 12.0},
                            {"x": 212.0, "y": 12.0},
                            {"x": 235.0, "y": 30.0},
                        ],
                        "fill": "#1F2937",
                        "stroke": "#111827",
                        "stroke_width": 1.5,
                        "layer": 2,
                        "label": "Windshield and side glass",
                    },
                    {
                        "id": "front_wheel_well",
                        "type": "circle",
                        "points": [{"x": 70.0, "y": 65.0}],
                        "radius": 24.0,
                        "fill": "#FFFFFF",
                        "layer": 3,
                    },
                    {
                        "id": "rear_wheel_well",
                        "type": "circle",
                        "points": [{"x": 250.0, "y": 65.0}],
                        "radius": 24.0,
                        "fill": "#FFFFFF",
                        "layer": 3,
                    },
                ],
                "operations": [
                    {
                        "id": "op_wheel_arch_cutouts",
                        "operation": "subtraction",
                        "target_id": "car_body_polygon",
                        "operand_ids": ["front_wheel_well", "rear_wheel_well"],
                        "label": "Cut out wheel arches from chassis",
                    }
                ],
                "tags": ["car", "chassis", "ferrari", "sports_car", "body"],
            },
        ]

        with self._get_connection() as conn:
            for p in starter_patterns:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO artist_patterns (
                        pattern_id, name, category, version, description,
                        parameters_json, primitives_json, operations_json,
                        tags_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        p["pattern_id"],
                        p["name"],
                        p["category"],
                        1,
                        p["description"],
                        json.dumps(p["parameters"]),
                        json.dumps(p["primitives"]),
                        json.dumps(p["operations"]),
                        json.dumps(p["tags"]),
                        _now_iso(),
                        _now_iso(),
                    ),
                )
            conn.commit()

    # ============================================================
    # PATTERN OPERATIONS
    # ============================================================

    def save_pattern(
        self,
        name: str,
        category: str,
        parameters: Dict[str, Any],
        primitives: List[Dict[str, Any]],
        operations: Optional[List[Dict[str, Any]]] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> PatternRecord:
        """Stores or increments version of a reusable geometric pattern."""
        now = _now_iso()
        existing = self.get_pattern_by_name(name)

        if existing:
            new_version = existing.version + 1
            pattern_id = existing.pattern_id
            with self._get_connection() as conn:
                conn.execute(
                    """
                    UPDATE artist_patterns
                    SET category = ?, version = ?, description = ?,
                        parameters_json = ?, primitives_json = ?, operations_json = ?,
                        tags_json = ?, updated_at = ?
                    WHERE pattern_id = ?
                    """,
                    (
                        category,
                        new_version,
                        description,
                        json.dumps(parameters),
                        json.dumps(primitives),
                        json.dumps(operations or []),
                        json.dumps(tags or []),
                        now,
                        pattern_id,
                    ),
                )
                conn.commit()
            return self.get_pattern(pattern_id)  # type: ignore

        pattern_id = f"pat-{uuid.uuid4().hex[:8]}"
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO artist_patterns (
                    pattern_id, name, category, version, description,
                    parameters_json, primitives_json, operations_json,
                    tags_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pattern_id,
                    name,
                    category,
                    1,
                    description,
                    json.dumps(parameters),
                    json.dumps(primitives),
                    json.dumps(operations or []),
                    json.dumps(tags or []),
                    now,
                    now,
                ),
            )
            conn.commit()

        return self.get_pattern(pattern_id)  # type: ignore

    def get_pattern(self, pattern_id: str) -> Optional[PatternRecord]:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM artist_patterns WHERE pattern_id = ?", (pattern_id,)
            ).fetchone()
            if not row:
                return None
            return self._row_to_pattern(row)

    def get_pattern_by_name(self, name: str) -> Optional[PatternRecord]:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM artist_patterns WHERE name = ?", (name,)
            ).fetchone()
            if not row:
                return None
            return self._row_to_pattern(row)

    def list_patterns(self, category: Optional[str] = None) -> List[PatternRecord]:
        with self._get_connection() as conn:
            if category:
                rows = conn.execute(
                    "SELECT * FROM artist_patterns WHERE category = ? ORDER BY name ASC",
                    (category,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM artist_patterns ORDER BY category ASC, name ASC"
                ).fetchall()
            return [self._row_to_pattern(r) for r in rows]

    def _row_to_pattern(self, row: sqlite3.Row) -> PatternRecord:
        return PatternRecord(
            pattern_id=row["pattern_id"],
            name=row["name"],
            category=row["category"],
            version=row["version"],
            description=row["description"] or "",
            parameters=json.loads(row["parameters_json"] or "{}"),
            primitives=json.loads(row["primitives_json"] or "[]"),
            operations=json.loads(row["operations_json"] or "[]"),
            tags=json.loads(row["tags_json"] or "[]"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # ============================================================
    # DRAWING RECORDS
    # ============================================================

    def save_drawing(self, record: DrawingRecord) -> DrawingRecord:
        now = record.created_at or _now_iso()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO artist_drawings (
                    drawing_id, prompt, filename, format, width, height,
                    storage_path, status, primitives_count, operations_count,
                    metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.drawing_id,
                    record.prompt,
                    record.filename,
                    record.format,
                    record.width,
                    record.height,
                    record.storage_path,
                    record.status,
                    record.primitives_count,
                    record.operations_count,
                    json.dumps(record.metadata),
                    now,
                ),
            )
            conn.commit()
        return record

    def get_drawing(self, drawing_id: str) -> Optional[DrawingRecord]:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM artist_drawings WHERE drawing_id = ?", (drawing_id,)
            ).fetchone()
            if not row:
                return None
            return DrawingRecord(
                drawing_id=row["drawing_id"],
                prompt=row["prompt"],
                filename=row["filename"],
                format=row["format"],
                width=row["width"],
                height=row["height"],
                storage_path=row["storage_path"],
                status=row["status"],
                primitives_count=row["primitives_count"],
                operations_count=row["operations_count"],
                metadata=json.loads(row["metadata_json"] or "{}"),
                created_at=row["created_at"],
            )

    def list_drawings(self, limit: int = 50) -> List[DrawingRecord]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM artist_drawings ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [
                DrawingRecord(
                    drawing_id=r["drawing_id"],
                    prompt=r["prompt"],
                    filename=r["filename"],
                    format=r["format"],
                    width=r["width"],
                    height=r["height"],
                    storage_path=r["storage_path"],
                    status=r["status"],
                    primitives_count=r["primitives_count"],
                    operations_count=r["operations_count"],
                    metadata=json.loads(r["metadata_json"] or "{}"),
                    created_at=r["created_at"],
                )
                for r in rows
            ]
