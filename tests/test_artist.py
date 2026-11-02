"""
Tests for The Artist (Ability 0.01) - Geometric Drawing Engine.
"""

import json
from pathlib import Path
import pytest
from PIL import Image

from artist.models import (
    Point,
    PrimitiveType,
    OperationType,
    GeometricPrimitive,
    GeometricOperation,
    DrawingPlan,
    DrawRequest,
    PatternCreateRequest,
)
from artist.geometry import (
    Vector2D,
    CircleMath,
    StrokeMath,
    BezierMath,
    PolygonMath,
    TransformMath,
)
from artist.grid import CanvasGrid
from artist.boolean import BooleanEngine
from artist.db import ArtistDB
from artist.patterns import PatternEngine
from artist.renderer import SVGRenderer, PNGRenderer
from artist.nebula_planner import NebulaPlanner
from artist.engine import ArtistEngine
from artist.tool import artist_draw


# ============================================================
# 1. GEOMETRY & MATH CALCULATIONS
# ============================================================

def test_circle_math_properties():
    c = CircleMath(Point(x=100.0, y=100.0), radius=50.0)
    assert c.diameter == 100.0
    assert pytest.approx(c.circumference, 0.01) == 314.159
    assert pytest.approx(c.area, 0.01) == 7853.98
    assert c.bounding_box == (50.0, 50.0, 150.0, 150.0)


def test_circle_relationships():
    c1 = CircleMath(Point(x=0.0, y=0.0), radius=30.0)
    c2 = CircleMath(Point(x=50.0, y=0.0), radius=30.0)
    rel = c1.relationship_with(c2)
    assert rel["relation"] == "intersecting"
    assert rel["overlap"] == 10.0

    c3 = CircleMath(Point(x=100.0, y=0.0), radius=20.0)
    assert c1.relationship_with(c3)["relation"] == "disjoint"

    c4 = CircleMath(Point(x=50.0, y=0.0), radius=20.0)
    assert c1.relationship_with(c4)["relation"] == "tangent_external"


def test_circle_intersection_points():
    c1 = CircleMath(Point(x=0.0, y=0.0), radius=5.0)
    c2 = CircleMath(Point(x=6.0, y=0.0), radius=5.0)
    pts = c1.intersection_points(c2)
    assert len(pts) == 2
    assert pytest.approx(pts[0].x, 0.01) == 3.0
    assert pytest.approx(abs(pts[0].y), 0.01) == 4.0


def test_stroke_math_and_bezier():
    p1 = Point(x=0.0, y=0.0)
    p2 = Point(x=10.0, y=0.0)
    p3 = Point(x=5.0, y=-5.0)
    p4 = Point(x=5.0, y=5.0)

    assert StrokeMath.length(p1, p2) == 10.0
    intersect = StrokeMath.line_intersection(p1, p2, p3, p4)
    assert intersect is not None
    assert pytest.approx(intersect.x) == 5.0
    assert pytest.approx(intersect.y) == 0.0

    # Quadratic Bezier
    q_mid = BezierMath.eval_quadratic(Point(x=0.0, y=0.0), Point(x=10.0, y=20.0), Point(x=20.0, y=0.0), 0.5)
    assert pytest.approx(q_mid.x) == 10.0
    assert pytest.approx(q_mid.y) == 10.0


def test_polygon_math_and_transforms():
    square = [Point(x=0.0, y=0.0), Point(x=10.0, y=0.0), Point(x=10.0, y=10.0), Point(x=0.0, y=10.0)]
    assert PolygonMath.area(square) == 100.0
    centroid = PolygonMath.centroid(square)
    assert centroid.x == 5.0 and centroid.y == 5.0

    # Rotate 90 degrees around origin
    p = Point(x=10.0, y=0.0)
    rotated = TransformMath.rotate_point(p, Point(x=0.0, y=0.0), 90.0)
    assert pytest.approx(rotated.x, 0.01) == 0.0
    assert pytest.approx(rotated.y, 0.01) == 10.0


# ============================================================
# 2. CANVAS GRID SYSTEM
# ============================================================

def test_canvas_grid_spatial_indexing():
    grid = CanvasGrid(width=800, height=600, resolution=50.0)
    assert grid.cols == 16
    assert grid.rows == 12

    cell_col, cell_row = grid.coord_to_cell_indices(125.0, 75.0)
    assert cell_col == 2 and cell_row == 1

    cells = grid.cells_for_circle(Point(x=100.0, y=100.0), radius=60.0)
    assert len(cells) > 0
    # Center cell (2, 2) must be in the list
    coords = {(c.col, c.row) for c in cells}
    assert (2, 2) in coords


# ============================================================
# 3. BOOLEAN OPERATIONS & MASK COMPOSITION
# ============================================================

def test_boolean_masks_union_and_subtraction():
    prim1 = GeometricPrimitive(
        id="c1",
        type=PrimitiveType.CIRCLE,
        points=[Point(x=50.0, y=50.0)],
        radius=40.0,
    )
    prim2 = GeometricPrimitive(
        id="c2",
        type=PrimitiveType.CIRCLE,
        points=[Point(x=50.0, y=50.0)],
        radius=20.0,
    )

    m1 = BooleanEngine.create_primitive_mask(prim1, 100, 100)
    m2 = BooleanEngine.create_primitive_mask(prim2, 100, 100)

    # Subtraction: inner circle cut out of outer circle
    sub_mask = BooleanEngine.apply_operation(m1, [m2], OperationType.SUBTRACTION)
    assert sub_mask.getpixel((50, 50)) == 0  # Center is cut out
    assert sub_mask.getpixel((50, 20)) > 200  # Rim is intact


# ============================================================
# 4. ARTIST DB & PATTERN REUSE
# ============================================================

def test_artist_db_and_pattern_parameterization(tmp_path):
    db_file = tmp_path / "test_artist.db"
    db = ArtistDB(db_path=db_file)

    # Verify seeded starter patterns
    wheel_pattern = db.get_pattern_by_name("wheel")
    assert wheel_pattern is not None
    assert wheel_pattern.category == "automotive"

    # Instantiate parameterized wheel
    pe = PatternEngine(db)
    prims, ops = pe.instantiate_pattern(
        pattern_name_or_id="wheel",
        instance_id="front_wheel",
        params={"center_x": 200.0, "center_y": 300.0, "radius": 50.0},
    )
    assert len(prims) >= 4
    tire = next(p for p in prims if "tire_outer" in p.id)
    assert tire.points[0].x == 200.0
    assert tire.points[0].y == 300.0
    assert tire.radius == 50.0


def test_pattern_versioning(tmp_path):
    db = ArtistDB(db_path=tmp_path / "version_test.db")
    p1 = db.save_pattern(
        name="custom_star",
        category="shapes",
        parameters={"points": 5},
        primitives=[{"id": "p1", "type": "circle", "radius": 10.0, "points": [{"x": 0.0, "y": 0.0}]}],
        description="Version 1",
    )
    assert p1.version == 1

    p2 = db.save_pattern(
        name="custom_star",
        category="shapes",
        parameters={"points": 6},
        primitives=[{"id": "p1", "type": "circle", "radius": 15.0, "points": [{"x": 0.0, "y": 0.0}]}],
        description="Version 2",
    )
    assert p2.version == 2
    assert p2.parameters["points"] == 6


# ============================================================
# 5. RENDERERS (SVG & PNG)
# ============================================================

def test_svg_renderer_generates_valid_xml():
    plan = DrawingPlan(
        prompt="Test Circle",
        width=400,
        height=300,
        primitives=[
            GeometricPrimitive(
                id="c1",
                type=PrimitiveType.CIRCLE,
                points=[Point(x=200.0, y=150.0)],
                radius=50.0,
                fill="#FF0000",
                stroke="#000000",
                stroke_width=2.0,
            )
        ],
    )
    svg = SVGRenderer.render(plan)
    assert '<svg' in svg
    assert 'viewBox="0 0 400 300"' in svg
    assert '<circle cx="200.00" cy="150.00" r="50.00"' in svg


def test_png_renderer_generates_image():
    plan = DrawingPlan(
        prompt="Test Box",
        width=200,
        height=200,
        primitives=[
            GeometricPrimitive(
                id="box",
                type=PrimitiveType.POLYGON,
                points=[Point(x=50.0, y=50.0), Point(x=150.0, y=50.0), Point(x=150.0, y=150.0), Point(x=50.0, y=150.0)],
                fill="#0000FF",
            )
        ],
    )
    img = PNGRenderer.render(plan)
    assert isinstance(img, Image.Image)
    assert img.size == (200, 200)
    # Check pixel inside blue box
    pixel = img.getpixel((100, 100))
    assert pixel[2] > 200  # Strong Blue component


# ============================================================
# 6. NEBULA INTELLIGENCE & THE ARTIST ENGINE
# ============================================================

def test_ferrari_drawing_execution(tmp_path, monkeypatch):
    from artist.config import ArtistSettings

    settings = ArtistSettings(
        data_dir=tmp_path,
        drawings_dir=tmp_path / "drawings",
        db_path=tmp_path / "artist.db",
    )
    engine = ArtistEngine(settings=settings)

    resp = engine.draw(DrawRequest(prompt="Draw a Ferrari", width=800, height=600))
    assert resp.status == "completed"
    assert "ferrari" in resp.filename.lower()
    assert Path(resp.storage_path).is_file()
    assert (tmp_path / "drawings" / f"{Path(resp.storage_path).stem}.svg").is_file()
    assert resp.primitives_count > 5
    assert resp.operations_count >= 1  # Wheel cutouts


def test_clock_drawing_execution(tmp_path):
    from artist.config import ArtistSettings

    settings = ArtistSettings(
        data_dir=tmp_path,
        drawings_dir=tmp_path / "drawings",
        db_path=tmp_path / "artist.db",
    )
    engine = ArtistEngine(settings=settings)

    resp = engine.draw(DrawRequest(prompt="Draw a clock", width=600, height=600, format="svg"))
    assert resp.status == "completed"
    assert resp.format == "svg"
    assert Path(resp.storage_path).is_file()


def test_artist_draw_tool():
    output = artist_draw.invoke({"prompt": "Draw a Ferrari sports car", "width": 800, "height": 600})
    assert "Successfully constructed geometric drawing" in output
    assert "Drawing ID:" in output
    assert "ferrari" in output.lower()


# ============================================================
# 7. FASTAPI HTTP API ENDPOINTS
# ============================================================

def test_artist_api_endpoints(client):
    # 1. Health check readiness reports artist_engine
    health = client.get("/health/ready").json()
    assert health["checks"]["artist_engine"] is True

    # 2. Draw endpoint
    draw_resp = client.post(
        "/artist/draw",
        json={"prompt": "Draw a Ferrari", "width": 800, "height": 600, "format": "png"},
    )
    assert draw_resp.status_code == 200
    data = draw_resp.json()
    drawing_id = data["drawing_id"]
    assert drawing_id.startswith("draw-")
    assert data["status"] == "completed"

    # 3. Retrieve drawing metadata
    get_resp = client.get(f"/artist/drawings/{drawing_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["drawing_id"] == drawing_id

    # 4. List drawings
    list_resp = client.get("/artist/drawings")
    assert list_resp.status_code == 200
    assert len(list_resp.json()) >= 1

    # 5. Fetch image file (PNG)
    png_file = client.get(f"/artist/drawings/{drawing_id}/file")
    assert png_file.status_code == 200
    assert png_file.headers["content-type"] == "image/png"
    assert len(png_file.content) > 1000

    # 6. Fetch vector file (SVG)
    svg_file = client.get(f"/artist/drawings/{drawing_id}/file?format=svg")
    assert svg_file.status_code == 200
    assert "image/svg+xml" in svg_file.headers["content-type"]
    assert b"<svg" in svg_file.content

    # 7. List patterns
    patterns_resp = client.get("/artist/patterns")
    assert patterns_resp.status_code == 200
    names = [p["name"] for p in patterns_resp.json()]
    assert "wheel" in names
    assert "gear" in names

    # 8. Retrieve specific pattern
    wheel_resp = client.get("/artist/patterns/wheel")
    assert wheel_resp.status_code == 200
    assert wheel_resp.json()["name"] == "wheel"
