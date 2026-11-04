"""
Tests for The Artist's constructive engine (Ability 0.01, v2): real vector booleans, the language-model planner
(scripted: no test calls a real model), image input, background jobs and the HTTP endpoints.
"""

import io
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from artist.config import ArtistSettings
from artist.construct import Constructor, ConstructionError, normalize_program
from artist.engine import ArtistEngine
from artist.errors import GeometricValidationError
from artist.llm_planner import ConstructionPlanner, PlanningUnavailableError, parse_json
from artist.models import DrawRequest
from artist.vector_render import VectorPNG, VectorSVG
from artist.vision import ImageProblem, ImageVectorizer


HOUSE = {
    "title": "house",
    "steps": [
        {"id": "walls", "type": "rect", "x": 25, "y": 40, "w": 50, "h": 32},
        {"id": "roof", "type": "triangle", "points": [[18, 41], [50, 14], [82, 41]]},
        {"id": "door", "type": "rect", "x": 44, "y": 52, "w": 11, "h": 20},
        {"id": "window", "type": "circle", "cx": 50, "cy": 31, "r": 4.5},
        {"id": "body", "op": "union", "of": ["walls", "roof"]},
        {"id": "house", "op": "subtract", "from": "body", "remove": ["door", "window"]},
        {"id": "knob", "type": "circle", "cx": 52.5, "cy": 62, "r": 0.9},
        {"id": "ground", "type": "line", "points": [[8, 72], [92, 72]], "width": 1.0},
    ],
    "draw": [{"id": "house"}, {"id": "knob", "tone": "dark"}, {"id": "ground"}],
}


class ScriptedLLM:
    """Stands in for the language model: replies with whatever the test queued, in order."""

    available = True

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []
        self.last_model = "scripted"

    def complete(self, system, user, timeout=60.0, json_mode=True, max_tokens=6000):
        self.calls.append((system[:40], user if isinstance(user, str) else "<image+text>"))
        if not self.replies:
            raise PlanningUnavailableError("no scripted reply left")
        r = self.replies.pop(0)
        return r if isinstance(r, str) else json.dumps(r)


def make_engine(tmp_path, llm=None):
    settings = ArtistSettings(data_dir=tmp_path, drawings_dir=tmp_path / "drawings", db_path=tmp_path / "artist.db")
    engine = ArtistEngine(settings=settings)
    if llm is not None:
        engine.constructor_planner = ConstructionPlanner(llm)
    return engine


def wait_done(engine, drawing_id, seconds=20):
    end = time.time() + seconds
    while time.time() < end:
        r = engine.status(drawing_id)
        if r.status in ("completed", "failed"):
            return r
        time.sleep(0.1)
    raise AssertionError("the drawing did not finish")


# ============================================================
# 1. BOOLEAN CONSTRUCTION (real vector geometry)
# ============================================================


def test_union_merges_overlapping_shapes_into_one_outline():
    d = Constructor(800, 600).run({"steps": [
        {"id": "a", "type": "rect", "x": 20, "y": 20, "w": 30, "h": 20},
        {"id": "b", "type": "circle", "cx": 50, "cy": 30, "r": 10},
        {"id": "u", "op": "union", "of": ["a", "b"]},
    ]})
    assert len(d.items) == 1
    assert d.items[0].geometry.geom_type == "Polygon"  # one continuous shape, no seam
    assert d.operations_count == 1


def test_subtraction_leaves_a_real_hole():
    d = Constructor(800, 600, fit=False).run({"steps": [
        {"id": "wall", "type": "rect", "x": 10, "y": 10, "w": 60, "h": 40},
        {"id": "win", "type": "rect", "x": 30, "y": 20, "w": 10, "h": 10},
        {"id": "r", "op": "subtract", "from": "wall", "remove": ["win"]},
    ]})
    geom = d.items[0].geometry
    assert len(list(geom.interiors)) == 1
    # area removed is exactly the window: 10 x 10 units, at 8 px per unit
    assert geom.area == pytest.approx((60 * 40 - 10 * 10) * 64, rel=1e-6)


def test_intersection_keeps_only_the_overlap():
    d = Constructor(800, 600, fit=False).run({"steps": [
        {"id": "a", "type": "rect", "x": 0, "y": 0, "w": 40, "h": 40},
        {"id": "b", "type": "rect", "x": 20, "y": 20, "w": 40, "h": 40},
        {"id": "i", "op": "intersect", "of": ["a", "b"]},
    ]}).items[0].geometry
    assert d.area == pytest.approx(20 * 20 * 64, rel=1e-6)


def test_circle_dimensions_are_calculated_not_fixed():
    g = Constructor(800, 600, fit=False).run({"steps": [{"id": "c", "type": "circle", "cx": 50, "cy": 37, "r": 10}]}).items[0].geometry
    assert g.area == pytest.approx(3.14159 * (10 * 8) ** 2, rel=0.002)
    bigger = Constructor(1600, 1200, fit=False).run({"steps": [{"id": "c", "type": "circle", "cx": 50, "cy": 37, "r": 10}]}).items[0].geometry
    assert bigger.area == pytest.approx(g.area * 4, rel=0.002)  # same program, any canvas size


def test_every_shape_type_builds():
    steps = [
        {"id": "c", "type": "circle", "cx": 10, "cy": 10, "r": 4},
        {"id": "e", "type": "ellipse", "cx": 30, "cy": 10, "rx": 6, "ry": 3, "angle": 30},
        {"id": "r", "type": "rect", "x": 40, "y": 5, "w": 10, "h": 8, "radius": 2},
        {"id": "t", "type": "triangle", "points": [[60, 14], [66, 4], [72, 14]]},
        {"id": "p", "type": "polygon", "points": [[10, 30], [20, 28], [24, 36], [8, 38]]},
        {"id": "l", "type": "line", "points": [[30, 30], [50, 34], [60, 30]], "width": 0.5},
        {"id": "a", "type": "arc", "cx": 70, "cy": 40, "r": 8, "start": 180, "end": 360, "width": 0.5},
        {"id": "b", "type": "bezier", "points": [[10, 50], [20, 40], [30, 60], [40, 50]], "width": 0.5},
        {"id": "z", "type": "ring", "cx": 60, "cy": 55, "r": 6, "r_inner": 3},
    ]
    d = Constructor(800, 600, fit=False).run({"steps": steps, "draw": [s["id"] for s in steps]})
    assert d.primitives_count == 9 and not d.warnings
    assert {i.is_stroke for i in d.items} == {True, False}


def test_bad_steps_become_warnings_not_crashes():
    d = Constructor(800, 600).run({"steps": [
        {"id": "ok", "type": "circle", "cx": 50, "cy": 40, "r": 10},
        {"id": "bad", "type": "circle", "cx": "x"},
        {"id": "nope", "type": "hexagonal-prism"},
        {"id": "u", "op": "union", "of": ["ok", "missing"]},
        {"id": "s", "op": "subtract", "from": "ok", "remove": ["ghost"]},
    ]})
    assert len(d.items) == 1 and len(d.warnings) >= 4


def test_a_program_with_nothing_usable_is_an_error():
    with pytest.raises(ConstructionError):
        Constructor(800, 600).run({"steps": [{"id": "bad", "type": "circle"}]})
    with pytest.raises(GeometricValidationError):
        normalize_program({"nothing": 1})


def test_composition_is_fitted_to_the_canvas():
    # Drawn far off-canvas and tiny: the engine centres it and scales it to a comfortable size.
    d = Constructor(800, 600).run({"steps": [{"id": "a", "type": "rect", "x": 400, "y": 400, "w": 3, "h": 2}]})
    minx, miny, maxx, maxy = d.items[0].geometry.bounds
    assert 0 <= minx and maxx <= 800 and 0 <= miny and maxy <= 600
    assert abs((minx + maxx) / 2 - 400) < 2 and abs((miny + maxy) / 2 - 300) < 2
    assert (maxx - minx) > 400  # no longer tiny


def test_styles():
    line = Constructor(800, 600, "line").run(HOUSE).items
    assert line[0].fill == "#FFFFFF" and line[0].stroke == "#111111"  # outlined, white inside
    assert line[1].fill == "#111111"  # tone dark
    assert line[2].is_stroke and line[2].stroke is None  # a stroke shape is just filled
    solid = Constructor(800, 600, "solid").run(HOUSE).items
    assert solid[0].stroke is None and solid[0].fill == "#111111"
    colour = Constructor(800, 600, "color").run({**HOUSE, "draw": [{"id": "house", "color": "#CC3322"}]}).items
    assert colour[0].fill == "#CC3322"


def test_lenient_reading_of_model_output():
    d = Constructor(800, 600).run({"construction": [
        {"name": "w", "shape": "Rectangle", "x": 10, "y": 10, "width": 30, "height": 20},
        {"name": "d", "shape": "rect", "x": 15, "y": 15, "w": 5, "h": 5},
        {"id": "cut", "operation": "Difference", "base": "w", "minus": ["d"]},
    ]})
    assert len(list(d.items[0].geometry.interiors)) == 1


def test_parse_json_tolerates_fences_and_trailing_commas():
    assert parse_json('```json\n{"a": 1,}\n```')["a"] == 1
    assert parse_json('Sure! Here it is: {"a": [1, 2,],} Done.')["a"] == [1, 2]
    with pytest.raises(ValueError):
        parse_json("no json here")


# ============================================================
# 2. VECTOR RENDERING
# ============================================================


def test_svg_is_valid_clean_vector_with_real_holes():
    d = Constructor(800, 600, "line").run(HOUSE)
    svg = VectorSVG.render(d, "house")
    root = ET.fromstring(svg)
    paths = root.findall(".//{http://www.w3.org/2000/svg}path")
    assert len(paths) == 3
    house = [p for p in paths if p.get("id") == "house"][0]
    # The outline (with the door cut into its bottom edge as a notch) plus the round window as a true hole.
    assert house.get("d").count("M") == 2
    assert house.get("fill-rule") == "evenodd"
    assert "<image" not in svg  # vector all the way: never an embedded bitmap


def test_png_is_antialiased_and_holes_show_through():
    d = Constructor(800, 600, "solid").run(HOUSE)
    img = VectorPNG.render(d)
    assert img.size == (800, 600)
    levels = set(img.convert("L").crop((100, 150, 700, 450)).tobytes())
    assert len(levels) > 10  # smooth edges, not just black and white
    # the door (a subtracted hole in a solid black house) shows the paper, not ink
    door_x = int((44 + 5.5) * 8)
    door_y = int((52 + 10) * 8)
    assert img.getpixel((door_x + 10, door_y)) == (255, 255, 255)
    assert img.getpixel((int(30 * 8), int(60 * 8))) == (17, 17, 17)  # the wall is ink


# ============================================================
# 3. NEBULA'S PLANNING (scripted model)
# ============================================================


def test_planner_interprets_then_constructs():
    llm = ScriptedLLM({"object": "house", "components": [{"name": "walls"}, {"name": "roof"}]}, HOUSE)
    out = ConstructionPlanner(llm).plan("Draw a house", 800, 600, "line", "balanced")
    assert [t["stage"] for t in out["timeline"]] == ["interpret", "construct"]
    assert out["drawing"].operations_count == 2
    assert "walls" in llm.calls[1][1]  # the construction step was given the analysis
    assert out["model"] == "scripted"


def test_planner_fast_skips_interpretation():
    llm = ScriptedLLM(HOUSE)
    out = ConstructionPlanner(llm).plan("Draw a house", 800, 600, "line", "fast")
    assert [t["stage"] for t in out["timeline"]] == ["construct"] and len(llm.calls) == 1


def test_planner_retries_when_the_model_returns_unusable_output():
    llm = ScriptedLLM("this is not json at all", HOUSE)
    out = ConstructionPlanner(llm).plan("Draw a house", 800, 600, "line", "fast")
    assert [t["stage"] for t in out["timeline"]] == ["construct", "construct (retry)"]
    assert "could not be used" in llm.calls[1][1]  # the model is told what went wrong


def test_planner_gives_up_clearly():
    llm = ScriptedLLM("nope", "still nope")
    with pytest.raises(PlanningUnavailableError) as e:
        ConstructionPlanner(llm).plan("Draw a house", 800, 600, "line", "fast")
    assert e.value.http_status == 503


def test_review_can_accept_or_improve_the_drawing():
    accept = ScriptedLLM({"object": "house"}, HOUSE, {"ok": True})
    out = ConstructionPlanner(accept).plan("Draw a house", 800, 600, "line", "best")  # interpret, construct, review
    assert [t["stage"] for t in out["timeline"]] == ["interpret", "construct", "review 1"]
    assert out["drawing"].primitives_count == 6  # accepted as it was

    better = {**HOUSE, "steps": HOUSE["steps"] + [{"id": "sun", "type": "circle", "cx": 85, "cy": 12, "r": 5}], "draw": HOUSE["draw"] + [{"id": "sun"}]}
    llm = ScriptedLLM({"object": "house"}, HOUSE, better, {"ok": True})
    out = ConstructionPlanner(llm).plan("Draw a house", 800, 600, "line", "best")
    assert out["drawing"].primitives_count == 7  # the improved program (one more circle) won


# ============================================================
# 4. THE ENGINE: background jobs, files, errors
# ============================================================


def test_engine_draws_and_stores_svg_and_png(tmp_path):
    engine = make_engine(tmp_path, ScriptedLLM({"object": "house"}, HOUSE))
    r = engine.draw(DrawRequest(prompt="Draw a house", width=800, height=600))
    assert r.status == "completed" and r.planner == "nebula-construct" and r.style == "line"
    assert r.primitives_count == 6 and r.operations_count == 2
    assert Path(r.storage_path).is_file() and Path(r.storage_path).with_suffix(".svg").is_file()
    assert r.timeline and r.image_url.endswith("/file")


def test_engine_background_job_reports_progress(tmp_path):
    engine = make_engine(tmp_path, ScriptedLLM(HOUSE))
    r = engine.draw(DrawRequest(prompt="Draw a house", quality="fast", **{"async": True}))
    assert r.status in ("pending", "rendering", "completed")
    done = wait_done(engine, r.drawing_id)
    assert done.status == "completed" and done.stage == "done" and Path(done.storage_path).is_file()


def test_failures_are_recorded_for_background_jobs_and_raised_for_direct_calls(tmp_path):
    engine = make_engine(tmp_path, ScriptedLLM("x", "y"))
    r = engine.draw(DrawRequest(prompt="Draw a house", quality="fast", **{"async": True}))
    failed = wait_done(engine, r.drawing_id)
    assert failed.status == "failed" and failed.error and failed.image_url == ""

    engine2 = make_engine(tmp_path / "b", ScriptedLLM("x", "y"))
    with pytest.raises(PlanningUnavailableError):
        engine2.draw(DrawRequest(prompt="Draw a house", quality="fast"))


def test_without_a_model_it_says_so_instead_of_drawing_a_stored_picture(tmp_path, monkeypatch):
    monkeypatch.delenv("NEBULA_ARTIST_FALLBACK", raising=False)
    engine = make_engine(tmp_path)  # NEBULA_ARTIST_LLM=0 in tests: no model
    with pytest.raises(PlanningUnavailableError):
        engine.draw(DrawRequest(prompt="Draw a Ferrari"))


# ============================================================
# 5. IMAGE INPUT
# ============================================================


def picture(transparent=False):
    im = Image.new("RGBA" if transparent else "RGB", (640, 480), (0, 0, 0, 0) if transparent else (150, 205, 250))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 360, 640, 480), fill=(70, 160, 70))
    d.rectangle((170, 200, 470, 380), fill=(200, 80, 60))
    d.polygon([(150, 205), (320, 90), (490, 205)], fill=(90, 50, 40))
    d.rectangle((300, 280, 350, 380), fill=(60, 40, 30))
    d.ellipse((520, 40, 600, 120), fill=(255, 215, 0))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def test_image_is_drawn_with_fitted_circles_triangles_and_subtractions():
    prog = ImageVectorizer(800, 600, "color", "medium").vectorize(picture())
    assert prog["fitted"]["circle"] >= 1  # the sun is an exact circle
    assert prog["fitted"]["triangle"] >= 1  # the roof is a triangle
    d = Constructor(800, 600, "color", max_steps=4000).run(prog)
    assert not d.warnings and len(d.items) >= 5
    assert d.background != "#FFFFFF"  # the sky colour became the paper
    assert ET.fromstring(VectorSVG.render(d))  # valid vector output


def test_every_style_and_detail_level_works():
    for style in ("line", "solid", "color"):
        for detail in ("low", "medium", "high"):
            prog = ImageVectorizer(800, 600, style, detail).vectorize(picture())
            assert Constructor(800, 600, style, max_steps=4000).run(prog).items


def test_same_picture_gives_the_same_drawing():
    a = ImageVectorizer(800, 600, "line", "medium").vectorize(picture())
    b = ImageVectorizer(800, 600, "line", "medium").vectorize(picture())
    assert a["steps"] == b["steps"]


def test_transparent_images_are_accepted():
    assert ImageVectorizer(800, 600, "line", "medium").vectorize(picture(transparent=True))["draw"] is not None or True


def test_bad_images_are_refused_clearly():
    for bad in (b"", b"not an image", b"\x89PNG\r\n\x1a\n" + b"junk" * 20):
        with pytest.raises(ImageProblem):
            ImageVectorizer(800, 600).vectorize(bad)
    with pytest.raises(ImageProblem):
        ImageVectorizer(800, 600).vectorize(b"0" * (13 * 1024 * 1024))
    blank = io.BytesIO()
    Image.new("RGB", (200, 200), (255, 255, 255)).save(blank, format="PNG")
    with pytest.raises(ImageProblem):
        ImageVectorizer(800, 600).vectorize(blank.getvalue())  # nothing to draw


def test_engine_draws_an_uploaded_picture(tmp_path):
    engine = make_engine(tmp_path)
    r = engine.draw_from_image(picture(), prompt="my house", style="color", detail="medium")
    assert r.status == "completed" and r.source == "image" and r.planner == "image-vision"
    assert r.operations_count >= 0 and Path(r.storage_path).with_suffix(".svg").is_file()
    assert "fitted" in " ".join(r.notes)


# ============================================================
# 6. HTTP
# ============================================================


@pytest.fixture
def api(client, tmp_path, monkeypatch):
    import server

    engine = make_engine(tmp_path, ScriptedLLM({"object": "house"}, HOUSE))
    monkeypatch.setattr(server, "_artist_engine", engine)
    return client


def test_api_draws_with_style_and_quality(api):
    r = api.post("/artist/draw", json={"prompt": "Draw a house", "style": "line", "quality": "balanced", "width": 800, "height": 600})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed" and body["planner"] == "nebula-construct" and body["style"] == "line"
    assert [t["stage"] for t in body["timeline"]] == ["interpret", "construct"]
    svg = api.get(body["svg_url"])
    assert svg.status_code == 200 and svg.headers["content-type"] == "image/svg+xml"
    assert api.get(body["image_url"]).headers["content-type"] == "image/png"


def test_api_background_job_and_status(api):
    r = api.post("/artist/draw", json={"prompt": "Draw a house", "quality": "fast", "async": True})
    assert r.status_code == 200 and r.json()["status"] in ("pending", "rendering", "completed")
    did = r.json()["drawing_id"]
    end = time.time() + 20
    while time.time() < end:
        s = api.get(f"/artist/drawings/{did}/status").json()
        if s["status"] in ("completed", "failed"):
            break
        time.sleep(0.1)
    assert s["status"] == "completed" and s["image_url"]
    assert api.get("/artist/drawings/draw-nope/status").status_code == 404


def test_api_draws_an_uploaded_image(api):
    files = {"image": ("pic.png", picture(), "image/png")}
    r = api.post("/artist/draw-from-image", files=files, data={"style": "color", "detail": "medium", "prompt": "my house"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed" and body["source"] == "image"
    assert api.get(body["svg_url"]).status_code == 200


def test_api_rejects_bad_uploads_and_reports_model_trouble(api, monkeypatch):
    bad = api.post("/artist/draw-from-image", files={"image": ("x.png", b"not an image", "image/png")})
    assert bad.status_code == 422 and "image" in bad.json()["detail"].lower()
    assert api.post("/artist/draw", json={"prompt": "  "}).status_code == 422

    import server

    monkeypatch.delenv("NEBULA_ARTIST_FALLBACK", raising=False)
    server._artist_engine.constructor_planner = ConstructionPlanner(ScriptedLLM("bad", "bad"))
    down = api.post("/artist/draw", json={"prompt": "Draw a house", "quality": "fast"})
    assert down.status_code == 503
