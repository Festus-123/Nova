"""
Tests for G-Stroke 2.0 (the sketching algorithm of The Artist) and for the guarantee that 1.0 stays the default.
No test calls a real model: the language model is scripted.
"""

import io
import json
import time
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from PIL import Image, ImageDraw

from artist.config import ArtistSettings
from artist.engine import ArtistEngine
from artist.errors import GeometricValidationError
from artist.gstroke import (
    ALGORITHMS,
    DEFAULT_ALGORITHM,
    ImageSketcher,
    SketchPlanner,
    SketchPNG,
    SketchSVG,
    Stroke,
    brush_names,
    get_brush,
    make_strand,
    program_to_sketch,
    resolve_algorithm,
)
from artist.gstroke.sketch import Sketch, smooth_path, stroke_parts
from artist.llm_planner import ConstructionPlanner, PlanningUnavailableError
from artist.models import DrawRequest


class ScriptedLLM:
    available = True

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []
        self.last_model = "scripted"

    def complete(self, system, user, timeout=60.0, json_mode=True, max_tokens=6000):
        self.calls.append(system[:40])
        if not self.replies:
            raise PlanningUnavailableError("no scripted reply left")
        r = self.replies.pop(0)
        return r if isinstance(r, str) else json.dumps(r)


def engine_for(tmp_path, llm=None):
    settings = ArtistSettings(data_dir=tmp_path, drawings_dir=tmp_path / "drawings", db_path=tmp_path / "artist.db")
    engine = ArtistEngine(settings=settings)
    if llm is not None:
        engine.sketch_planner = SketchPlanner(llm)
    return engine


def scene_png(w=320, h=240) -> bytes:
    """A shaded ball and a dark box on a pale ground: edges and tones to sketch."""
    img = Image.new("RGB", (w, h), (235, 235, 230))
    d = ImageDraw.Draw(img)
    for r in range(70, 0, -2):
        v = int(40 + (70 - r) * 3)
        d.ellipse((100 - r, 120 - r, 100 + r, 120 + r), fill=(v, v // 2, v // 2))
    d.rectangle((190, 80, 290, 200), fill=(30, 40, 70))
    d.rectangle((210, 100, 240, 130), fill=(220, 220, 120))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


SKETCH_PROGRAM = {
    "title": "mug",
    "strokes": [
        {"brush": "ink", "width": 0.6, "points": [[30, 30], [28, 50], [30, 70], [50, 74], [70, 70], [72, 50], [70, 30]]},
        {"shape": "ellipse", "cx": 50, "cy": 30, "rx": 20, "ry": 5, "width": 0.5},
        {"brush": "pen", "width": 0.4, "points": [[72, 40], [84, 42], [86, 56], [72, 62]]},
    ],
    "strands": [{"guide": [[40, 10], [42, 20], [40, 26]], "count": 4, "spread": 0.8, "fan": 0.3, "width": 0.2}],
    "hatches": [{"region": [[52, 34], [70, 34], [70, 70], [52, 72]], "angle": 45, "spacing": 1.2, "width": 0.2}],
}


# ============================================================
# 1. VERSION SELECTION
# ============================================================


def test_default_is_1_0_and_both_versions_listed():
    assert DEFAULT_ALGORITHM == "1.0"
    assert set(ALGORITHMS) == {"1.0", "2.0"}
    assert resolve_algorithm(None) == "1.0"
    assert resolve_algorithm("") == "1.0"


@pytest.mark.parametrize("value,expected", [("1", "1.0"), ("1.0", "1.0"), ("2", "2.0"), ("2.0", "2.0"), ("V2", "2.0"), ("g-stroke-2.0", "2.0"), (" 2.0 ", "2.0")])
def test_algorithm_aliases(value, expected):
    assert resolve_algorithm(value) == expected


def test_unknown_algorithm_is_rejected():
    with pytest.raises(GeometricValidationError):
        resolve_algorithm("3.0")


def test_request_without_algorithm_means_1_0():
    r = DrawRequest(prompt="Draw a house")
    assert r.algorithm == "1.0" and r.brush is None


# ============================================================
# 2. BRUSHES, STROKES, STRANDS
# ============================================================


def test_all_brushes_exist_and_unknown_falls_back():
    assert {"pen", "fineliner", "ink", "pencil", "charcoal", "marker"} <= set(brush_names())
    assert get_brush("graphite").name == "pencil"
    assert get_brush("nonsense").name == "pencil"
    assert get_brush(None, "pen").name == "pen"


def test_smooth_path_passes_through_control_points_and_is_denser():
    pts = np.array([[0, 0, 0.5], [30, 20, 0.5], [60, 0, 0.5], [90, 25, 0.5]], dtype=float)
    out = smooth_path(pts, step=2.0)
    assert len(out) > 4 * 10
    for p in pts:
        assert np.min(np.hypot(out[:, 0] - p[0], out[:, 1] - p[1])) < 1.5
    assert np.all((out[:, 2] >= 0) & (out[:, 2] <= 1))


def _ribbon_width(parts):
    poly = parts[0].polygon
    return poly[:, 1].max() - poly[:, 1].min()


def test_ink_brush_tapers_and_swells_with_pressure():
    line = [(x, 50.0, 0.9) for x in range(0, 120, 10)]
    ink = stroke_parts(Stroke(line, "ink", width=4), seed=1, index=0)
    poly = ink[0].polygon

    def thickness(x_lo, x_hi):
        ys = poly[(poly[:, 0] >= x_lo) & (poly[:, 0] <= x_hi), 1]
        return ys.max() - ys.min()

    assert thickness(50, 70) > 2.5 * thickness(0, 4)  # the middle is much thicker than the tip

    # lighter pressure gives a thinner line
    soft = stroke_parts(Stroke([(x, 50.0, 0.1) for x in range(0, 120, 10)], "ink", width=4), seed=1, index=0)[0].polygon
    mid = soft[(soft[:, 0] >= 50) & (soft[:, 0] <= 70), 1]
    assert (mid.max() - mid.min()) < thickness(50, 70)


def test_pencil_makes_several_translucent_passes_pen_makes_one():
    line = [(0.0, 0.0, 0.7), (40.0, 8.0, 0.7), (80.0, 0.0, 0.7)]
    pencil = stroke_parts(Stroke(line, "pencil", width=2), 1, 0)
    pen = stroke_parts(Stroke(line, "pen", width=2), 1, 0)
    assert len(pencil) == 3 and all(p.opacity < 0.9 for p in pencil)
    assert len(pen) == 1 and pen[0].opacity > 0.95


def test_strokes_are_deterministic():
    line = [(0.0, 0.0, 0.7), (40.0, 8.0, 0.7), (80.0, 0.0, 0.7)]
    a = stroke_parts(Stroke(line, "charcoal", width=3), 5, 2)
    b = stroke_parts(Stroke(line, "charcoal", width=3), 5, 2)
    assert all(np.array_equal(x.polygon, y.polygon) for x, y in zip(a, b))


def test_single_point_stroke_is_a_dot():
    parts = stroke_parts(Stroke([(10.0, 10.0, 0.7)], "pen", width=3), 1, 0)
    assert len(parts) == 1 and len(parts[0].polygon) >= 6


def test_strand_is_a_bundle_of_parallel_strokes():
    guide = [(10, 50), (60, 40), (110, 50)]
    strokes = make_strand(guide, count=6, spread=3.0, fan=0.4, brush="pencil", width=1.0, group="hair", seed=3)
    assert len(strokes) == 6 and all(s.group == "hair" for s in strokes)
    mids = sorted(s.points[len(s.points) // 2][1] for s in strokes)
    assert mids[-1] - mids[0] > 8  # spread across the strand, not stacked on one line
    assert make_strand([(0, 0)], 3, 1.0) == []


# ============================================================
# 3. RENDERING
# ============================================================


def test_svg_and_png_render_the_same_sketch():
    sk = program_to_sketch(SKETCH_PROGRAM, 400, 300, "line", "pencil")
    svg = SketchSVG.render(sk, "a mug")
    root = ET.fromstring(svg)  # well-formed
    assert root.tag.endswith("svg") and root.get("viewBox") == "0 0 400 300"
    assert "<path" in svg and "fill-opacity" in svg
    img = SketchPNG.render(sk)
    assert img.size == (400, 300)
    a = np.asarray(img.convert("L"))
    assert a.min() < 120 and a.max() > 230  # ink on paper


def test_background_is_paper_not_pure_white():
    sk = program_to_sketch(SKETCH_PROGRAM, 200, 150)
    assert sk.paper != "#FFFFFF"


# ============================================================
# 4. IMAGE -> SKETCH
# ============================================================


def test_image_becomes_strokes_not_shapes():
    sk = ImageSketcher(480, 360, "shaded", "low", "pencil").sketch(scene_png())
    assert sk.meta["contour_strokes"] > 0 and sk.meta["hatch_strokes"] > 0
    assert all(s.brush == "pencil" for s in sk.strokes)
    assert {s.group.split("-")[0] for s in sk.strokes} == {"contour", "hatch"}


def test_shaded_has_more_hatching_than_line():
    line = ImageSketcher(480, 360, "line", "low").sketch(scene_png())
    shaded = ImageSketcher(480, 360, "shaded", "low").sketch(scene_png())
    assert shaded.meta["hatch_strokes"] > line.meta["hatch_strokes"]


def test_solid_is_an_alias_for_shaded():
    sk = ImageSketcher(480, 360, "solid", "low").sketch(scene_png())
    assert sk.meta["style"] == "shaded"


def test_color_style_colours_the_strokes():
    sk = ImageSketcher(480, 360, "color", "low").sketch(scene_png())
    assert len({s.color for s in sk.strokes}) > 5
    mono = ImageSketcher(480, 360, "line", "low").sketch(scene_png())
    assert len({s.color for s in mono.strokes}) == 1


def test_same_image_same_sketch():
    a = ImageSketcher(480, 360, "shaded", "low").sketch(scene_png())
    b = ImageSketcher(480, 360, "shaded", "low").sketch(scene_png())
    assert SketchSVG.render(a) == SketchSVG.render(b)


def test_strokes_stay_on_the_canvas():
    sk = ImageSketcher(480, 360, "shaded", "low").sketch(scene_png())
    xy = np.array([(x, y) for s in sk.strokes for x, y, _ in s.points])
    assert xy[:, 0].min() > -2 and xy[:, 0].max() < 482 and xy[:, 1].min() > -2 and xy[:, 1].max() < 362


def test_bad_image_is_a_clear_error():
    from artist.vision import ImageProblem

    with pytest.raises(ImageProblem):
        ImageSketcher().sketch(b"not an image")
    blank = io.BytesIO()
    Image.new("RGB", (200, 200), (255, 255, 255)).save(blank, format="PNG")
    with pytest.raises(ImageProblem):
        ImageSketcher().sketch(blank.getvalue())


# ============================================================
# 5. PROMPT -> SKETCH (scripted model)
# ============================================================


def test_program_validation_clamps_and_drops_bad_input():
    program = {
        "strokes": [
            {"points": [[1000, -900], [50, 50], ["x", "y"]]},   # clamped, junk point dropped
            {"points": [[1, 1]]},                               # one point: not a stroke
            "garbage",
            {"brush": "laser", "width": 999, "points": [[10, 10], [20, 20]]},  # unknown brush, huge width
        ],
        "hatches": [{"region": [[0, 0], [1, 1]]}],              # not a polygon
    }
    sk = program_to_sketch(program, 400, 300)
    assert len(sk.strokes) == 2
    assert all(-30 <= x <= 430 and -30 <= y <= 330 for s in sk.strokes for x, y, _ in s.points)
    assert sk.strokes[1].brush == "pencil" and sk.strokes[1].width <= 4.0 * 4
    with pytest.raises(GeometricValidationError):
        program_to_sketch({"strokes": []}, 400, 300)


def test_monochrome_ignores_model_colours_but_color_style_keeps_them():
    prog = {"strokes": [{"points": [[10, 10], [50, 50]], "color": "#FF0000", "brush": "pen"}]}
    assert program_to_sketch(prog, 400, 300, "line").strokes[0].color != "#FF0000"
    assert program_to_sketch(prog, 400, 300, "color").strokes[0].color == "#FF0000"


def test_planner_builds_a_sketch_from_a_program():
    out = SketchPlanner(ScriptedLLM(SKETCH_PROGRAM)).plan("Draw a mug", 400, 300, "line", "fast", "pen")
    sk = out["sketch"]
    groups = {s.group.split("-")[0] for s in sk.strokes}
    assert {"stroke", "strand", "hatch"} <= groups
    assert out["timeline"][0]["stage"] == "sketch" and out["model"] == "scripted"


def test_planner_retries_once_after_an_unusable_reply():
    llm = ScriptedLLM("not json at all", SKETCH_PROGRAM)
    out = SketchPlanner(llm).plan("Draw a mug", 400, 300, "line", "fast")
    assert len(llm.calls) == 2 and out["sketch"].stroke_count > 0


def test_planner_gives_up_clearly():
    with pytest.raises(PlanningUnavailableError):
        SketchPlanner(ScriptedLLM("nope", "still nope")).plan("Draw a mug", 400, 300, "line", "fast")


def test_best_quality_lets_the_model_revise_its_sketch():
    revised = dict(SKETCH_PROGRAM, title="better mug")
    llm = ScriptedLLM(SKETCH_PROGRAM, revised, {"ok": True})
    out = SketchPlanner(llm).plan("Draw a mug", 400, 300, "line", "best")
    assert out["sketch"].meta["title"] == "better mug" and len(llm.calls) == 3


# ============================================================
# 6. ENGINE + HTTP: 1.0 stays the default, 2.0 is opt-in
# ============================================================


def test_default_request_is_still_g_stroke_1_0(tmp_path, monkeypatch):
    monkeypatch.setenv("NEBULA_ARTIST_FALLBACK", "1")
    engine = engine_for(tmp_path)
    r = engine.draw(DrawRequest(prompt="Draw a wheel", width=400, height=300))
    assert r.algorithm == "1.0" and r.brush is None
    assert r.planner == "pattern-fallback"  # the untouched v1 pipeline


def test_v1_image_path_unchanged_by_default(tmp_path):
    engine = engine_for(tmp_path)
    r = engine.draw_from_image(scene_png(), width=480, height=360)
    assert r.algorithm == "1.0" and r.planner == "image-vision" and r.operations_count >= 0


def test_engine_draws_an_image_with_2_0(tmp_path):
    engine = engine_for(tmp_path)
    r = engine.draw_from_image(scene_png(), style="shaded", detail="low", width=480, height=360, algorithm="2.0", brush="charcoal", prompt="ball and box")
    assert r.status == "completed" and r.algorithm == "2.0" and r.brush == "charcoal"
    assert r.planner == "gstroke-image" and r.primitives_count > 0 and r.operations_count == 0
    assert (tmp_path / "drawings" / r.filename).is_file()
    assert Image.open(r.storage_path).size == (480, 360)
    ET.parse(str(tmp_path / "drawings" / r.filename.replace(".png", ".svg")))


def test_engine_draws_a_prompt_with_2_0(tmp_path):
    engine = engine_for(tmp_path, ScriptedLLM(SKETCH_PROGRAM))
    r = engine.draw(DrawRequest(prompt="Draw a mug", algorithm="2.0", quality="fast", width=400, height=300, format="svg"))
    assert r.status == "completed" and r.algorithm == "2.0" and r.format == "svg"
    assert r.planner == "gstroke-sketch" and r.model == "scripted" and r.brush == "pencil"


def test_2_0_prompt_without_a_model_fails_clearly(tmp_path):
    class Off:
        available = False

    engine = engine_for(tmp_path, Off())
    with pytest.raises(PlanningUnavailableError):
        engine.draw(DrawRequest(prompt="Draw a mug", algorithm="2.0"))


def test_2_0_background_job_reports_progress(tmp_path):
    engine = engine_for(tmp_path)
    r = engine.draw_from_image(scene_png(), detail="low", width=480, height=360, algorithm="2.0", run_async=True)
    assert r.status == "pending"
    end = time.time() + 30
    while time.time() < end:
        s = engine.status(r.drawing_id)
        if s.status in ("completed", "failed"):
            break
        time.sleep(0.1)
    assert s.status == "completed" and s.algorithm == "2.0"


@pytest.fixture
def api(client, tmp_path, monkeypatch):
    import server

    engine = engine_for(tmp_path, ScriptedLLM(SKETCH_PROGRAM))
    monkeypatch.setattr(server, "_artist_engine", engine)
    return client


def test_http_algorithms_listing(api):
    r = api.get("/artist/algorithms").json()
    assert r["default"] == "1.0"
    assert [a["version"] for a in r["algorithms"]] == ["1.0", "2.0"]
    assert "pencil" in [b["name"] for b in r["brushes_2_0"]]


def test_http_prompt_2_0_and_default(api, monkeypatch):
    r = api.post("/artist/draw", json={"prompt": "Draw a mug", "algorithm": "2.0", "quality": "fast", "width": 400, "height": 300})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["algorithm"] == "2.0" and body["status"] == "completed"
    assert api.get(body["image_url"]).headers["content-type"].startswith("image/png")
    assert api.get(body["svg_url"]).headers["content-type"].startswith("image/svg")
    # no algorithm given: the 1.0 path (no model configured in tests, so it reports that, as it always did)
    import server

    server._artist_engine.constructor_planner = ConstructionPlanner(ScriptedLLM("bad", "bad"))
    down = api.post("/artist/draw", json={"prompt": "Draw a house", "quality": "fast"})
    assert down.status_code == 503


def test_http_image_2_0(api):
    r = api.post(
        "/artist/draw-from-image",
        files={"image": ("scene.png", scene_png(), "image/png")},
        data={"algorithm": "2.0", "style": "color", "detail": "low", "brush": "pen", "width": "480", "height": "360"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["algorithm"] == "2.0" and body["brush"] == "pen" and body["style"] == "color"
    default = api.post("/artist/draw-from-image", files={"image": ("scene.png", scene_png(), "image/png")}, data={"detail": "low"})
    assert default.json()["algorithm"] == "1.0"


def test_http_unknown_algorithm_is_422(api):
    assert api.post("/artist/draw", json={"prompt": "Draw a mug", "algorithm": "9.9"}).status_code == 422
    r = api.post("/artist/draw-from-image", files={"image": ("s.png", scene_png(), "image/png")}, data={"algorithm": "9.9"})
    assert r.status_code == 422
