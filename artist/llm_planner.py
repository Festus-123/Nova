"""
Nebula's drawing intelligence for The Artist (Ability 0.01).

Nebula never writes pixels or SVG. Given a request ("Draw a house") it:

  1. INTERPRETS the object: its parts, its proportions, the clearest view to draw it from, and where each part
     sits inside the canvas (the "decomposition").
  2. CONSTRUCTS it: decides, part by part, which circle, stroke, triangle, rectangle, polygon, arc or curve goes at
     which position on the grid, and which UNION / INTERSECT / SUBTRACT operations join and cut them. This is the
     construction program that artist.construct turns into exact vector geometry.
  3. REVIEWS the result (optional, "best" quality): the drawing is rendered, shown back to the model, and the model
     moves, resizes, adds or removes parts until it looks right.

If the language model is not available, planning fails with a clear error. It does NOT fall back to stored pictures.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .construct import DESIGN_WIDTH, ConstructionError, Constructor, ResolvedDrawing, normalize_program
from .errors import ArtistError, GeometricValidationError
from .vector_render import VectorPNG

logger = logging.getLogger("nebula.artist.llm")

# Tried in this order when no model is configured. The router's own default ("gpt-5.4-nano") is not served by it.
DEFAULT_MODELS = ["claude-sonnet-5-5", "gpt-5.5"]


class PlanningUnavailableError(ArtistError):
    """The language model could not be reached or produced nothing usable."""

    def __init__(self, message: str):
        super().__init__(message, code="planning_unavailable", http_status=503)


# --------------------------------------------------------------------------------------------------------------
# Talking to the model
# --------------------------------------------------------------------------------------------------------------


class LLMClient:
    """A thin client over an OpenAI-compatible endpoint that finds a model that works and remembers it."""

    _working: Optional[str] = None

    def __init__(self) -> None:
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.base_url = os.environ.get("NEBULA_LLM_BASE_URL", "https://www.routeapi.ai/v1")
        configured = [m.strip() for m in (os.environ.get("NEBULA_ARTIST_MODEL", "") + "," + os.environ.get("NEBULA_MODEL", "")).split(",") if m.strip()]
        self.models = list(dict.fromkeys(configured + DEFAULT_MODELS))
        self.last_model: Optional[str] = None

    @property
    def available(self) -> bool:
        return bool(self.api_key) and os.environ.get("NEBULA_ARTIST_LLM") != "0"

    def complete(self, system: str, user: Any, timeout: float = 60.0, json_mode: bool = True, max_tokens: int = 6000) -> str:
        if not self.available:
            raise PlanningUnavailableError("Nebula's language model is not configured (OPENAI_API_KEY is missing or NEBULA_ARTIST_LLM=0).")
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=timeout, max_retries=0)
        order = ([LLMClient._working] if LLMClient._working else []) + [m for m in self.models if m != LLMClient._working]
        errors: List[str] = []
        for model in order:
            kwargs: Dict[str, Any] = dict(model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], max_completion_tokens=max_tokens)
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            effort = os.environ.get("NEBULA_ARTIST_EFFORT", "low").strip()
            if effort:
                # Geometry planning does not need long hidden reasoning: "low" is about 40% faster.
                kwargs["extra_body"] = {"reasoning_effort": effort}
            try:
                try:
                    r = client.chat.completions.create(**kwargs)
                except Exception as e:
                    # Some models reject JSON mode, the effort setting or the token argument: retry once, plainly.
                    msg = str(e).lower()
                    if "model_not_found" in msg or "no valid upstream" in msg or "404" in msg:
                        raise
                    kwargs.pop("response_format", None)
                    kwargs.pop("extra_body", None)
                    if "max_completion_tokens" in msg or "max_tokens" in msg:
                        kwargs.pop("max_completion_tokens", None)
                        kwargs["max_tokens"] = max_tokens
                    r = client.chat.completions.create(**kwargs)
                text = (r.choices[0].message.content or "").strip()
                if not text:
                    raise ValueError("empty reply")
                LLMClient._working = model
                self.last_model = model
                return text
            except Exception as e:
                errors.append(f"{model}: {type(e).__name__}: {str(e)[:120]}")
                logger.warning("Artist model %s failed: %s", model, str(e)[:200])
                if LLMClient._working == model:
                    LLMClient._working = None
                continue
        raise PlanningUnavailableError("None of Nebula's drawing models answered. " + " | ".join(errors[:3]))


def parse_json(text: str) -> Any:
    """Reads the JSON object out of a reply, tolerating code fences, chatter and trailing commas."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.IGNORECASE)
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in the reply")
    body = t[start : end + 1]
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return json.loads(re.sub(r",\s*([}\]])", r"\1", body))


# --------------------------------------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------------------------------------


def _grid(width: int, height: int) -> Tuple[float, float]:
    return DESIGN_WIDTH, round(DESIGN_WIDTH * height / width, 2)


def _style_note(style: str) -> str:
    return {
        "line": "STYLE line: clean black line art on white. Shapes are outlined; give a part tone 'dark' only for solid black parts (tyres, glass, shadows, pupils).",
        "solid": "STYLE solid: a bold silhouette. Shapes are filled; use 'light' tone for parts that must read as white cut-outs and 'mid' for grey.",
        "color": "STYLE color: flat colours. Give every item in 'draw' a \"color\" (#RRGGBB) that suits the object; keep a small consistent palette.",
    }.get(style, "")


SYSTEM_DECOMPOSE = """You are Nebula's drawing intelligence. You will draw an object using ONLY geometry: circles, ellipses, rectangles, triangles, polygons, strokes (lines), arcs and Bezier curves, combined with the boolean operations UNION, INTERSECT and SUBTRACT. First you analyse the object.

The canvas is a design grid: x runs 0 to {W} (left to right) and y runs 0 to {H} (top to bottom, y grows DOWNWARD).

Return ONLY a JSON object:
{{
  "object": "what is being drawn",
  "view": "the single clearest view to draw it from (e.g. side elevation, front elevation, top view)",
  "bbox": [x0, y0, x1, y1],            // the whole drawing; fill about 75% of the canvas, centred
  "proportions": "key real-world proportions that must be respected (e.g. wheel diameter is about 0.55 x body height)",
  "components": [
    {{"name": "...", "bbox": [x0, y0, x1, y1], "role": "body | detail | cutout | dark | stroke", "geometry": "which primitives build it, e.g. 2 circles + 1 triangle, or a cubic bezier for the roofline", "notes": "how it connects to other parts"}}
  ],
  "operations": ["which parts are UNIONed into one silhouette", "which are SUBTRACTed from what (windows, doors, arches, holes)", "which are INTERSECTed to trim a shape"],
  "draw_order": ["bottom layer first ..."]
}}
Be concrete and complete: list every part a viewer needs to recognise the object, plus its finer details. Use real proportions."""

SYSTEM_CONSTRUCT = """You are Nebula's drawing intelligence. You build drawings out of GEOMETRY ONLY. You never write SVG, pixels or paths. You decide which circle, stroke, triangle or other primitive goes where, and which UNION / INTERSECT / SUBTRACT operations join and cut them.

CANVAS: a design grid. x runs 0 to {W} (left to right). y runs 0 to {H} (top to bottom: y grows DOWNWARD). Origin is the top-left corner. Keep everything inside the grid.

SHAPE STEPS (every step has a unique "id"):
  {{"id":"a","type":"circle","cx":50,"cy":40,"r":10}}
  {{"id":"b","type":"ellipse","cx":50,"cy":40,"rx":12,"ry":6,"angle":0}}
  {{"id":"c","type":"rect","x":20,"y":30,"w":40,"h":20,"angle":0,"radius":0}}     x,y is the top-left corner; radius rounds the corners
  {{"id":"d","type":"triangle","points":[[x,y],[x,y],[x,y]]}}
  {{"id":"e","type":"polygon","points":[[x,y],[x,y],[x,y],[x,y]]}}                 3 or more points; use for sloped/angular parts
  {{"id":"f","type":"line","points":[[x,y],[x,y]],"width":0.4,"cap":"round"}}      a STROKE; 2+ points makes a polyline. width is in grid units
  {{"id":"g","type":"arc","cx":50,"cy":50,"r":20,"start":200,"end":340,"width":0.4}}  degrees: 0 = right, 90 = DOWN, 180 = left, 270 = UP
  {{"id":"h","type":"bezier","points":[[x,y],[x,y],[x,y]],"width":0.4}}            3 points = quadratic, 4 = cubic; add "closed":true to get a filled region
  {{"id":"i","type":"ring","cx":50,"cy":50,"r":10,"r_inner":6}}

OPERATION STEPS:
  {{"id":"x","op":"union","of":["a","b","c"]}}                       merges shapes into ONE continuous outline
  {{"id":"y","op":"intersect","of":["a","b"]}}                       keeps only the region both share (use it to trim a shape)
  {{"id":"z","op":"subtract","from":"a","remove":["b","c"]}}         cuts shapes out of a: windows, doors, wheel arches, holes
Shapes consumed by an operation are not drawn by themselves: the result is. Later steps may use earlier results.

OUTPUT (JSON only, no commentary):
{{"title":"...","steps":[ ... ],"draw":[{{"id":"z"}},{{"id":"f","tone":"dark"}}]}}
"draw" lists what to show, BOTTOM layer first: later items are painted over earlier ones. "tone" is "light" (white, the default), "mid" (grey) or "dark" (black). {style_note}

HOW TO WORK (this is what makes a good drawing):
 1. Use the decomposition you are given: place every component where its bbox says, using real proportions.
 2. Build each silhouette from simple shapes that OVERLAP a little where they join, then UNION them so the outline is one continuous shape with no seams.
 3. Cut openings with SUBTRACT, and make sure each subtracted shape really overlaps what it cuts. Use INTERSECT to trim (a circle clipped by a rectangle makes a dome or half circle).
 4. Use circles for wheels, hubs, knobs, windows, eyes, suns; strokes for panel lines, handles, spokes, edges and fine detail; triangles and polygons for roofs, fins and angular parts; curves for smooth contours.
 5. Detail makes a drawing recognisable: aim for 20 to 70 steps for a typical object. Fine strokes are 0.25 to 0.5 wide, bold ones 0.8 to 2.
 6. Symmetric objects must be exactly symmetric about their centre line. Parts of the same kind must be the same size.
 7. Never leave a part floating by accident: everything must connect to, or sit sensibly against, the rest.
Return only the JSON object."""

SYSTEM_REVIEW = """You are Nebula's drawing intelligence reviewing your own work. You will see the drawing that your construction program produced, as an image (or, if no image, a coarse ASCII map where # is ink), together with the program.

Compare it with what was asked. Look for: parts in the wrong place or size, parts that float or do not connect, missing parts or details, openings that were not cut, a drawing that is off-centre, tiny or cropped, and anything that makes the object hard to recognise.

If it already looks right, reply {{"ok": true}}.
Otherwise reply with the COMPLETE corrected construction program (same format as before: {{"title","steps","draw"}}), fixing the problems. Keep what is good; move, resize, add or remove parts as needed. The grid is x 0..{W}, y 0..{H}, y grows DOWNWARD. {style_note}
Reply with JSON only."""


# --------------------------------------------------------------------------------------------------------------
# The planner
# --------------------------------------------------------------------------------------------------------------

ProgressFn = Callable[[str], None]


class ConstructionPlanner:
    """Nebula's drawing intelligence: turns a request into a construction program and the geometry built from it."""

    def __init__(self, client: Optional[LLMClient] = None):
        self.llm = client or LLMClient()

    @property
    def available(self) -> bool:
        return self.llm.available

    def plan(
        self,
        prompt: str,
        width: int = 800,
        height: int = 600,
        style: str = "line",
        quality: str = "balanced",
        budget_s: float = 110.0,
        on_stage: Optional[ProgressFn] = None,
    ) -> Dict[str, Any]:
        """
        Returns {"program", "drawing" (ResolvedDrawing), "decomposition", "timeline", "model"}.
        quality: fast = construct only; balanced = interpret, then construct; best = also review and refine.
        """
        t0 = time.time()
        W, H = _grid(width, height)
        timeline: List[Dict[str, Any]] = []
        left = lambda: max(8.0, budget_s - (time.time() - t0))  # noqa: E731

        labels = {"interpret": "interpreting the object", "construct": "placing circles, strokes and shapes", "construct (retry)": "fixing a problem and trying again"}

        def stage(name: str, fn: Callable[[], Any]) -> Any:
            if on_stage:
                try:
                    on_stage(labels.get(name, "reviewing the drawing" if name.startswith("review") else name))
                except Exception:  # progress reporting must never stop a drawing
                    pass
            s = time.time()
            try:
                out = fn()
                timeline.append({"stage": name, "ms": int((time.time() - s) * 1000), "ok": True})
                return out
            except PlanningUnavailableError:
                raise
            except Exception as e:
                timeline.append({"stage": name, "ms": int((time.time() - s) * 1000), "ok": False, "note": str(e)[:160]})
                raise

        decomposition: Optional[Dict[str, Any]] = None
        if quality in ("balanced", "best"):
            try:
                decomposition = stage("interpret", lambda: parse_json(self.llm.complete(SYSTEM_DECOMPOSE.format(W=W, H=H), f"Object to draw: {prompt}", timeout=min(60, left()))))
            except PlanningUnavailableError:
                raise
            except Exception as e:
                logger.info("Interpretation step skipped: %s", e)

        user = f"Draw: {prompt}\n"
        if decomposition:
            user += "\nYour analysis of the object (follow it):\n" + json.dumps(decomposition, indent=1)[:6000]

        program, drawing = self._construct(user, W, H, style, width, height, left, stage, timeline)

        if quality == "best":
            for round_no in range(2):
                if left() < 35:
                    break
                try:
                    revised = stage(f"review {round_no + 1}", lambda: self._review(prompt, program, drawing, W, H, style, left()))
                except PlanningUnavailableError:
                    break
                except Exception:
                    break
                if revised is None:  # the model is satisfied
                    break
                try:
                    new_drawing = Constructor(width, height, style).run(revised)
                    if new_drawing.items:
                        program, drawing = revised, new_drawing
                except ConstructionError:
                    break

        return {"program": program, "drawing": drawing, "decomposition": decomposition, "timeline": timeline, "model": self.llm.last_model}

    # -- steps ---------------------------------------------------------------------------------------------

    def _construct(self, user: str, W: float, H: float, style: str, width: int, height: int, left: Callable[[], float], stage: Callable, timeline: List[Dict[str, Any]]):
        system = SYSTEM_CONSTRUCT.format(W=W, H=H, style_note=_style_note(style))
        last_error = ""
        for attempt in range(2):
            ask = user if not last_error else user + f"\n\nYour previous program could not be used: {last_error}\nReturn a corrected, complete program."
            raw = stage("construct" if attempt == 0 else "construct (retry)", lambda: self.llm.complete(system, ask, timeout=min(90, left()), max_tokens=9000))
            try:
                program = normalize_program(parse_json(raw))
                drawing = Constructor(width, height, style).run(program)
                if len(drawing.items) >= 1:
                    return program, drawing
                last_error = "it produced no drawable shapes"
            except (ValueError, ConstructionError, GeometricValidationError) as e:
                last_error = str(e)[:300]
            if left() < 20:
                break
        raise PlanningUnavailableError(f"Nebula could not build a usable drawing ({last_error or 'no usable program'}).")

    def _review(self, prompt: str, program: Dict[str, Any], drawing: ResolvedDrawing, W: float, H: float, style: str, remaining: float) -> Optional[Dict[str, Any]]:
        system = SYSTEM_REVIEW.format(W=W, H=H, style_note=_style_note(style))
        text = f"Requested drawing: {prompt}\n\nCurrent construction program:\n{json.dumps({k: program[k] for k in ('title', 'steps', 'draw')}, separators=(',', ':'))[:14000]}\n"
        if drawing.warnings:
            text += "\nProblems the engine noticed: " + "; ".join(drawing.warnings[:6]) + "\n"
        content: Any
        try:
            content = [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": "data:image/png;base64," + _png_b64(drawing)}}]
            reply = self.llm.complete(system, content, timeout=min(80, remaining))
        except PlanningUnavailableError:
            raise
        except Exception:
            # The model may not read images: show it an ASCII map instead.
            reply = self.llm.complete(system, text + "\nASCII map of the drawing (# = ink):\n" + _ascii(drawing), timeout=min(80, remaining))
        data = parse_json(reply)
        if isinstance(data, dict) and data.get("ok") is True and "steps" not in data:
            return None
        return normalize_program(data)


def _png_b64(drawing: ResolvedDrawing, side: int = 640) -> str:
    img = VectorPNG.render(drawing)
    img.thumbnail((side, side))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _ascii(drawing: ResolvedDrawing, cols: int = 72) -> str:
    img = VectorPNG.render(drawing).convert("L")
    rows = max(8, int(cols * drawing.height / drawing.width * 0.5))
    img = img.resize((cols, rows))
    px = img.load()
    return "\n".join("".join("#" if px[x, y] < 140 else ("+" if px[x, y] < 215 else ".") for x in range(cols)) for y in range(rows))
