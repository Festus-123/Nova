# Ability 0.01: The Artist, Geometric Drawing Engine (v2: constructive)

The Artist draws with **geometry only**: circles, ellipses, rectangles, triangles, polygons, strokes (lines), arcs and
Bézier curves, joined and cut with three boolean operations (**union, intersection, subtraction**). The result is
**vector art** (SVG, with a smooth PNG): clear at any size.

Nebula decides *what to build and where*. The Artist builds it exactly.

```
 prompt ──► Nebula (language model) ──► construction program ──► Constructor (exact vector booleans) ──► SVG + PNG
 image  ──► image processing (OpenCV) ──► construction program ─┘
```

## What changed from v1

v1 mostly replayed hand-written, coloured templates (a stored Ferrari, a clock) because its language-model path
had a 10 s timeout and used a model name the router does not serve, then silently fell back. v2:

- Nebula **constructs every drawing from scratch**: it interprets the object, places each circle / stroke / triangle
  on a grid, and chooses the union / intersect / subtract steps. No stored pictures.
- Booleans act on the **real vector geometry** (shapely), not on pixels: a union is one continuous outline, a
  subtraction is a true hole.
- Default look is **clean black line art**. No colour unless you ask for it (`style: "color"`).
- **Image input**: send a picture, get it back drawn from the same primitives.
- **Background jobs** with progress, because a language model takes 30 to 90 seconds.
- If the language model is unavailable the request **fails with a clear 503**. It does not draw a stored picture.
  (Developers can opt back into the old templates with `NEBULA_ARTIST_FALLBACK=1`.)

## Architecture

```
artist/
├── construct.py        Constructive engine: shapes + UNION / INTERSECT / SUBTRACT on real geometry (shapely),
│                       design grid, styles, auto-fit composition.      <- the heart
├── llm_planner.py      Nebula's drawing intelligence: interpret -> construct -> (review). Model client.
├── vision.py           Image input: OpenCV segmentation, then circle / rectangle / triangle / polygon fitting.
├── vector_render.py    SVG (exact paths) and anti-aliased PNG (3x supersampled).
├── engine.py           Orchestrator: jobs, storage, status, errors.
├── models.py / db.py / config.py / errors.py / grid.py / geometry.py / patterns.py / tool.py
└── nebula_planner.py, renderer.py, boolean.py   v1 stored-pattern engine, kept for the developer fallback
```

### The design grid

Programs are written on a resolution-independent grid: **x from 0 to 100** (left to right) and **y from 0 to
100 x height/width** (top to bottom, **y grows downward**). The same program draws at any canvas size. After building,
the engine centres the drawing and scales it to fit (8% margin) if the model placed it badly.

### A construction program

```json
{
  "title": "house",
  "steps": [
    {"id": "walls", "type": "rect",     "x": 25, "y": 40, "w": 50, "h": 32},
    {"id": "roof",  "type": "triangle", "points": [[18, 41], [50, 14], [82, 41]]},
    {"id": "door",  "type": "rect",     "x": 44, "y": 52, "w": 11, "h": 20},
    {"id": "win",   "type": "circle",   "cx": 50, "cy": 31, "r": 4.5},
    {"id": "body",  "op": "union",    "of": ["walls", "roof"]},
    {"id": "house", "op": "subtract", "from": "body", "remove": ["door", "win"]},
    {"id": "sill",  "type": "line", "points": [[30, 56], [39, 56]], "width": 0.4}
  ],
  "draw": [{"id": "house"}, {"id": "sill", "tone": "dark"}]
}
```

| Shape `type` | Fields |
|---|---|
| `circle` | `cx, cy, r` |
| `ellipse` | `cx, cy, rx, ry, angle` |
| `rect` | `x, y, w, h, angle, radius` (top-left corner; `radius` rounds the corners) |
| `triangle` | `points` (3) |
| `polygon` | `points` (3 or more) |
| `line` (stroke) | `points` (2 or more), `width`, `cap` |
| `arc` | `cx, cy, r, start, end, width` (degrees: 0 = right, 90 = down) |
| `bezier` | `points` (3 quadratic, 4 cubic), `width`, `closed` |
| `ring` | `cx, cy, r, r_inner` |

| Operation `op` | Meaning |
|---|---|
| `union` `of: [...]` | merge into one continuous shape |
| `intersect` `of: [...]` | keep only the overlap |
| `subtract` `from`, `remove: [...]` | cut shapes out (windows, doors, wheel arches) |

Shapes consumed by an operation are not drawn by themselves; the result is. `draw` lists what to show, bottom layer
first. `tone` is `light` (white, default), `mid` (grey) or `dark` (black). In `color` style an item may carry `color`.

### Styles

| `style` | Look |
|---|---|
| `line` (default) | black outlines on white; `tone: dark` parts are solid black |
| `solid` | bold silhouette, no outlines |
| `color` | flat fills from each item's `color`, thin dark outline |

### Quality (how many planning steps)

| `quality` | Steps | Typical time |
|---|---|---|
| `fast` | construct | 30 to 45 s |
| `balanced` (default) | interpret the object, then construct | 60 to 90 s |
| `best` | also renders its own drawing, looks at it, and corrects it (up to 2 rounds) | longer |

### Image input

1. Load and normalise (orientation, transparency, size).
2. Segment into tones or colours (k-means on colour; `line`/`solid` use fewer clusters), clean with median and
   morphology filters.
3. Trace each region. A region with holes is **outer shape, subtract its holes**.
4. **Fit geometry**: an exact circle if the region is round, a triangle or rectangle if it is one, else a simplified
   polygon. Specks are dropped. The background cluster becomes the paper.
5. The result is a construction program, drawn exactly like a prompt drawing.

`detail`: `low`, `medium` (default), `high` controls how many regions are kept and how closely edges are followed.
Limits: 12 MB, 60 million pixels; PNG, JPG, WEBP, GIF.

## HTTP API

### Draw from a prompt: `POST /artist/draw`
```json
{ "prompt": "Draw a house", "width": 800, "height": 600, "format": "png",
  "style": "line", "quality": "balanced", "async": false }
```
With `"async": true` it returns at once with `status: "pending"`; poll the status endpoint. Response:
```json
{ "drawing_id": "draw-…", "status": "completed", "filename": "draw_a_house_draw-….png",
  "image_url": "/artist/drawings/draw-…/file", "svg_url": "/artist/drawings/draw-…/file?format=svg",
  "primitives_count": 40, "operations_count": 4, "style": "line", "planner": "nebula-construct",
  "model": "claude-sonnet-5-5", "stage": "done",
  "timeline": [{"stage": "interpret", "ms": 19000, "ok": true}, {"stage": "construct", "ms": 32000, "ok": true}],
  "notes": [], "error": null }
```

### Draw from an image: `POST /artist/draw-from-image` (multipart)
Fields: `image` (file), `prompt` (optional name), `style`, `detail`, `width`, `height`, `format`, `async`.

### Progress: `GET /artist/drawings/{id}/status`
`status` is `pending`, `rendering`, `completed` or `failed`. `stage` says what it is doing; `error` says why it
failed. A drawing stuck "pending" for 10 minutes (the service restarted mid-job) is reported as failed.

### Files and records
- `GET /artist/drawings/{id}/file` (`?format=svg` or `png`): the drawing
- `GET /artist/drawings/{id}`, `GET /artist/drawings`: records
- `GET/POST /artist/patterns`: the stored pattern library (v1; still available)

### Errors
`422` bad input (empty prompt, unreadable image, size out of range) · `404` unknown drawing ·
`503` the language model is unavailable, so nothing was drawn.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | none | key for the model router. Without it, prompt drawing returns 503 |
| `NEBULA_LLM_BASE_URL` | `https://www.routeapi.ai/v1` | OpenAI-compatible endpoint |
| `NEBULA_ARTIST_MODEL` | none | the model the Artist should use, e.g. `claude-sonnet-5-5` (comma-separate several) |
| `NEBULA_MODEL` | none | used after `NEBULA_ARTIST_MODEL` **if set** |
| (built in) | `claude-sonnet-5-5`, `gpt-5.5` | tried next, in this order. The first that answers is remembered |
| `NEBULA_ARTIST_EFFORT` | `low` | reasoning effort sent to the model; `low` is about 40% faster. Empty disables it |
| `NEBULA_ARTIST_LLM` | on | `0` turns the language model off (tests) |
| `NEBULA_ARTIST_FALLBACK` | off | `1` re-enables the v1 stored-pattern drawings when no model is available (developers) |
| `NEBULA_DATA_DIR`, `NEBULA_DRAWINGS_DIR`, `NEBULA_ARTIST_DB` | project dir | where drawings and the Artist DB live |

> **Important:** the old default model name `gpt-5.4-nano` is **not served** by the router. It returns 503
> `model_not_found`. That is why v1 never used the language model. Check the router's model list
> (`GET {base_url}/models`) if a model stops answering.

## Render notes

- Install list: `directions/requirement.txt` now includes `shapely`, `opencv-python-headless` and `numpy`.
- Render's disk is **ephemeral**: drawings are lost on restart or deploy. Set `NEBULA_DATA_DIR` to a persistent disk
  if they must survive.
- Background jobs run in the web process (2 at a time). A restart during a job loses that job.
- A free instance sleeps when idle; the first request can take about a minute.

## Tests

`tests/test_artist.py` (v1, 15 tests) and `tests/test_artist_construct.py` (v2, 32 tests: booleans, rendering,
planner with a scripted model, image input, background jobs, HTTP). **No test calls a real model**: `conftest.py`
sets `NEBULA_ARTIST_LLM=0` and `NEBULA_ARTIST_FALLBACK=1`.
