# G-Stroke 2.0: the sketching algorithm

G-Stroke is the drawing algorithm of The Artist (Nebula Ability 0.01). It now has two versions. A client picks one per
request with the `algorithm` field.

| Version | What it draws | How |
|---|---|---|
| **1.0** (default) | Clean geometry: avatars, logos, diagrams, objects built from circles and shapes | Circles, polygons and strokes joined by union / intersection / subtraction |
| **2.0** | Sketches: pencil, ink and charcoal drawings with line weight and shading | Brushes, points, strokes and strands; contours, hatching, optional colour |

**1.0 is not changed by 2.0 and stays the default.** A request that does not send `algorithm` runs the same code path
it always did. Systems that depend on 1.0 need no change. 2.0 is opt-in, per request.

---

## Choosing a version

| Where | How |
|---|---|
| `POST /artist/draw` (JSON) | `"algorithm": "2.0"`, optional `"brush": "pencil"` |
| `POST /artist/draw-from-image` (multipart) | form fields `algorithm=2.0`, optional `brush=pencil` |
| `artist_draw` agent tool | `algorithm="2.0"`, `brush="pencil"` |
| Python | `DrawRequest(prompt=..., algorithm="2.0")`, `engine.draw_from_image(data, algorithm="2.0")` |
| Discovery | `GET /artist/algorithms` lists the versions, the default and the brushes |

Accepted spellings: `1`, `1.0`, `v1`, `g-stroke-1.0` and `2`, `2.0`, `v2`, `g-stroke-2.0`. Anything else returns `422`.
Every response carries `algorithm` (and `brush` for 2.0), so a client can see which version drew a picture. Drawings
made before this change report `"1.0"`.

---

## The model: points, strokes, strands

```
point    (x, y, pressure)      one sample of the pen's path
stroke   a path of points + a brush + a colour + a width
strand   a bundle of near-parallel strokes drawn together: hatching, hair, grass, folds, wood grain
sketch   an ordered list of strokes on a sheet of paper
```

**Stroke to ink.** A stroke's points are fitted with a Catmull-Rom spline, so even a few points give a flowing curve.
The curve becomes a variable-width ribbon: the width follows the brush's taper at the ends, the pressure of each point,
and a little unevenness. The ribbon is a filled vector path, so the SVG stays sharp at any size.

**Strand.** `count` strokes follow a guide path `spread` pixels apart; `fan` opens them out along the guide. Each member
gets its own length, offset and weight so the bundle reads as drawn by hand.

**Determinism.** A fixed seed drives all the "hand" randomness. The same input and settings give the same sketch.

### Brushes

| Brush | Look |
|---|---|
| `pencil` (default) | Graphite. Three soft, slightly offset, translucent passes: grainy, builds up where strokes cross |
| `pen` | Clean, even ink line with a light taper |
| `fineliner` | Thin, precise, technical |
| `ink` | Brush pen: strong taper, swells with pressure |
| `charcoal` | Broad, smudgy, four loose passes |
| `marker` | Bold, flat, blunt ends |

Unknown names fall back to `pencil`. A brush is a small dataclass (`artist/gstroke/brushes.py`): width scale, taper,
pressure response, unevenness, hand wobble, opacity and the list of passes. Adding a brush is one entry.

---

## Drawing a picture (image to sketch)

`POST /artist/draw-from-image` with `algorithm=2.0`. Classical image processing, no language model needed.

1. **Contours.** Two scales of edge detection find the structure (big outlines) and the detail (small marks). Edge
   pixels are walked into long pen paths, smoothed, and drawn as strokes whose weight follows the strength of the edge.
   Outlines are inked last, over the shading.
2. **Shading.** The picture's tones are cut into levels. Each darker level is hatched at a new angle (45, 135, 12, 98
   degrees), which gives cross-hatching in the shadows. Hatch lines are short hand-length strokes with ragged ends.
3. **Colour.** In `color` style each stroke takes the colour of the picture where it lies.

`style` for 2.0:

| `style` | Result |
|---|---|
| `line` (default) | Contours, with a single light hatch only in the very darkest areas |
| `shaded` (also `solid`) | Contours plus full cross-hatched tone |
| `color` | Shaded, with strokes tinted from the picture |

`detail`: `low`, `medium` (default), `high`. More detail means a larger working image, more strokes and finer hatching.
`high` on a portrait is about 12 s; `medium` is about 5 s. Limits are the same as 1.0 (12 MB, 60 megapixels).

## Drawing from a prompt

`POST /artist/draw` with `algorithm=2.0` and a prompt such as "Draw a Ferrari". Nebula's language model acts as the
artist's hand. It does not write SVG or pixels. It returns a **sketch program**: where each stroke goes, which brush,
how hard it presses, where strands and hatching go. The engine validates it (clamps coordinates, drops anything
unusable, caps the number of strokes) and draws it.

```json
{
  "title": "mug",
  "strokes": [
    {"brush": "ink", "width": 0.6, "points": [[30,30],[28,50],[30,70],[50,74],[70,70],[72,50],[70,30]]},
    {"shape": "ellipse", "cx": 50, "cy": 30, "rx": 20, "ry": 5, "start": 0, "end": 360, "width": 0.5}
  ],
  "strands": [{"guide": [[40,10],[42,20],[40,26]], "count": 4, "spread": 0.8, "fan": 0.3, "width": 0.2}],
  "hatches": [{"region": [[52,34],[70,34],[70,70],[52,72]], "angle": 45, "spacing": 1.2, "width": 0.2, "opacity": 0.7}]
}
```

Coordinates are the same design grid as 1.0: x runs 0 to 100, y runs 0 to the canvas height on the same scale, y grows
downward. Point format is `[x, y]` or `[x, y, pressure]`. A stroke may be a `shape` of `ellipse`, `circle` or `arc`.

`quality`: `fast` and `balanced` make one planning call (and one retry if the reply is unusable). `best` also renders
the sketch, shows it to the model and lets it correct the program, up to two rounds.

If the model is not available, a 2.0 prompt returns `503`, like 1.0. A 2.0 image sketch never needs the model.

---

## Request and response

```http
POST /artist/draw
{"prompt": "Draw a lion", "algorithm": "2.0", "brush": "ink", "style": "shaded", "quality": "balanced",
 "width": 900, "height": 700, "format": "png", "async": false}
```

```http
POST /artist/draw-from-image        (multipart)
image=<file>  algorithm=2.0  brush=pencil  style=shaded  detail=medium  width=900  height=700  format=png
```

The response is the usual `DrawResponse` with two new fields, `algorithm` and `brush`. For 2.0, `primitives_count` is
the number of strokes and `operations_count` is `0`. `planner` is `gstroke-image` or `gstroke-sketch`. Progress,
background jobs (`async`), files (`/artist/drawings/{id}/file`, `?format=svg`) and errors work exactly as in 1.0.

Output is a PNG (2x supersampled, then scaled down, on warm off-white paper with a faint tooth) and an SVG of the same ink.
SVG size grows with stroke count: a medium shaded photo is about 1.5 to 3 MB.

## Code map

| File | Role |
|---|---|
| `artist/gstroke/__init__.py` | version registry, `resolve_algorithm` |
| `artist/gstroke/brushes.py` | the brushes |
| `artist/gstroke/sketch.py` | `Stroke`, `Sketch`, strands, spline smoothing, ribbon geometry |
| `artist/gstroke/render.py` | `SketchSVG`, `SketchPNG` |
| `artist/gstroke/imaging.py` | `ImageSketcher`: contours, hatching, colour |
| `artist/gstroke/planner.py` | `SketchPlanner`: the sketch program, validation and the review loop |
| `artist/engine.py` | `_run_gstroke2`: a separate pipeline; the 1.0 pipeline is untouched |

2.0 needs only `opencv-python-headless`, `numpy` and `pillow`, which are already requirements. It does not use shapely.

## Limits and notes

- The image sketcher draws edges and tone. It does not understand the subject, so a very busy photo (foliage, crowds)
  gives a busy sketch. Use `detail=low` for a calmer one.
- A very dark picture hatches heavily everywhere, because tone is measured against the picture's own range.
- Prompt sketches are only as good as the model's drawing. `quality=best` helps.
- Brush grain comes from stacked translucent passes, so the SVG and PNG match, and no SVG filters are used.

## Tests

`tests/test_gstroke2.py` (42 tests): version selection and aliases, 1.0 as the default (prompt and image), brush
behaviour, spline and taper maths, strands, SVG/PNG rendering, image sketching (styles, determinism, bounds, bad input),
program validation, the planner with a scripted model (retry, give-up, review), the engine, background jobs and HTTP.
No test calls a real model.
