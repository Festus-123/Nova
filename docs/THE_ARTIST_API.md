# Ability 0.01: The Artist — Geometric Drawing Engine

The Artist is a geometric drawing capability for **Nebula** that converts natural language drawing instructions into mathematically calculated geometric compositions using dynamic circles, curves, sharp strokes, and Boolean operations.

---

## 1. Core Architecture

The Artist is structured as a modular package (`artist/`) integrated into Nebula without impacting existing services:

```
artist/
├── __init__.py           # Package exports
├── config.py             # Settings, directories, environment variables
├── errors.py             # Custom Artist error hierarchy
├── models.py             # Pydantic schemas (primitives, operations, plans, API schemas)
├── geometry.py           # Mathematical engine (circles, tangents, curves, polygons, transforms)
├── grid.py               # Canvas Grid & spatial indexing
├── boolean.py            # Boolean operations (UNION, INTERSECTION, SUBTRACTION)
├── db.py                 # Artist DB (SQLite pattern storage & drawing records)
├── patterns.py           # Reusable parameterized pattern engine
├── renderer.py           # SVG vector and PNG raster renderers
├── nebula_planner.py     # Nebula Intelligence drawing decomposition & planning
├── engine.py             # Orchestrator for Ability 0.01
└── tool.py               # LangChain @tool for the Nebula agent
```

---

## 2. Fundamental Geometric Principles

1. **Boolean Operations**:
   - **UNION**: Merges two or more geometric shapes into a continuous form.
   - **INTERSECTION**: Retains only the overlapping region shared by shapes.
   - **SUBTRACTION**: Cuts out geometric regions from a target shape (e.g. cutting wheel wells out of a chassis, or an axle hole out of a gear).

2. **Dynamically Calculated Circles**:
   - Dimensions are never hardcoded or fixed.
   - Dynamic calculations include: radius, diameter, circumference, center coordinates, overlap distance, intersection points, and external/internal tangency.

3. **Extensible Primitives**:
   - Circles and ellipses
   - Sharp straight lines and strokes
   - Quadratic and Cubic Bézier curves
   - Polygons and custom paths
   - Multi-layer compound shapes

---

## 3. Pattern Memory (Artist DB)

Persistent SQLite database (`nebula_artist.db`) storing reusable, parameterized geometric patterns:
- **Parameterized**: Patterns can be scaled, translated, rotated, and recolored without redrawing from scratch.
- **Versioning**: Updating a pattern automatically increments its version number.
- **Starter Patterns**: Includes `wheel`, `sports_chassis`, and `gear`.

---

## 4. HTTP API Endpoints

### 4.1 Construct a Drawing
- **`POST /artist/draw`**
  ```json
  {
    "prompt": "Draw a Ferrari",
    "width": 800,
    "height": 600,
    "format": "png"
  }
  ```
  **Response (200 OK):**
  ```json
  {
    "drawing_id": "draw-d2d22d5cfee3",
    "prompt": "Draw a Ferrari",
    "filename": "draw_a_ferrari_draw-d2d.png",
    "format": "png",
    "width": 800,
    "height": 600,
    "image_url": "/artist/drawings/draw-d2d22d5cfee3/file",
    "svg_url": "/artist/drawings/draw-d2d22d5cfee3/file?format=svg",
    "status": "completed",
    "primitives_count": 14,
    "operations_count": 3,
    "storage_path": "...",
    "created_at": "2026-11-02T15:42:25.000000Z"
  }
  ```

### 4.2 Retrieve Rendered Image
- **`GET /artist/drawings/{drawing_id}/file`**
  - Query parameters: `format=png` (default) or `format=svg`
  - Returns binary `image/png` or vector `image/svg+xml`.

### 4.3 Drawing Metadata
- **`GET /artist/drawings/{drawing_id}`**
  - Returns drawing details and metadata.
- **`GET /artist/drawings`**
  - Lists completed drawings.

### 4.4 Pattern Management
- **`GET /artist/patterns`**: List all saved patterns.
- **`GET /artist/patterns/{pattern_id_or_name}`**: Retrieve pattern definition.
- **`POST /artist/patterns`**: Create or update a parameterized pattern.

---

## 5. Agent Integration (Nebula LangChain Tool)

The `@tool artist_draw` is registered in `main.py` under Nebula's active tools list:
```python
tools = [
    ...,
    sort_documents,
    artist_draw,
]
```
Users chatting with Nebula can ask naturally (e.g. *"Draw a Ferrari"* or *"Create a clock with geometric hands"*), and Nebula will execute `artist_draw` to generate the file and deliver the HTTP URL.
