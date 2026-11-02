"""
LangChain tool exposing The Artist (Ability 0.01) to Nebula.
"""

from typing import Optional
from langchain_core.tools import tool

from .engine import ArtistEngine
from .models import DrawRequest
from .errors import ArtistError


@tool
def artist_draw(
    prompt: str,
    format: str = "png",
    width: int = 800,
    height: int = 600,
) -> str:
    """
    Construct a geometric drawing using Nebula's drawing engine (The Artist).
    Converts a drawing instruction into mathematically calculated circles, curves,
    sharp strokes, and boolean operations (union, intersection, subtraction).

    Args:
        prompt: What to draw (e.g. 'Draw a Ferrari', 'Draw a sports car', 'Draw a clock', 'Draw a mechanical gear').
        format: Output image format ('png' or 'svg'). Defaults to 'png'.
        width: Canvas width in pixels (100 to 4096). Defaults to 800.
        height: Canvas height in pixels (100 to 4096). Defaults to 600.

    Returns:
        Summary containing the drawing ID, saved image filename, local storage path,
        and HTTP endpoint URL to access the drawing.
    """
    try:
        request = DrawRequest(
            prompt=prompt,
            format=format,
            width=width,
            height=height,
        )
        engine = ArtistEngine()
        result = engine.draw(request)
        return (
            f"Successfully constructed geometric drawing with The Artist:\n"
            f"- Drawing ID: {result.drawing_id}\n"
            f"- Prompt: {result.prompt}\n"
            f"- Filename: {result.filename}\n"
            f"- Dimensions: {result.width}x{result.height}\n"
            f"- File Path: {result.storage_path}\n"
            f"- Image URL: {result.image_url}\n"
            f"- Geometric Primitives: {result.primitives_count}\n"
            f"- Boolean Operations: {result.operations_count}"
        )
    except ArtistError as exc:
        return f"The Artist could not complete drawing ({exc.code}): {exc.message}"
    except Exception as exc:
        return f"Drawing generation error: {exc}"
