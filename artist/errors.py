"""
Custom exceptions for The Artist drawing engine.
"""

from typing import Optional, Dict, Any


class ArtistError(Exception):
    """Base exception for The Artist drawing engine."""

    def __init__(
        self,
        message: str,
        code: str = "artist_error",
        http_status: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.http_status = http_status
        self.details = details or {}


class GeometricValidationError(ArtistError):
    """Raised when geometric properties or constraints are invalid."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            code="geometric_validation_error",
            http_status=422,
            details=details,
        )


class PatternNotFoundError(ArtistError):
    """Raised when a requested geometric pattern cannot be found."""

    def __init__(self, pattern_id_or_name: str):
        super().__init__(
            message=f"Pattern not found: {pattern_id_or_name}",
            code="pattern_not_found",
            http_status=404,
            details={"pattern": pattern_id_or_name},
        )


class DrawingNotFoundError(ArtistError):
    """Raised when a requested drawing cannot be found."""

    def __init__(self, drawing_id: str):
        super().__init__(
            message=f"Drawing not found: {drawing_id}",
            code="drawing_not_found",
            http_status=404,
            details={"drawing_id": drawing_id},
        )


class RenderingError(ArtistError):
    """Raised when rendering to SVG or PNG fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            code="rendering_error",
            http_status=500,
            details=details,
        )


class PlanningError(ArtistError):
    """Raised when Nebula Intelligence fails to plan a drawing."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            code="planning_error",
            http_status=422,
            details=details,
        )
