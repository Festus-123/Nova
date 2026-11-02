"""
Configuration for The Artist (Ability 0.01).
"""

import os
from pathlib import Path
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("NEBULA_DATA_DIR") or BASE_DIR)


class ArtistSettings(BaseModel):
    data_dir: Path = Field(default_factory=lambda: DATA_DIR)
    drawings_dir: Path = Field(
        default_factory=lambda: Path(os.environ.get("NEBULA_DRAWINGS_DIR") or (DATA_DIR / "drawings"))
    )
    db_path: Path = Field(
        default_factory=lambda: Path(os.environ.get("NEBULA_ARTIST_DB") or (DATA_DIR / "nebula_artist.db"))
    )
    default_width: int = Field(default=800, ge=100, le=4096)
    default_height: int = Field(default=600, ge=100, le=4096)
    default_grid_size: int = Field(default=20, ge=4, le=200)
    default_format: str = Field(default="png")
    max_primitives: int = Field(default=500)

    @classmethod
    def from_env(cls) -> "ArtistSettings":
        current_data_dir = Path(os.environ.get("NEBULA_DATA_DIR") or BASE_DIR)
        current_drawings_dir = Path(
            os.environ.get("NEBULA_DRAWINGS_DIR") or (current_data_dir / "drawings")
        )
        current_db_path = Path(
            os.environ.get("NEBULA_ARTIST_DB") or (current_data_dir / "nebula_artist.db")
        )
        return cls(
            data_dir=current_data_dir,
            drawings_dir=current_drawings_dir,
            db_path=current_db_path,
        )


ARTIST_SETTINGS = ArtistSettings.from_env()
