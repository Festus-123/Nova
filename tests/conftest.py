"""
Test configuration.

Environment is set BEFORE any project module is imported so the memory and
history databases are created in a throwaway directory (never the real
nebula_*.db files) and no real model is called.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_DATA_DIR = tempfile.mkdtemp(prefix="nebula-tests-")
os.environ["NEBULA_DATA_DIR"] = _DATA_DIR
os.environ["NEBULA_API_KEY"] = "test-nebula-key"
if os.environ.get("NEBULA_LIVE_TESTS") == "1":
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder-key-not-real-000000")
os.environ["NEBULA_SORT_OCR"] = "0"

from document_sorter.ai import (  # noqa: E402
    GroupingPlan,
    PageAnalysis,
    SortingContext,
)
from document_sorter.config import SorterSettings  # noqa: E402
from document_sorter.engine import DocumentSortingEngine  # noqa: E402
from document_sorter.pages import Page  # noqa: E402

API_KEY = "test-nebula-key"

TACSFON_SCHEMA = ["material_type", "level", "school", "department", "course", "session", "semester"]
TACSFON_ALLOWED = {
    "material_type": ["Past Questions", "Manuals / Lecture Notes", "Workbooks", "Materials"],
    "levels": ["100", "200", "300", "400", "500"],
    "schools": [],
    "departments": [],
    "courses": [],
    "sessions": [],
    "semesters": [],
}


class ScriptedModel:
    """
    Test double for the AI stages. It does not "understand" anything: each
    test scripts the observations/plan the model would return, and the tests
    verify how the engine validates and turns them into results.
    """

    def __init__(
        self,
        plan: Optional[Callable[[Sequence[Page], SortingContext], dict]] = None,
        analysis: Optional[Callable[[Page], dict]] = None,
    ):
        self.calls = 0
        self.seen_pages: List[Page] = []
        self.seen_context: Optional[SortingContext] = None
        self._plan = plan
        self._analysis = analysis

    def analyze_pages(self, pages, context):
        self.calls += 1
        self.seen_pages.extend(pages)
        self.seen_context = context
        out = []
        for page in pages:
            data = {"page_index": page.index, "readable": True}
            if self._analysis:
                data.update(self._analysis(page))
            out.append(PageAnalysis.model_validate(data))
        return out

    def plan_groups(self, pages, analyses, context):
        self.calls += 1
        self.seen_context = context
        if self._plan is None:
            raise AssertionError("no plan scripted")
        return GroupingPlan.model_validate(self._plan(pages, context))


def group(indices, decision="classified", confidence="high", fields=None, **extra):
    return {
        "page_indices": list(indices),
        "decision": decision,
        "confidence": confidence,
        "relationship_evidence": extra.pop("relationship_evidence", ["scripted"]),
        "fields": {
            name: (value if isinstance(value, dict) else {"value": value, "source": "page_content", "evidence": "scripted"})
            for name, value in (fields or {}).items()
        },
        **extra,
    }


def tacsfon_fields(course, **overrides):
    values = {
        "material_type": "Past Questions",
        "level": "300",
        "school": "School of Science",
        "department": "Computer Science",
        "course": course,
        "session": "2024/2025",
        "semester": "First Semester",
    }
    values.update(overrides)
    return values


@pytest.fixture
def settings():
    return SorterSettings(ocr_enabled=False, request_timeout_seconds=60)


@pytest.fixture
def make_engine(settings):
    def factory(model, **overrides):
        merged = SorterSettings(**{**settings.__dict__, **overrides})
        return DocumentSortingEngine(model=model, settings=merged)

    return factory


@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient

    import server

    return TestClient(server.app)


@pytest.fixture
def use_model(monkeypatch):
    """Route the API's sorting engine to a scripted model."""
    import server

    def install(model, **overrides):
        merged = SorterSettings(**{**server.SORTER_SETTINGS.__dict__, "ocr_enabled": False, **overrides})
        monkeypatch.setattr(server, "_sorting_engine", DocumentSortingEngine(model=model, settings=merged))
        return model

    return install
