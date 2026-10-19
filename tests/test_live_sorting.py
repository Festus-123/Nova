"""
End-to-end tests against the REAL configured model (costs API credits).

Run with:  NEBULA_LIVE_TESTS=1 pytest tests/test_live_sorting.py -v
Filenames are deliberately meaningless so only content can drive the result.
"""

import io
import os

import pytest

from conftest import TACSFON_SCHEMA
from document_sorter import DocumentSortingEngine, SortingRequest, UploadInput
from document_sorter.config import SorterSettings
from fixtures import page_image

pytestmark = pytest.mark.skipif(
    os.environ.get("NEBULA_LIVE_TESTS") != "1", reason="set NEBULA_LIVE_TESTS=1 to call the real model"
)

HEADER = ["FEDERAL UNIVERSITY OF TECHNOLOGY", "SCHOOL OF ENGINEERING", "DEPARTMENT OF ELECTRICAL ENGINEERING"]


def _engine():
    from dotenv import load_dotenv

    load_dotenv(override=False)
    if os.environ.get("OPENAI_API_KEY", "").startswith("sk-test-placeholder"):
        pytest.skip("real OPENAI_API_KEY not available")
    return DocumentSortingEngine(settings=SorterSettings.from_env())


def _jpg(lines):
    return page_image(lines, fmt="JPEG")


def _request(**extra):
    return SortingRequest.model_validate(
        {
            "schema": ["course", "level", "material_type"],
            "allowed_values": {
                "material_type": ["Past Questions", "Manuals / Lecture Notes", "Workbooks", "Materials"],
                "levels": ["100", "200", "300", "400", "500"],
                "courses": ["EEE 306", "EEE 308", "CPS 401", "CSC 308"],
            },
            **extra,
        }
    )


def test_live_continuation_and_boundary():
    files = [
        ("IMG_001.jpg", _jpg(HEADER + ["EEE 306: SIGNALS AND SYSTEMS", "300 LEVEL EXAMINATION 2024/2025",
                                        "Answer ALL questions", "1. Define a linear time-invariant system.",
                                        "2. State the sampling theorem.", "3. Find the Fourier transform of",
                                        "   a rectangular pulse.", "4. Explain convolution."])),
        ("IMG_002.jpg", _jpg(["5. Determine the Laplace transform of e^(-2t)u(t).",
                              "6. Sketch the magnitude response of a first-order", "   low-pass filter.",
                              "", "--- END OF PAPER ---"])),
        ("IMG_003.jpg", _jpg(["FEDERAL UNIVERSITY OF TECHNOLOGY", "SCHOOL OF COMPUTING",
                              "CPS 401: ARTIFICIAL INTELLIGENCE", "400 LEVEL EXAMINATION 2024/2025",
                              "1. Define a rational agent.", "2. Compare BFS and DFS."])),
    ]
    uploads = [UploadInput(name, io.BytesIO(data), "image/jpeg") for name, data in files]
    result = _engine().sort(_request(), uploads)
    print(result.model_dump_json(indent=2, by_alias=True))

    all_groups = result.groups + result.isolated
    by_files = {tuple(s.original_filename for s in g.source_files): g for g in all_groups}
    assert ("IMG_001.jpg", "IMG_002.jpg") in by_files, "continuation page must join EEE 306"
    assert ("IMG_003.jpg",) in by_files, "CPS 401 must be a separate document"
    eee = by_files[("IMG_001.jpg", "IMG_002.jpg")]
    cps = by_files[("IMG_003.jpg",)]
    assert eee.status.value == "classified" and eee.classification["course"].replace(" ", "") == "EEE306"
    assert cps.status.value == "classified" and cps.classification["course"].replace(" ", "") == "CPS401"
    assert eee.classification["level"] == "300" and cps.classification["level"] == "400"


def test_live_orphan_page_is_isolated_then_admin_correction_classifies():
    orphan = _jpg(["7. Explain the difference between FIR and IIR filters.", "8. What is aliasing?"])
    engine = _engine()

    first = engine.sort(_request(), [UploadInput("scan_9382.jpg", io.BytesIO(orphan), "image/jpeg")])
    print(first.model_dump_json(indent=2, by_alias=True))
    assert not first.groups, "a page with no identifying information must not be classified"
    assert first.isolated[0].reason

    second = engine.sort(
        _request(
            additional_context={"course": "EEE 306", "level": "300", "source": "administrator"},
            previous_result=first.isolated[0].model_dump(mode="json"),
        ),
        [UploadInput("scan_9382.jpg", io.BytesIO(orphan), "image/jpeg")],
    )
    print(second.model_dump_json(indent=2, by_alias=True))
    grp = (second.groups + second.isolated)[0]
    assert grp.classification["course"] == "EEE 306"
    assert grp.field_sources["course"].value == "administrator"


def test_live_field_descriptions_for_unrestricted_field():
    page = _jpg(["DEPARTMENT OF CHEMISTRY", "CHM 201: ORGANIC CHEMISTRY I", "200 LEVEL EXAMINATION",
                 "1. Name the functional groups.", "2. Draw ethanol."])
    request = SortingRequest.model_validate(
        {
            "schema": ["course", "level"],
            "allowed_values": {"levels": ["100", "200", "300"]},
            "field_descriptions": {"course": "the course code only (letters and number), e.g. ABC 101"},
        }
    )
    result = _engine().sort(request, [UploadInput("IMG_7781.jpg", io.BytesIO(page), "image/jpeg")])
    print(result.model_dump_json(indent=2, by_alias=True))
    grp = (result.groups + result.isolated)[0]
    assert grp.classification["course"] == "CHM 201"
