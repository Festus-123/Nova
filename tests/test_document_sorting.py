import io
import json
from pathlib import Path

import pytest

from conftest import (
    API_KEY,
    TACSFON_ALLOWED,
    TACSFON_SCHEMA,
    ScriptedModel,
    group,
    tacsfon_fields,
)
from document_sorter import DocumentSortingResponse, SortingRequest, UploadInput
from document_sorter.errors import NoFilesError, RequestTooLargeError
from fixtures import docx_file, image_pdf, page_image, text_pdf

AUTH = {"Authorization": f"Bearer {API_KEY}"}


def uploads(*items):
    return [UploadInput(filename=name, stream=io.BytesIO(data), content_type=ctype) for name, data, ctype in items]


def tacsfon_request(**extra):
    return SortingRequest.model_validate(
        {"schema": TACSFON_SCHEMA, "allowed_values": TACSFON_ALLOWED, **extra}
    )


def one_group_plan(course="CSC 308", **field_overrides):
    return lambda pages, ctx: {
        "groups": [group([p.index for p in pages], fields=tacsfon_fields(course, **field_overrides))]
    }


def post(client, files, schema=TACSFON_SCHEMA, allowed=TACSFON_ALLOWED, headers=AUTH, **form):
    data = {"schema": json.dumps(schema) if not isinstance(schema, str) else schema}
    if allowed is not None:
        data["allowed_values"] = json.dumps(allowed)
    data.update(form)
    return client.post("/sort-documents", files=files, data=data, headers=headers)


# ============================================================
# 1-5  SUPPORTED FILE TYPES
# ============================================================

def test_pdf_upload_extracts_text_layer_and_metadata(make_engine):
    model = ScriptedModel(plan=one_group_plan())
    pdf = text_pdf(
        [["FEDERAL UNIVERSITY", "CSC 308 - Operating Systems", "1. Define a process."], ["2. Explain paging."]],
        title="CSC 308 exam",
    )
    result = make_engine(model).sort(tacsfon_request(), uploads(("scan_9382.pdf", pdf, "application/pdf")))

    assert result.status == "success"
    assert [p.text_source for p in model.seen_pages] == ["text_layer", "text_layer"]
    assert "CSC 308" in model.seen_pages[0].text
    assert model.seen_pages[0].file_metadata.get("title") == "CSC 308 exam"
    assert model.seen_pages[0].signals.question_numbers == [1]
    grp = result.groups[0]
    assert grp.source_files[0].page_numbers == [1, 2]
    assert grp.proposed_filename.endswith(".pdf")


def test_scanned_pdf_pages_are_rendered_for_vision(make_engine):
    model = ScriptedModel(plan=one_group_plan())
    pdf = image_pdf([["EEE 306", "Question 1"], ["Question 5"]])
    result = make_engine(model).sort(tacsfon_request(), uploads(("scan.pdf", pdf, "application/pdf")))
    assert result.summary.total_pages == 2
    assert all(p.images for p in model.seen_pages), "image-only PDF pages must be rendered"


@pytest.mark.parametrize(
    "name,fmt,ctype,ext",
    [
        ("IMG_001.jpg", "JPEG", "image/jpeg", ".jpg"),
        ("photo.jpeg", "JPEG", "image/jpg", ".jpg"),
        ("image2.png", "PNG", "image/png", ".png"),
        ("WhatsApp Image 2026-01-01.webp", "WEBP", "image/webp", ".webp"),
    ],
)
def test_image_uploads(make_engine, name, fmt, ctype, ext):
    model = ScriptedModel(plan=one_group_plan())
    data = page_image(["CSC 308", "1. Define a process"], fmt=fmt)
    result = make_engine(model).sort(tacsfon_request(), uploads((name, data, ctype)))
    assert result.status == "success", result
    assert model.seen_pages[0].file_kind == "image"
    assert model.seen_pages[0].images and model.seen_pages[0].images[0][:3] == b"\xff\xd8\xff"
    assert result.groups[0].proposed_filename.endswith(ext)
    assert result.groups[0].source_files[0].original_filename == name


def test_docx_upload_splits_on_page_breaks_and_reads_tables(make_engine):
    model = ScriptedModel(plan=one_group_plan())
    data = docx_file(
        [["CPS 401 Lecture Notes", "Introduction"], ["Chapter 2"]],
        table_rows=[["Course", "CPS 401"], ["Session", "2024/2025"]],
    )
    result = make_engine(model).sort(
        tacsfon_request(),
        uploads(("document.docx", data, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
    )
    assert result.summary.total_pages == 2
    assert "Course | CPS 401" in model.seen_pages[0].text
    assert result.groups[0].proposed_filename.endswith(".docx")


# ============================================================
# 6-9  VALIDATION / LIMITS
# ============================================================

def test_invalid_file_type_is_rejected_by_content_not_extension(make_engine):
    model = ScriptedModel(plan=one_group_plan())
    fake_pdf = b"MZ\x90\x00 this is an executable, not a pdf"
    png_named_pdf = page_image(["CSC 308"], fmt="PNG")
    result = make_engine(model).sort(
        tacsfon_request(),
        uploads(("notes.pdf", fake_pdf, "application/pdf"), ("misnamed.pdf", png_named_pdf, "application/pdf")),
    )
    assert [r.reason_code.value for r in result.rejected] == ["unsupported_type"]
    assert result.rejected[0].original_filename == "notes.pdf"
    # The PNG is processed as an image despite its .pdf name, with a warning.
    assert model.seen_pages[0].file_kind == "image"
    assert any("does not match its content" in w for w in result.groups[0].warnings)
    assert result.status == "partial"


def test_invalid_file_type_via_api(client, use_model):
    use_model(ScriptedModel(plan=one_group_plan()))
    resp = post(client, [("files", ("notes.txt", b"plain text", "text/plain"))])
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "needs_review"
    assert body["rejected"][0]["reason_code"] == "unsupported_type"


def test_oversized_request_rejected_while_streaming(make_engine):
    engine = make_engine(ScriptedModel(), max_request_bytes=50_000)
    big = page_image(["x"], fmt="PNG", size=(2000, 2000)) + b"\0" * 60_000
    with pytest.raises(RequestTooLargeError):
        engine.sort(tacsfon_request(), uploads(("a.png", big, "image/png")))


def test_oversized_request_rejected_by_content_length_before_reading(client, monkeypatch):
    import server
    from document_sorter.config import SorterSettings

    monkeypatch.setattr(server, "SORTER_SETTINGS", SorterSettings(max_request_bytes=1000))
    monkeypatch.setattr(server, "MULTIPART_OVERHEAD_BYTES", 100)
    resp = post(client, [("files", ("a.png", b"\x89PNG\r\n\x1a\n" + b"0" * 5000, "image/png"))])
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "request_too_large"


def test_default_limit_is_100_mb():
    import server

    assert server.SORTER_SETTINGS.max_request_bytes == 100 * 1024 * 1024


def test_empty_request(client, use_model, make_engine):
    use_model(ScriptedModel())
    resp = post(client, [])
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "no_files"
    assert resp.json()["status"] == "error"

    with pytest.raises(NoFilesError):
        make_engine(ScriptedModel()).sort(tacsfon_request(), [])


def test_empty_file_is_rejected(make_engine):
    result = make_engine(ScriptedModel()).sort(tacsfon_request(), uploads(("empty.jpg", b"", "image/jpeg")))
    assert result.rejected[0].reason_code.value == "empty_file"
    assert result.status == "needs_review"


@pytest.mark.parametrize(
    "schema,code",
    [
        ("not json", "malformed_schema"),
        ('{"schema": ["a"]}', "malformed_schema"),
        ("[]", "malformed_schema"),
        ('["Course Code"]', "malformed_schema"),
        ('["course", "course"]', "malformed_schema"),
        ("[1, 2]", "malformed_schema"),
    ],
)
def test_malformed_schema(client, use_model, schema, code):
    use_model(ScriptedModel())
    resp = post(client, [("files", ("a.png", page_image(["x"]), "image/png"))], schema=schema, allowed=None)
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == code


def test_missing_schema(client, use_model):
    use_model(ScriptedModel())
    resp = client.post("/sort-documents", files=[("files", ("a.png", page_image(["x"]), "image/png"))], headers=AUTH)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "missing_schema"


def test_invalid_allowed_values(client, use_model):
    use_model(ScriptedModel())
    files = [("files", ("a.png", page_image(["x"]), "image/png"))]
    resp = post(client, files, schema=["category"], allowed={"colour": ["red"]})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_allowed_values"
    resp = post(client, files, schema=["category"], allowed={"category": "not-a-list"})
    assert resp.status_code == 422


# ============================================================
# 10  DYNAMIC SCHEMA
# ============================================================

def test_valid_dynamic_schema_for_another_client(client, use_model):
    schema = ["document_type", "country", "state", "institution", "year"]
    model = use_model(
        ScriptedModel(
            plan=lambda pages, ctx: {
                "groups": [
                    group(
                        [0],
                        fields={
                            "document_type": "Certificate",
                            "country": "Nigeria",
                            "state": "Ogun",
                            "institution": "Example Polytechnic",
                            "year": "2023",
                        },
                    )
                ]
            }
        )
    )
    resp = post(
        client,
        [("files", ("document.png", page_image(["CERTIFICATE"]), "image/png"))],
        schema=schema,
        allowed={"document_type": ["Certificate", "Transcript"]},
        request_id="client-42",
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["request_id"] == "client-42"
    assert body["schema"] == schema
    assert list(body["groups"][0]["classification"]) == schema
    assert body["groups"][0]["proposed_filename"] == "certificate-nigeria-ogun-example_polytechnic-2023.png"
    assert model.seen_context.schema_fields == schema
    DocumentSortingResponse.model_validate(body)


# ============================================================
# 11-12  CONTINUATION + BOUNDARIES (cross-file)
# ============================================================

def _exam_batch():
    return uploads(
        ("IMG_001.jpg", page_image(["EEE 306", "Question 1", "Question 2", "Question 3", "Question 4"], fmt="JPEG"), "image/jpeg"),
        ("IMG_002.jpg", page_image(["Question 5", "Question 6"], fmt="JPEG"), "image/jpeg"),
        ("IMG_003.jpg", page_image(["CPS 401", "Question 1", "Question 2"], fmt="JPEG"), "image/jpeg"),
    )


def _exam_plan(pages, ctx):
    return {
        "page_relationships": [
            {"page_index": 0, "relationship": "first_page"},
            {"page_index": 1, "relationship": "continuation", "related_page_index": 0},
            {"page_index": 2, "relationship": "new_document"},
        ],
        "groups": [
            group(
                [0, 1],
                fields={
                    **{k: v for k, v in tacsfon_fields("EEE 306").items()},
                    "course": {"value": "EEE 306", "source": "page_content", "evidence": "Header on IMG_001"},
                },
                relationship_evidence=["IMG_002 continues question numbering (5-6) after IMG_001 (1-4)"],
            ),
            group([2], fields=tacsfon_fields("CPS 401")),
        ],
    }


def test_continuation_pages_grouped_across_files(make_engine):
    result = make_engine(ScriptedModel(plan=_exam_plan)).sort(tacsfon_request(), _exam_batch())
    first = result.groups[0]
    assert first.classification["course"] == "EEE 306"
    assert [s.original_filename for s in first.source_files] == ["IMG_001.jpg", "IMG_002.jpg"]
    assert all(s.page_numbers == [1] for s in first.source_files)
    assert any("continues question numbering" in e for e in first.evidence)
    assert first.proposed_filename.endswith(".jpg")


def test_new_document_boundary_detected(make_engine):
    result = make_engine(ScriptedModel(plan=_exam_plan)).sort(tacsfon_request(), _exam_batch())
    assert [g.group_id for g in result.groups] == ["group_001", "group_002"]
    assert result.groups[1].classification["course"] == "CPS 401"
    assert [s.original_filename for s in result.groups[1].source_files] == ["IMG_003.jpg"]
    assert result.status == "success"


def test_every_page_is_accounted_for_exactly_once(make_engine):
    result = make_engine(ScriptedModel(plan=_exam_plan)).sort(tacsfon_request(), _exam_batch())
    refs = [
        (s.file_id, n)
        for g in result.groups + result.isolated
        for s in g.source_files
        for n in s.page_numbers
    ]
    assert sorted(refs) == [("file_001", 1), ("file_002", 1), ("file_003", 1)]


# ============================================================
# 13-14  ISOLATION / CONFLICTS
# ============================================================

def test_missing_information_isolates_instead_of_guessing(make_engine):
    plan = lambda pages, ctx: {  # noqa: E731
        "groups": [
            group(
                [0],
                decision="isolated",
                confidence="insufficient",
                isolation_reason_code="missing_required_field",
                isolation_reason="Course code could not be determined from the document or its neighbouring pages.",
                fields={"material_type": "Past Questions"},
            )
        ]
    }
    result = make_engine(ScriptedModel(plan=plan)).sort(
        tacsfon_request(), uploads(("IMG_9.jpg", page_image(["Question 5"], fmt="JPEG"), "image/jpeg"))
    )
    assert not result.groups
    iso = result.isolated[0]
    assert iso.status.value == "isolated"
    assert "missing_required_field" in [r.value for r in iso.isolation_reasons]
    assert iso.proposed_filename is None
    assert iso.classification["course"] is None
    assert "course" in iso.reason
    assert result.status == "needs_review"


def test_engine_isolates_when_model_claims_classified_but_required_field_missing(make_engine):
    plan = lambda pages, ctx: {"groups": [group([0], fields={"course": "CSC 308"})]}  # noqa: E731
    result = make_engine(ScriptedModel(plan=plan)).sort(
        tacsfon_request(), uploads(("a.png", page_image(["CSC 308"]), "image/png"))
    )
    assert result.isolated and not result.groups
    assert "'level'" in result.isolated[0].reason


def test_conflicting_metadata_is_isolated(make_engine):
    plan = lambda pages, ctx: {  # noqa: E731
        "groups": [
            group(
                [0, 1],
                confidence="low",
                fields=tacsfon_fields("EEE 306"),
                conflicts=["Page 1 header says EEE 306 but page 2 header says EEE 308"],
            )
        ]
    }
    result = make_engine(ScriptedModel(plan=plan)).sort(
        tacsfon_request(),
        uploads(("a.png", page_image(["EEE 306"]), "image/png"), ("b.png", page_image(["EEE 308"]), "image/png")),
    )
    iso = result.isolated[0]
    codes = [r.value for r in iso.isolation_reasons]
    assert "conflicting_metadata" in codes
    assert any("EEE 308" in w for w in iso.warnings)


def test_value_outside_allowed_values_is_isolated_not_coerced(make_engine):
    plan = lambda pages, ctx: {"groups": [group([0], fields=tacsfon_fields("CSC 308", level="600"))]}  # noqa: E731
    result = make_engine(ScriptedModel(plan=plan)).sort(
        tacsfon_request(), uploads(("a.png", page_image(["CSC 308"]), "image/png"))
    )
    iso = result.isolated[0]
    assert "invalid_schema_value" in [r.value for r in iso.isolation_reasons]
    assert iso.classification["level"] is None


def test_allowed_value_normalization_is_conservative():
    from document_sorter.classification import match_allowed_value

    allowed = ["Past Questions", "Materials", "CSC 308", "2024/2025", "100"]
    assert match_allowed_value("past questions", allowed) == "Past Questions"
    assert match_allowed_value("CSC308", allowed) == "CSC 308"
    assert match_allowed_value("2024-2025", allowed) == "2024/2025"
    assert match_allowed_value("Material", allowed) is None
    assert match_allowed_value("1000", allowed) is None
    assert match_allowed_value("Past", allowed) is None
    assert match_allowed_value("CSC 308: Operating Systems", allowed) == "CSC 308"
    assert match_allowed_value("2024-2025", ["2024"]) is None
    assert match_allowed_value("100 - 200", allowed) is None


# ============================================================
# 15  ADMIN CORRECTION / REPROCESSING
# ============================================================

def test_admin_correction_is_external_context(client, use_model):
    def plan(pages, ctx):
        fields = tacsfon_fields(None)
        fields["course"] = {"value": "EEE 306", "source": "administrator", "evidence": "from context"}
        return {"groups": [group([0], fields=fields)]}

    model = use_model(ScriptedModel(plan=plan))
    previous = {"status": "isolated", "reason": "Course could not be determined."}
    resp = post(
        client,
        [("files", ("IMG_002.jpg", page_image(["Question 5", "Question 6"], fmt="JPEG"), "image/jpeg"))],
        additional_context=json.dumps({"course": "EEE 306", "source": "administrator"}),
        previous_result=json.dumps(previous),
        request_id="reprocess-1",
    )
    assert resp.status_code == 200, resp.text
    grp = resp.json()["groups"][0]
    assert grp["classification"]["course"] == "EEE 306"
    assert grp["field_sources"]["course"] == "administrator"
    assert grp["field_sources"]["level"] == "document"
    assert any("not read from the document" in e for e in grp["evidence"])
    assert model.seen_context.administrator_values == {"course": "EEE 306"}
    assert model.seen_context.previous_result == previous


def test_admin_value_overrides_model_isolation_for_missing_field(make_engine):
    plan = lambda pages, ctx: {  # noqa: E731
        "groups": [
            group(
                [0],
                decision="isolated",
                isolation_reason_code="missing_required_field",
                isolation_reason="course missing",
                fields=tacsfon_fields(None),
            )
        ]
    }
    request = tacsfon_request(additional_context={"course": "EEE 306"})
    result = make_engine(ScriptedModel(plan=plan)).sort(
        request, uploads(("a.jpg", page_image(["Q5"], fmt="JPEG"), "image/jpeg"))
    )
    assert result.groups and result.groups[0].classification["course"] == "EEE 306"


def test_admin_value_conflicting_with_document_produces_warning(make_engine):
    plan = lambda pages, ctx: {"groups": [group([0], fields=tacsfon_fields("EEE 308"))]}  # noqa: E731
    result = make_engine(ScriptedModel(plan=plan)).sort(
        tacsfon_request(additional_context={"course": "EEE 306", "source": "administrator"}),
        uploads(("a.jpg", page_image(["EEE 308"], fmt="JPEG"), "image/jpeg")),
    )
    grp = result.groups[0]
    assert grp.classification["course"] == "EEE 306"
    assert any("EEE 308" in w for w in grp.warnings)


def test_admin_context_with_unknown_field_is_rejected(client, use_model):
    use_model(ScriptedModel())
    resp = post(
        client,
        [("files", ("a.png", page_image(["x"]), "image/png"))],
        additional_context=json.dumps({"colour": "red"}),
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_additional_context"


# ============================================================
# 16  STRUCTURED OUTPUT VALIDATION (real model adapter, fake LLM)
# ============================================================

class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.received = []

    def invoke(self, messages):
        from langchain_core.messages import AIMessage

        self.received.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return AIMessage(content=reply)


def _analysis_reply(indices):
    return json.dumps({"pages": [{"page_index": i, "readable": True} for i in indices]})


def _plan_reply(groups):
    return "```json\n" + json.dumps({"groups": groups}) + "\n```"


def test_malformed_model_output_is_repaired(settings):
    from document_sorter.ai import LangChainSorterModel
    from document_sorter.engine import DocumentSortingEngine

    fields = {k: {"value": v, "source": "page_content", "evidence": "x"} for k, v in tacsfon_fields("CSC 308").items()}
    llm = FakeLLM(
        [
            "not json at all",
            _analysis_reply([0, 1]),
            # plan that forgets page 1 -> rejected by validation, then repaired
            _plan_reply([{"page_indices": [0], "decision": "classified", "confidence": "high", "fields": fields}]),
            _plan_reply([{"page_indices": [0, 1], "decision": "classified", "confidence": "high", "fields": fields}]),
        ]
    )
    model = LangChainSorterModel(llm, settings)
    result = DocumentSortingEngine(model=model, settings=settings).sort(
        tacsfon_request(),
        uploads(("a.png", page_image(["CSC 308"]), "image/png"), ("b.png", page_image(["2."]), "image/png")),
    )
    assert result.status == "success"
    assert model.calls == 4
    repair_prompt = llm.received[3][-1].content
    assert "not assigned to any group" in repair_prompt


def test_persistently_invalid_plan_isolates_everything(settings):
    from document_sorter.ai import LangChainSorterModel
    from document_sorter.engine import DocumentSortingEngine

    llm = FakeLLM([_analysis_reply([0]), "{}", '{"groups": "nope"}'])
    model = LangChainSorterModel(llm, settings)
    result = DocumentSortingEngine(model=model, settings=settings).sort(
        tacsfon_request(), uploads(("a.png", page_image(["x"]), "image/png"))
    )
    assert not result.groups
    assert result.isolated[0].isolation_reasons[0].value == "model_output_invalid"


def test_unknown_fields_from_model_are_dropped(make_engine):
    plan = lambda pages, ctx: {  # noqa: E731
        "groups": [group([0], fields={**tacsfon_fields("CSC 308"), "lecturer": "Dr X"})]
    }
    result = make_engine(ScriptedModel(plan=plan)).sort(
        tacsfon_request(), uploads(("a.png", page_image(["x"]), "image/png"))
    )
    assert "lecturer" not in result.groups[0].classification
    assert any("lecturer" in w for w in result.groups[0].warnings)


def test_model_failure_returns_structured_error(client, use_model, settings):
    from document_sorter.ai import LangChainSorterModel

    use_model(LangChainSorterModel(FakeLLM([RuntimeError("boom"), RuntimeError("boom")]), settings))
    resp = post(client, [("files", ("a.png", page_image(["x"]), "image/png"))], request_id="err-1")
    assert resp.status_code == 502
    body = resp.json()
    assert body == {
        "status": "error",
        "request_id": "err-1",
        "error": {"code": "model_unavailable", "message": body["error"]["message"], "details": {}},
    }
    assert "Traceback" not in resp.text and "boom" not in resp.text


def test_timeout_returns_504(client, use_model):
    use_model(ScriptedModel(plan=one_group_plan()), request_timeout_seconds=-1)
    resp = post(client, [("files", ("a.png", page_image(["x"]), "image/png"))])
    assert resp.status_code == 504
    assert resp.json()["error"]["code"] == "timeout"


# ============================================================
# 17  DUPLICATES
# ============================================================

def test_duplicate_files_are_reported_not_deleted(make_engine):
    image = page_image(["CSC 308", "1."], fmt="JPEG")
    model = ScriptedModel(plan=one_group_plan())
    result = make_engine(model).sort(
        tacsfon_request(),
        uploads(("IMG_001.jpg", image, "image/jpeg"), ("IMG_001 (1).jpg", image, "image/jpeg")),
    )
    assert result.duplicates[0].file_id == "file_002"
    assert result.duplicates[0].duplicate_of_file_id == "file_001"
    assert len(model.seen_pages) == 1
    assert result.status == "success"


def test_groups_with_identical_classification_flagged(make_engine):
    plan = lambda pages, ctx: {  # noqa: E731
        "groups": [group([0], fields=tacsfon_fields("CSC 308")), group([1], fields=tacsfon_fields("CSC 308"))]
    }
    result = make_engine(ScriptedModel(plan=plan)).sort(
        tacsfon_request(),
        uploads(("a.png", page_image(["CSC 308 A"]), "image/png"), ("b.png", page_image(["CSC 308 B"]), "image/png")),
    )
    assert result.groups[1].possible_duplicate_of == "group_001"


# ============================================================
# 18  CLEANUP
# ============================================================

def _track_workspaces(monkeypatch):
    import document_sorter.ingestion as ingestion

    created = []
    real = ingestion.tempfile.mkdtemp

    def tracking(*args, **kwargs):
        path = real(*args, **kwargs)
        created.append(Path(path))
        return path

    monkeypatch.setattr(ingestion.tempfile, "mkdtemp", tracking)
    return created


def test_temporary_files_removed_after_success(make_engine, monkeypatch):
    created = _track_workspaces(monkeypatch)
    make_engine(ScriptedModel(plan=one_group_plan())).sort(
        tacsfon_request(), uploads(("a.png", page_image(["x"]), "image/png"))
    )
    assert created and not any(p.exists() for p in created)


def test_temporary_files_removed_after_error(make_engine, monkeypatch):
    created = _track_workspaces(monkeypatch)

    def broken_plan(pages, ctx):
        raise RuntimeError("unexpected failure")

    with pytest.raises(RuntimeError):
        make_engine(ScriptedModel(plan=broken_plan)).sort(
            tacsfon_request(), uploads(("a.png", page_image(["x"]), "image/png"))
        )
    assert created and not any(p.exists() for p in created)


# ============================================================
# 19  AUTHENTICATION
# ============================================================

def test_api_requires_valid_key(client, use_model, monkeypatch):
    use_model(ScriptedModel(plan=one_group_plan()))
    files = [("files", ("a.png", page_image(["x"]), "image/png"))]

    assert post(client, files, headers={}).status_code == 401
    wrong = post(client, files, headers={"Authorization": "Bearer nope"})
    assert wrong.status_code == 401 and wrong.json()["error"]["code"] == "unauthorized"
    assert post(client, files, headers={"X-API-Key": API_KEY}).status_code == 200
    assert post(client, files).status_code == 200

    monkeypatch.delenv("NEBULA_API_KEY")
    resp = post(client, files)
    assert resp.status_code == 503 and resp.json()["error"]["code"] == "auth_not_configured"


def test_other_routes_do_not_require_sorting_key(client):
    assert client.get("/health").json() == {"status": "ok", "service": "nebula"}
    assert client.get("/").json()["status"] == "Nebula Agent Running"


# ============================================================
# GENERIC ENGINE (no client-specific logic)
# ============================================================

def test_engine_has_no_client_specific_values():
    package = Path(__file__).resolve().parent.parent / "document_sorter"
    forbidden = ["tacsfon", "oaustech", "school of science", "computer science", "csc 308",
                 "500 level", "2024/2025", "first semester", "past questions"]
    for source in package.glob("*.py"):
        text = source.read_text(encoding="utf-8").lower()
        for term in forbidden:
            assert term not in text, f"{term!r} hardcoded in {source.name}"


def test_proposed_filename_for_tacsfon_example():
    from document_sorter.filenames import propose_filename

    classification = tacsfon_fields("CSC 308", level="500")
    name = propose_filename(classification, TACSFON_SCHEMA, ["image/jpeg"])
    assert name == "past_questions-500-school_of_science-computer_science-CSC_308-2024_2025-first_semester.jpg"
    assert propose_filename(classification, TACSFON_SCHEMA, ["image/jpeg", "application/pdf"]).endswith(".pdf")


# ============================================================
# FIELD DESCRIPTIONS / CONCURRENCY
# ============================================================

def test_field_descriptions_reach_the_model_and_are_validated(client, use_model):
    model = use_model(ScriptedModel(plan=one_group_plan()))
    files = [("files", ("a.png", page_image(["x"]), "image/png"))]
    resp = post(client, files, field_descriptions=json.dumps({"course": "course code only, e.g. ABC 101"}))
    assert resp.status_code == 200, resp.text
    assert model.seen_context.field_descriptions == {"course": "course code only, e.g. ABC 101"}

    bad = post(client, files, field_descriptions=json.dumps({"colour": "x"}))
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "malformed_schema"


def test_concurrent_page_analysis_keeps_page_order(make_engine):
    import random
    import time as _time

    class SlowModel(ScriptedModel):
        def analyze_pages(self, pages, context):
            _time.sleep(random.uniform(0, 0.05))
            return super().analyze_pages(pages, context)

    seen = {}

    def plan(pages, ctx):
        seen["order"] = [p.index for p in pages]
        return {"groups": [group([p.index for p in pages], fields=tacsfon_fields("CSC 308"))]}

    files = [(f"IMG_{i:03d}.png", page_image([f"page {i}"]), "image/png") for i in range(9)]
    engine = make_engine(SlowModel(plan=plan), pages_per_analysis_call=1, analysis_concurrency=4)
    result = engine.sort(tacsfon_request(), uploads(*files))
    assert seen["order"] == list(range(9))
    assert [s.original_filename for s in result.groups[0].source_files] == [f[0] for f in files]
