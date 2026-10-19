# Nebula Document Sorting API — Integration Contract

Nebula groups and classifies a batch of uploaded documents against a schema
**the client supplies**. It does not store, publish, approve or rename files:
it returns a structured proposal; the client (e.g. TACSFON) owns storage,
approval, publication and final filenames.

Engine version: `1.0.0` (returned as `engine_version`).

---

## 1. Endpoint

| | |
|---|---|
| Method | `POST` |
| Path | `/sort-documents` |
| Body | `multipart/form-data` |
| Success | `200 OK`, JSON `DocumentSortingResponse` |

## 2. Authentication (server-to-server only)

Send the shared secret configured on the Nebula server as `NEBULA_API_KEY`:

```
Authorization: Bearer <NEBULA_API_KEY>
```

(`X-API-Key: <NEBULA_API_KEY>` is also accepted.)

- Call Nebula **from the TACSFON backend only**. Never ship the key to a browser.
- The key is checked before the upload body is read.
- Missing/wrong key → `401 unauthorized`. Key not configured on Nebula → `503 auth_not_configured`.
- CORS is not an authentication mechanism and is not involved in this call.

## 3. Multipart fields

| Field | Required | Format | Description |
|---|---|---|---|
| `files` | yes | file part, repeat once per file | Upload **in page order**. Order matters: it is the batch context for continuation/boundary reasoning. |
| `schema` | yes | JSON array of strings | Fields to classify by, in filename order. Names: lowercase letters, digits, `_`, start with a letter, ≤ 64 chars, unique, ≤ 20 fields. `source`/`notes` are reserved. |
| `allowed_values` | no | JSON object | `field → [allowed values]`. Keys may be the field name or its plural (`level` or `levels`). An empty list means unrestricted. Unknown keys → 422. ≤ 5000 values/field, each ≤ 200 chars. |
| `required_fields` | no | JSON array | Fields that must be determined for a group to be `classified`. **Default: every schema field.** Must be a subset of `schema`. |
| `field_descriptions` | no | JSON object | `field → plain-language meaning` (≤ 500 chars). Strongly recommended for unrestricted fields, e.g. `{"course": "the course code only, e.g. ABC 101"}`. |
| `additional_context` | no | JSON object | External values for reprocessing (see §6). |
| `previous_result` | no | JSON object | The earlier group result being reprocessed (for reference). |
| `request_id` | no | string | 1–128 chars of `A-Z a-z 0-9 . _ : -`. Echoed back. Generated (`nsr_<hex>`) if omitted. |
| `task` | no | string | If sent, must be `document_sorting`. |

### TACSFON example

```bash
curl -X POST "$NEBULA_URL/sort-documents" \
  -H "Authorization: Bearer $NEBULA_API_KEY" \
  -F 'request_id=6f1c2c1e-6a0b-4f0e-9a51-0d6e3e1f8a11' \
  -F 'schema=["material_type","level","school","department","course","session","semester"]' \
  -F 'allowed_values={"material_type":["Past Questions","Manuals / Lecture Notes","Workbooks","Materials"],"levels":["100","200","300","400","500"],"schools":[],"departments":[],"courses":[],"sessions":[],"semesters":[]}' \
  -F 'field_descriptions={"course":"course code only, e.g. ABC 101","session":"academic session, e.g. 2023/2024","material_type":"Past Questions = past examination or test question papers"}' \
  -F 'files=@IMG_001.jpg;type=image/jpeg' \
  -F 'files=@IMG_002.jpg;type=image/jpeg' \
  -F 'files=@scan_9382.pdf;type=application/pdf'
```

Node (server side):

```js
const form = new FormData();
form.append("request_id", uploadId);
form.append("schema", JSON.stringify(schema));
form.append("allowed_values", JSON.stringify(allowedValues));
for (const f of files) form.append("files", new Blob([f.buffer], { type: f.mime }), f.name);
const res = await fetch(`${NEBULA_URL}/sort-documents`, {
  method: "POST",
  headers: { Authorization: `Bearer ${process.env.NEBULA_API_KEY}` },
  body: form,
  signal: AbortSignal.timeout(330_000),
});
```

## 4. Supported files and limits

| Type | MIME | How it is read |
|---|---|---|
| PDF | `application/pdf` | Text layer; pages with < 200 chars of text are rendered and analysed visually. Title/subject metadata used as a weak signal. Encrypted PDFs are rejected. |
| JPEG | `image/jpeg` (`image/jpg` accepted) | Visual analysis (+ OCR text when Tesseract is installed). EXIF orientation corrected. |
| PNG | `image/png` | as JPEG |
| WebP | `image/webp` | as JPEG |
| DOCX | `application/vnd.openxmlformats-officedocument.wordprocessingml.document` | Paragraphs, tables and header/footer text in order; split into pages at explicit page breaks; embedded images analysed when text is sparse. |

- The type is detected **from file content**, not the extension or Content-Type.
  A mismatch is accepted (content wins) with a warning; unrecognised content is rejected.
- **Maximum total upload: 100 MB per request** (`NEBULA_SORT_MAX_REQUEST_MB`).
  Requests above it get `413 request_too_large`; this is checked from `Content-Length`
  before upload and enforced while streaming.
- Max files per request: 200. Max pages per request: 250. A file that would exceed the
  page limit is rejected with `page_limit_exceeded`; send it in another request.
- Uploads are written to a per-request temporary directory that is deleted after the
  response (also on errors). Nothing is retained.

## 5. Response (`200 OK`)

```json
{
  "status": "partial",
  "task": "document_sorting",
  "request_id": "6f1c2c1e-6a0b-4f0e-9a51-0d6e3e1f8a11",
  "engine_version": "1.0.0",
  "schema": ["material_type", "level", "school", "department", "course", "session", "semester"],
  "groups": [
    {
      "group_id": "group_001",
      "status": "classified",
      "source_files": [
        {"file_id": "file_001", "original_filename": "IMG_001.jpg", "page_numbers": [1]},
        {"file_id": "file_002", "original_filename": "IMG_002.jpg", "page_numbers": [1]}
      ],
      "classification": {
        "material_type": "Past Questions", "level": "300", "school": "School of Engineering",
        "department": "Electrical Engineering", "course": "EEE 306",
        "session": "2024/2025", "semester": "First Semester"
      },
      "field_sources": {
        "material_type": "document", "level": "document", "school": "document",
        "department": "document", "course": "document", "session": "document",
        "semester": "batch_context"
      },
      "proposed_filename": "past_questions-300-school_of_engineering-electrical_engineering-EEE_306-2024_2025-first_semester.jpg",
      "confidence": "medium",
      "reason": null,
      "isolation_reasons": [],
      "evidence": [
        "IMG_002 continues the question numbering (5-6) of IMG_001 (1-4) with the same layout",
        "course: 'EEE 306' - course header on IMG_001"
      ],
      "warnings": [],
      "possible_duplicate_of": null
    }
  ],
  "isolated": [
    {
      "group_id": "group_002",
      "status": "isolated",
      "source_files": [{"file_id": "file_003", "original_filename": "scan_9382.pdf", "page_numbers": [1, 2]}],
      "classification": {"material_type": "Past Questions", "level": null, "school": null,
                         "department": null, "course": null, "session": null, "semester": null},
      "field_sources": {"material_type": "document", "level": null, "school": null,
                        "department": null, "course": null, "session": null, "semester": null},
      "proposed_filename": null,
      "confidence": "insufficient",
      "reason": "Could not determine 'level', 'school', 'department', 'course', 'session', 'semester' from the document or its neighbouring pages.",
      "isolation_reasons": ["missing_required_field"],
      "evidence": [],
      "warnings": [],
      "possible_duplicate_of": null
    }
  ],
  "rejected": [
    {"file_id": "file_004", "original_filename": "notes.doc", "reason_code": "unsupported_type",
     "reason": "Unsupported or unrecognised file: legacy binary Office format (.doc) is not supported; convert to DOCX or PDF."}
  ],
  "duplicates": [
    {"file_id": "file_005", "original_filename": "IMG_001 (1).jpg", "duplicate_of_file_id": "file_001",
     "reason": "Identical file content (SHA-256 match) to an earlier file in this request."}
  ],
  "summary": {
    "total_files": 5, "accepted_files": 3, "total_bytes": 4718233, "total_pages": 4,
    "classified_groups": 1, "isolated_groups": 1, "rejected_files": 1, "duplicate_files": 1,
    "model_calls": 2, "processing_ms": 61234
  }
}
```

### Field reference

| Field | Meaning |
|---|---|
| `status` | `success`: everything accepted was classified and nothing was rejected. `partial`: at least one classified group, plus isolated/rejected content. `needs_review`: nothing classified. |
| `file_id` | `file_001`, `file_002`, … in the order the `files` parts were sent. Use it to map results back to your raw uploads. |
| `groups[]` | Classified groups only (`status = "classified"`). |
| `isolated[]` | Groups needing human review (`status = "isolated"`). Same shape as `groups[]`; `proposed_filename` is always `null`; `reason` is always set. |
| `source_files` | Every raw file/page in the group. **Every accepted page appears in exactly one group** across `groups` + `isolated`. |
| `classification` | One key per schema field, in schema order. `null` = unknown. Values in `allowed_values` are returned in their exact allowed spelling. |
| `field_sources` | `document` (read from the group's pages), `batch_context` (inferred from related pages in the batch), `administrator` (from `additional_context`, **not** from the document), or `null`. |
| `confidence` | `high`: every value directly evidenced and boundaries unambiguous. `medium`: some value relies on batch context, or minor ambiguity/warnings. `low` / `insufficient`: never `classified`. There is no numeric score. |
| `isolation_reasons` | Any of `missing_required_field`, `ambiguous_value`, `ambiguous_boundary`, `unreadable_content`, `conflicting_metadata`, `unsupported_content`, `insufficient_context`, `multiple_possible_classifications`, `invalid_schema_value`, `low_confidence`, `model_output_invalid`. |
| `evidence` / `warnings` | Human-readable explanations for the administrator. Warnings include conflicts, normalizations (e.g. `'csc308' normalized to 'CSC 308'`), extension/content mismatches, possible duplicate pages. |
| `proposed_filename` | A **suggestion** built from classification values in schema order (`-` between fields, `_` within; codes in capitals keep their case). Extension = the source type, or `.pdf` when a group mixes file types. TACSFON generates/validates the final storage name. |
| `possible_duplicate_of` | `group_id` of an earlier group with identical classification. Nothing is ever removed automatically. |
| `rejected[].reason_code` | `empty_file`, `unsupported_type`, `corrupted_file`, `encrypted_file`, `unreadable_content`, `unrelated_content`, `page_limit_exceeded`. Rejection is only used for genuinely unusable input; ambiguous material is isolated. |
| `duplicates[]` | Byte-identical repeats of an earlier file in the same request. They are not analysed again; the original's group covers them. |

## 6. Reprocessing an isolated group (administrator correction)

Re-send the group's raw files (the `source_files`, in order) together with the
admin-supplied values and, optionally, the previous result:

```
-F 'additional_context={"course":"EEE 306","source":"administrator","notes":"Confirmed by HOD"}'
-F 'previous_result=<the isolated group JSON from the earlier response>'
```

- `additional_context` keys must be schema fields (plus optional `source`, `notes`).
  Values must be in `allowed_values` when that field is restricted, otherwise `422`.
- Supplied values are used as given and reported with `field_sources[field] = "administrator"`
  and an evidence line saying they were not read from the document.
- If the document shows a different value, the admin value is still used and a warning records the difference.
- Fields the admin did not supply are re-evaluated from the document; the group may stay isolated
  if other required fields are still unknown.

## 7. Errors

All errors on this endpoint use one shape:

```json
{
  "status": "error",
  "request_id": "6f1c2c1e-6a0b-4f0e-9a51-0d6e3e1f8a11",
  "error": {"code": "malformed_schema", "message": "The sorting request is invalid.",
            "details": {"errors": [{"field": "schema.0", "message": "Invalid schema field name 'Course Code' ..."}]}}
}
```

| HTTP | `error.code` | When |
|---|---|---|
| 400 | `no_files` | No `files` parts. |
| 401 | `unauthorized` | Missing or wrong API key. |
| 413 | `request_too_large` / `too_many_files` | Over 100 MB total, or over 200 files. |
| 422 | `missing_schema` | `schema` field absent/empty. |
| 422 | `malformed_schema` | `schema` / `required_fields` / `field_descriptions` invalid. |
| 422 | `invalid_allowed_values` | Not an object, unknown field key, non-string values. |
| 422 | `invalid_additional_context` | Unknown field, or value not allowed. |
| 422 | `invalid_previous_result`, `invalid_request_id`, `unsupported_task`, `invalid_request` | Other invalid input. |
| 500 | `temporary_storage_failure`, `internal_error` | Server-side failure. No stack traces are returned. |
| 502 | `model_unavailable` | The AI model could not be reached/failed. Safe to retry later. |
| 503 | `auth_not_configured`, `model_unavailable` | Nebula is not configured for sorting. |
| 504 | `timeout` | Processing exceeded the time budget. Retry with fewer files per request. |

Per-file problems (unsupported, corrupted, empty) are **not** request errors: they
appear in `rejected[]` of a `200` response. Malformed AI output that cannot be
repaired yields isolated groups with `model_output_invalid`, never an invented classification.

## 8. Timeouts and sizing

- Processing is synchronous. Typical time: ~20–60 s for a few pages; larger
  batches scale with page count (pages are analysed 4 per model call, 4 calls in parallel).
- Server budget per request: `NEBULA_SORT_TIMEOUT_SECONDS` (default **300 s**). Per model call: 120 s.
- **Set the TACSFON client timeout to at least 330 s.**
- For very large uploads, split into several requests of related files (keep pages of
  one document in the same request — grouping only works within a request).
- The request/response contract is designed so a future asynchronous job mode
  (submit → poll) can return the same `DocumentSortingResponse`.

## 9. Other clients

Nothing in the engine is TACSFON-specific. Another application only changes the
request, for example:

```
-F 'schema=["document_type","country","state","institution","year"]'
-F 'allowed_values={"document_type":["Certificate","Transcript"]}'
-F 'required_fields=["document_type","year"]'
```
