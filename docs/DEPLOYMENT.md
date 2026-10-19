# Nebula — Deployment & Operations (Render)

## Build / start

| Setting | Value |
|---|---|
| Build command | `pip install -r requirement.txt` |
| Start command | `uvicorn server:app --host 0.0.0.0 --port $PORT` |
| Health check path | `/health` (liveness only, no model calls) |

`requirement.txt` contains only server dependencies. The desktop voice agent's
packages (`pyaudio`, `vosk`, `pyttsx3`, `SpeechRecognition`) moved to
`requirements-desktop.txt`, because `pyaudio` cannot be installed on Render
without PortAudio system headers. Test tools are in `requirements-dev.txt`.

## Environment variables (set in the Render dashboard — never commit `.env`)

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | yes | — | Key for the OpenAI-compatible model endpoint. |
| `NEBULA_API_KEY` | for sorting | — | Shared secret trusted clients (TACSFON backend) send to `/sort-documents`. Generate a long random value. Without it the endpoint returns 503. |
| `NEBULA_MODEL` | no | `gpt-5.4-nano` | Model for the general agent. |
| `NEBULA_SORTER_MODEL` | no | `NEBULA_MODEL` | Model for document sorting (must accept image input for scanned pages/photos). |
| `NEBULA_LLM_BASE_URL` | no | `https://www.routeapi.ai/v1` | OpenAI-compatible base URL. |
| `NEBULA_CORS_ORIGINS` | no | — | Extra browser origins, comma-separated. Built-in: `http://localhost:3000`, `http://localhost:5173`, `https://nova-flash-psi.vercel.app`. The `*` wildcard was removed. |
| `NEBULA_DATA_DIR` | no | project dir | Directory for `nebula_memory.db` / `nebula_history.db`. Point it at a Render persistent disk mount. |
| `NEBULA_LOG_LEVEL` | no | `INFO` | Log level. |
| `NEBULA_SORT_MAX_REQUEST_MB` | no | `100` | Max total upload per sorting request. |
| `NEBULA_SORT_MAX_FILES` / `NEBULA_SORT_MAX_PAGES` | no | `200` / `250` | Batch limits. |
| `NEBULA_SORT_TIMEOUT_SECONDS` | no | `300` | Budget per sorting request. |
| `NEBULA_SORT_MODEL_TIMEOUT_SECONDS` | no | `120` | Timeout per model call. |
| `NEBULA_SORT_PAGES_PER_CALL` / `NEBULA_SORT_CONCURRENCY` | no | `4` / `4` | Page-analysis batching and parallelism. |
| `NEBULA_SORT_VISION` | no | `true` | Send page images to the model. |
| `NEBULA_SORT_OCR` | no | `true` | Use Tesseract as an extra text signal if installed (not installed on Render's native runtime; vision covers images). |
| `NEBULA_MAX_MEMORIES_PER_USER` / `NEBULA_MAX_ACTIONS_PER_USER` | no | `200` / `50` | Long-term memory growth limits. |

`GET /health/ready` reports which of these are configured (without calling the model).

## Persistence

Render's filesystem is ephemeral: SQLite files written next to the code are lost
on every deploy/restart. To keep memories and conversations, attach a Render
persistent disk and set `NEBULA_DATA_DIR` to its mount path (single instance),
or move memory/history to a managed database. Document-sorting uploads are never
persisted; they live only in a temporary directory during the request.

The database files are no longer tracked in git (`.gitignore`). Existing local
files (`nebula_*.db` and the legacy `nova_*.db`) are left in place; on start the
app copies `nova_*.db` to `nebula_*.db` only if the new file does not exist.

## Long-term memory behaviour

- One value per `(user_id, category, key)`; a newer value replaces the older one
  (e.g. `favorite_language: Python` → `C++`). Synonymous keys
  (`coding_language`, `preferred_language` …) are normalized to one canonical key,
  and legacy rows are consolidated at start-up.
- Categories are limited to `identity, preference, project, skill, goal, instruction, context`.
- The same fact under a different key refreshes the existing row instead of duplicating it.
- Per-user cap with eviction of the least important / stalest memories; the
  `actions` log is capped per user.
- Secrets (API keys, tokens, passwords, recovery answers, private keys, and keys
  named like `*_password`) are refused, both from the API and from automatic extraction.
- If the memory database is unavailable, the agent still answers (without memory).

## Tests

```
pip install -r requirements-dev.txt
pytest tests                                  # offline, no model calls
NEBULA_LIVE_TESTS=1 pytest tests/test_live_sorting.py -s   # real model, costs credits
```
