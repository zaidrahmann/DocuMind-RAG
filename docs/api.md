# API Reference

DocuMind-RAG exposes a REST API for health checks, index status, and RAG queries. The OpenAPI (Swagger) schema is available at `/docs` when the server is running.

## Base URL

Default: `http://localhost:8000`. Override with `DOCUMIND_API_URL` (used by the Gradio UI).

## Endpoints

### Operations

#### `GET /health`

Liveness check.

**Response:** `200 OK`

```json
{ "status": "ok" }
```

---

#### `GET /status`

Index and watcher status. Use this to see document count, chunk count, last indexed time, and any recent error.

**Response:** `200 OK`

```json
{
  "status": "ready",
  "doc_count": 5,
  "chunk_count": 42,
  "embedding_dim": 384,
  "last_indexed": "2025-02-28T12:00:00",
  "watching_dir": "C:\\path\\to\\data\\raw_pdfs",
  "last_error": ""
}
```

| Field | Description |
|-------|-------------|
| `status` | `initializing` \| `ready` \| `indexing` \| `error` |
| `doc_count` | Number of source documents (by filename) in the current index |
| `chunk_count` | Number of indexed chunks |
| `embedding_dim` | Vector dimension |
| `last_indexed` | ISO-8601 UTC timestamp of last successful index build |
| `watching_dir` | Directory watched for PDF changes (hot-reload) |
| `last_error` | Last indexing error message (empty if none) |

---

### RAG

#### `POST /ask`

Ask a question; returns answer and sources only.

**Request:** `application/json`

```json
{ "question": "Your question here" }
```

- `question`: Non-empty string, max length per `DOCUMIND_MAX_QUESTION_LENGTH` (default 4096).

**Response:** `200 OK`

```json
{
  "answer": "Generated answer text.",
  "sources": [
    { "source": "document.pdf", "page": 1, "text": "..." }
  ]
}
```

**Errors:**

- `400` — Invalid request (e.g. empty or too long question).
- `503` — Pipeline unavailable (e.g. index not loaded or generation failed). Response body does not expose internal details.

---

#### `POST /query`

Same as `/ask`, but the response includes similarity scores for each source.

**Request:** Same as `/ask`.

**Response:** `200 OK`

```json
{
  "answer": "Generated answer text.",
  "sources": [ { "source": "document.pdf", "page": 1, "text": "..." } ],
  "scores": [ 0.92, 0.88, ... ]
}
```

**Errors:** Same as `/ask`.

---

## Interactive Docs

- **Swagger UI:** `GET http://localhost:8000/docs`
- **ReDoc:** `GET http://localhost:8000/redoc` (if enabled)

Use these to explore request/response schemas and try requests from the browser.

## Root Redirect

`GET /` redirects to `/docs`.
