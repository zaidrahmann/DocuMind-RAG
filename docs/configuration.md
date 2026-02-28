# Configuration Reference

All runtime behavior is driven by environment variables and optional `.env` file. Settings are loaded via Pydantic Settings in `src/config.py`.

## Loading Configuration

- **Source order:** Environment variables override `.env` values.
- **File:** Copy `.env.example` to `.env` and set values as needed. Do not commit `.env` (it is in `.gitignore`).

## Variables

### Paths

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DOCUMIND_INDEX_PATH` | path | `storage/doc_index.index` | Path to the FAISS index file. The matching metadata file must exist as the same path with `.meta.json` suffix. |
| `DOCUMIND_PDF_DIR` | path | `data/raw_pdfs` | Directory containing PDFs. Used for building the index and as the watch directory for hot-reload. |

### Generator (LLM)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DOCUMIND_GENERATOR` | string | `ollama` | Backend: `ollama` (local) or `hf` (HuggingFace Inference API). |

### Ollama (when `DOCUMIND_GENERATOR=ollama`)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `OLLAMA_URL` | string | `http://localhost:11434` | Ollama server URL. |
| `OLLAMA_MODEL` | string | `llama3` | Model name (e.g. `llama3`, `mistral`). |

### HuggingFace (when `DOCUMIND_GENERATOR=hf`)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `HF_API_KEY` | string | — | HuggingFace token. Required for `hf`. Create at https://huggingface.co/settings/tokens (Read or Inference). |
| `HF_MODEL` | string | `Qwen/Qwen2.5-7B-Instruct` | Model ID for chat completion. |

### Reranking

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DOCUMIND_USE_RERANKER` | bool | `true` | Whether to use cross-encoder reranking (retrieve more, rerank, keep top-5). |
| `DOCUMIND_RERANKER_MODEL` | string | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Sentence-transformers cross-encoder model name. |

### API / Server

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DOCUMIND_API_URL` | string | `http://localhost:8000` | Base URL of the DocuMind API. Used by the Gradio UI to call the backend. |
| `DOCUMIND_LOG_LEVEL` | string | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`. |

### Validation Limits

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DOCUMIND_MAX_QUESTION_LENGTH` | int | `4096` | Maximum allowed character length for the question field (1–65536). |
| `DOCUMIND_MAX_TOP_K` | int | `50` | Maximum retrieval top-k (1–100). |

## Example `.env` Snippets

**Ollama (default):**

```env
DOCUMIND_GENERATOR=ollama
OLLAMA_MODEL=llama3
```

**HuggingFace:**

```env
DOCUMIND_GENERATOR=hf
HF_API_KEY=hf_xxxxxxxxxxxx
HF_MODEL=Qwen/Qwen2.5-7B-Instruct
```

**Custom paths and debug logging:**

```env
DOCUMIND_INDEX_PATH=storage/doc_index.index
DOCUMIND_PDF_DIR=data/raw_pdfs
DOCUMIND_LOG_LEVEL=DEBUG
```

## Code Access

Use `get_settings()` from `src.config` to obtain the singleton `Settings` instance. The `index_path` property is an alias for `documind_index_path`.
