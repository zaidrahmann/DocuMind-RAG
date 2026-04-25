# Development Guide

This document covers local setup, running tests, and code conventions for contributing to DocuMind-RAG.

## Prerequisites

- **Python:** 3.10 or higher
- **Ollama (optional):** For local LLM; install from [ollama.com](https://ollama.com) and run e.g. `ollama run llama3`
- **HuggingFace (optional):** For cloud inference; set `HF_API_KEY` in `.env`

## Local Setup

1. Clone the repository and create a virtual environment:

   ```bash
   git clone https://github.com/your-org/DocuMind-RAG.git
   cd DocuMind-RAG
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   # source .venv/bin/activate   # macOS/Linux
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and set any variables (e.g. `HF_API_KEY` if using HuggingFace).

4. Add PDFs to `data/raw_pdfs/` and build the index:

   ```bash
   python build_index.py --pdf-dir data/raw_pdfs --output storage/doc_index.index
   ```

5. Run the API server:

   ```bash
   python main.py
   ```

   Use `http://localhost:8000/docs` for the API UI. For the Gradio UI run `python app_gradio.py` in another terminal.

   If port 8000 is already in use, set `DOCUMIND_PORT=8001` in `.env` (and `DOCUMIND_API_URL=http://localhost:8001` for the Gradio UI).

## Running Tests

From the project root:

```bash
# All tests
pytest tests/ -v

# Or use the project test runner
python -m tests.run_all
```

See [tests/README.md](../tests/README.md) for per-module test descriptions and requirements (e.g. network for first-time model/encoding downloads).

## Linting and Formatting

- **Lint:** `ruff check src documind main.py build_index.py app_gradio.py tests`
- **Format:** `ruff format src documind main.py build_index.py app_gradio.py tests`

Configuration lives in `pyproject.toml`.

## Type Checking

```bash
mypy src documind
```

Install mypy and dev dependencies: `pip install mypy` (or `pip install -e ".[dev]"` for pytest, ruff, mypy). The project uses `types-requests` for request stubs; ensure all deps from `requirements.txt` are installed.

## Project Layout

- **`src/`** — Core library: config, ingestion, embeddings, vectorstore, retrieval, generation, pipeline, indexer.
- **`documind/`** — CLI entry point (`python -m documind`).
- **`main.py`** — FastAPI app and hot-reload watcher.
- **`build_index.py`** — CLI to build the FAISS index.
- **`app_gradio.py`** — Gradio web UI.
- **`tests/`** — Unit and integration tests.
- **`docs/`** — Additional documentation.

Configuration is centralized in `src/config.py`; add new options there and document them in [docs/configuration.md](configuration.md) and in `.env.example`.

## Hot-Reload and Watchdog

The server watches `data/raw_pdfs/` for new or changed PDFs and rebuilds the index automatically. Do not run uvicorn with `--reload` when using the file watcher; the reloader would restart the process and kill the observer. For development, restart the server manually after code changes.

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for branch workflow, commit messages, and pull request process.
