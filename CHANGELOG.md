# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) where applicable.

## [Unreleased]

### Added

- **`GET /knowledge-base` endpoint:** Returns detailed index statistics: per-document breakdown (filename, page count, chunk count), total chunk count, and chunking configuration (strategy, chunk size, overlap).
- **Gradio knowledge base panel:** "Knowledge base" accordion in the UI with a "View details" button to inspect document counts and chunking settings.
- **`DOCUMIND_PORT`:** Configurable API server port (default 8000). Use when port 8000 is already in use; set `DOCUMIND_API_URL` to match for the Gradio UI.
- **Build metadata:** Index builds now write `storage/doc_index.build.json` with chunk_size, overlap, and strategy for the knowledge-base endpoint.

### Changed

- **`doc_count` semantics:** Status and `/status` now report the number of **source PDF files** (unique filenames), not page count. Previously reported pages (e.g. 22) instead of PDFs (e.g. 3).
- **Hot-reload on delete:** Removing a PDF from `data/raw_pdfs/` triggers an index rebuild (previously only create/modify did).
- **Server startup:** API server starts even when the index file is missing; status shows `no_index` and queries return 503 until an index is built.
- **Gradio UI:** Improved layout, status pill with integrated refresh button, wider container, Gradio 6.0 compatibility (CSS moved to `launch()`).

### Fixed

- **Exception chaining:** Re-raised exceptions in `/ask` and `/query` now use `raise ... from e` for proper traceback linking (mypy/CI compliance).
- **Type annotations:** Added type guards and annotations for mypy (faiss_store, loader, reranker, embedder, indexer). Added `types-requests` for ollama_generator.

## [1.1.0] - 2025-02-28

### Added

- **Hot-reload indexing:** Server watches `data/raw_pdfs/` for new or modified PDFs and rebuilds the FAISS index automatically (no restart). Uses `watchdog` with a 5-second debounce.
- **`GET /status` endpoint:** Returns index stats (`doc_count`, `chunk_count`, `embedding_dim`, `last_indexed`, `watching_dir`, `last_error`) and pipeline status.
- **Shared indexer:** `src/indexer.py` provides `build_index()` used by both `build_index.py` and the hot-reload watcher for consistent index builds.
- **Configuration:** `DOCUMIND_PDF_DIR`, `DOCUMIND_MAX_QUESTION_LENGTH`, `DOCUMIND_MAX_TOP_K` for paths and validation limits.
- **Documentation:** `docs/` (architecture, configuration, API, development), CONTRIBUTING.md, CHANGELOG.md, SECURITY.md, READMEs for `data/` and `storage/`.
- **Gitignore:** PDFs, storage indexes, and document data excluded from version control; `.gitkeep` in `data/raw_pdfs/` and `storage/` to preserve directory structure.

### Changed

- **HuggingFace default model:** From `mistralai/Mistral-7B-Instruct-v0.2` to `Qwen/Qwen2.5-7B-Instruct`.
- **API description:** FastAPI app description updated to mention hot-reload.

### Fixed

- (None in this release.)

## [1.0.0] - (initial)

### Added

- RAG pipeline: PDF ingestion, tiktoken chunking, multilingual embeddings, FAISS vector store, cross-encoder reranking, Ollama and HuggingFace generators.
- CLI: `python -m documind ask "..."`.
- REST API: `/health`, `/ask`, `/query`.
- Gradio UI.
- Centralized config via Pydantic Settings (`src/config.py`).
- Custom exceptions and logging setup.
- Test suite for loader, chunker, embedder, FAISS store, and full pipeline.

---

[Unreleased]: https://github.com/your-org/DocuMind-RAG/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/your-org/DocuMind-RAG/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/your-org/DocuMind-RAG/releases/tag/v1.0.0
