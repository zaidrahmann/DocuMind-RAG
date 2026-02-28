# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) where applicable.

## [Unreleased]

### Added

- (Placeholder for upcoming changes.)

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
