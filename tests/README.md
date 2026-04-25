# DocuMind-RAG Tests

Scripts to verify that ingestion, embeddings, and vector store components work correctly.

## Quick Run

From the project root:

```bash
# Run all tests (chunker, loader, FAISS, embedder, full pipeline)
python -m tests.run_all

# Run individual test modules
python tests/test_chunker.py
python tests/test_loader.py
python tests/test_faiss_store.py
python tests/test_embedder.py
python tests/test_full_pipeline.py
```

## Test Overview

| Script | What it tests |
|--------|---------------|
| `test_chunker.py` | Token-based document chunking (uses tiktoken; first run downloads encoding) |
| `test_loader.py` | PDF loading from directory (creates a blank sample PDF in temp dir) |
| `test_faiss_store.py` | FAISS index add/search/save/load |
| `test_embedder.py` | Multilingual embedder encode_documents/encode_query (downloads model on first run) |
| `test_full_pipeline.py` | End-to-end: chunk → embed → FAISS → search |

## Requirements

- All dependencies from `requirements.txt` installed
- **Chunker** / **Full pipeline**: Network access for tiktoken to download encoding on first run
- **Embedder** / **Full pipeline**: Network access to download sentence-transformers model (~400MB) on first run

## Optional: pytest

You can also run with pytest (if installed):

```bash
pytest tests/ -v
```
