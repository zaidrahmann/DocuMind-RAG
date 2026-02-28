# Architecture

This document describes the design and data flow of DocuMind-RAG.

## Overview

DocuMind-RAG is a Retrieval-Augmented Generation (RAG) pipeline that:

1. Ingests PDF documents from a directory
2. Chunks and embeds them into a vector index
3. Answers user questions by retrieving relevant chunks and generating answers with an LLM

## High-Level Data Flow

```
                    BUILD TIME
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────────┐
│  PDF files  │───▶│ Load & extract│───▶│   Chunk      │───▶│   Embed      │───▶│ FAISS index  │
│ (raw_pdfs/) │    │   (pypdf)     │    │ (tiktoken)   │    │ (sentence-   │    │ + meta JSON  │
└─────────────┘    └──────────────┘    └─────────────┘    │  transformers)│    └──────────────┘
                                                         └─────────────┘

                    QUERY TIME
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────────┐
│  Question   │───▶│ Embed query   │───▶│ FAISS search │───▶│  Rerank      │───▶│ LLM generate  │
│             │    │              │    │ (top-20)     │    │ (top-5)      │    │ (Ollama/HF)   │
└─────────────┘    └──────────────┘    └─────────────┘    └─────────────┘    └──────────────┘
                                                                                    │
                                                                                    ▼
                                                                            Answer + sources
```

## Components

### Ingestion (`src/ingestion/`)

- **Loader:** Reads PDFs from a directory, extracts text per page using pypdf. Produces `Document` objects (content + metadata: filename, page number).
- **Chunker:** Splits documents into token-based chunks using tiktoken (encoding: `cl100k_base`). Configurable chunk size and overlap. Preserves metadata and adds `chunk_index`, `token_count`.

### Embeddings (`src/embeddings/`)

- **MultilingualEmbedder:** Wraps a sentence-transformers model (default: `paraphrase-multilingual-MiniLM-L12-v2`). Singleton. Encodes documents in batches (with optional progress bar) and single queries. Outputs L2-normalized vectors for cosine similarity via dot product.

### Vector Store (`src/vectorstore/`)

- **FaissVectorStore:** FAISS `IndexFlatIP` (exact search, inner product on normalized vectors = cosine similarity). Stores vectors and metadata separately; metadata (including chunk text) is persisted as JSON (`.meta.json`) alongside the binary index (`.index`). Supports add, search, save, load.

### Retrieval (`src/retrieval/`)

- **Retriever:** Embeds the query and runs FAISS search.
- **Reranker:** Cross-encoder (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`) reranks (query, chunk) pairs for better precision.
- **RAG pipeline (retrieval only):** Loads index from disk, retrieves top-k (e.g. 20), optionally reranks and keeps top-5, builds a context string, returns `RAGResult` (context, sources, scores).

### Generation (`src/generation/`)

- **OllamaGenerator:** Local LLM via Ollama API (default model: `llama3`).
- **HFGenerator:** HuggingFace Inference API via `InferenceClient` (default model: `Qwen/Qwen2.5-7B-Instruct`). Retry logic for 503/model loading; user-friendly error messages for auth/network issues.

### Pipeline (`src/`)

- **Indexer (`indexer.py`):** Shared `build_index()`: load PDFs → chunk → embed → build FAISS → save. Returns `IndexStats`. Used by `build_index.py` (CLI) and by the hot-reload watcher in `main.py`.
- **RAGPipeline (`pipeline.py`):** Loads FAISS index, runs retrieval (with optional reranking), selects generator from config (Ollama or HF), returns `AskResult` (answer, sources, scores).

### API and UI

- **main.py:** FastAPI app. Endpoints: `/health`, `/status`, `/knowledge-base`, `/ask`, `/query`. Lifespan: load RAG pipeline from index (or start with `no_index` if missing), start watchdog on `data/raw_pdfs/`, on shutdown stop observer and clear pipeline. Hot-reload: on PDF create, modify, or **delete** (with debounce), rebuild index and swap pipeline atomically. `/status` exposes index stats; `/knowledge-base` exposes per-document breakdown and chunking config.
- **app_gradio.py:** Gradio UI that calls the API for questions and displays answers with source citations. Includes a knowledge base details panel showing document counts and chunking configuration.

### CLI

- **documind/__main__.py:** Entry point for `python -m documind ask "..."`. Loads pipeline from index, runs ask, prints answer and formatted sources.

## Concurrency and Hot-Reload

- Rebuilds run under an `asyncio.Lock` so only one rebuild runs at a time.
- The pipeline reference is swapped after a successful rebuild; in-flight requests keep using the old pipeline; new requests use the new one (atomic pointer swap under the GIL).
- Watchdog uses a 5-second cooldown to debounce rapid file events.

## File Layout (runtime)

| Path | Purpose |
|------|---------|
| `data/raw_pdfs/` | Input PDFs (watched for hot-reload: create, modify, delete) |
| `storage/doc_index.index` | FAISS binary index |
| `storage/doc_index.meta.json` | Chunk metadata (and text) for retrieval |
| `storage/doc_index.build.json` | Build config (chunk_size, overlap, strategy) for `/knowledge-base` |

Configuration is centralized in `src/config.py` (Pydantic Settings from environment and `.env`).
