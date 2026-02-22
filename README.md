# DocuMind-RAG

**Production-ready Retrieval-Augmented Generation (RAG) pipeline for document Q&A.** Ask questions about your PDFs and get AI-generated answers with source citations.

---

## Features

- **PDF ingestion** — Load PDFs from a directory, extract text per page
- **Token-based chunking** — Tiktoken-based splitting with configurable chunk size and overlap
- **Multilingual embeddings** — Sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2`) for semantic search
- **FAISS vector store** — Fast similarity search with IndexFlatIP for cosine similarity
- **Flexible LLM backend** — **Ollama** (local, no API key) or **HuggingFace Inference API** (cloud)
- **FastAPI** — REST API with `/query` and `/health` endpoints
- **Gradio UI** — Web interface for interactive Q&A
- **Source citations** — Every answer includes filename, page number, and similarity scores

---

## Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│  PDFs           │───▶│  Load &      │───▶│  Chunk          │
│  data/raw_pdfs/ │    │  Extract     │    │  (tiktoken)     │
└─────────────────┘    └──────────────┘    └────────┬────────┘
                                                    │
                                                    ▼
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│  Answer +       │◀───│  LLM         │◀───│  Embed &        │
│  Sources        │    │  (Ollama/HF) │    │  Index (FAISS)  │
└─────────────────┘    └──────────────┘    └─────────────────┘
         ▲                        ▲
         │                        │
         └────────────────────────┘
              Query flow
```

**Build-time:** PDFs → Load → Chunk → Embed → FAISS index  
**Query-time:** Question → Embed → Retrieve top-k → Combine context → Generate answer

---

## Prerequisites

- **Python 3.10+**
- **Ollama** (recommended for local LLM) — [ollama.com](https://ollama.com)  
  Or **HuggingFace API key** (for cloud inference)
- For Ollama: `ollama run llama3` (or your preferred model)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/DocuMind-RAG.git
cd DocuMind-RAG

# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Copy environment template and configure
copy .env.example .env
# Edit .env with your HF_API_KEY (if using HuggingFace) or leave DOCUMIND_GENERATOR=ollama
```

---

## Configuration

Copy `.env.example` to `.env` and configure:

| Variable | Description | Default |
|----------|-------------|---------|
| `DOCUMIND_GENERATOR` | `ollama` or `hf` | `ollama` |
| `HF_API_KEY` | HuggingFace token (required for `hf`) | — |
| `OLLAMA_URL` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | Ollama model name | `llama3` |
| `HF_MODEL` | HuggingFace model (when `hf`) | `mistralai/Mistral-7B-Instruct-v0.2` |
| `DOCUMIND_API_URL` | API base URL for Gradio UI | `http://localhost:8000` |

---

## Quick Start

### 1. Build the vector index

Place PDFs in `data/raw_pdfs/`, then:

```bash
python build_index.py --pdf-dir data/raw_pdfs --output storage/doc_index.index
```

Options:
- `--chunk-size` — Tokens per chunk (default: 512)
- `--overlap` — Overlap between chunks (default: 64)
- `--verbose` — Debug logging

### 2. Start the API server

```bash
python main.py
```

Or with uvicorn:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API runs at `http://localhost:8000`. Docs: `http://localhost:8000/docs`

### 3. Launch the Gradio UI (optional)

```bash
python app_gradio.py
```

Opens a web interface to query your documents. Ensure the API is running first.

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check; returns `{"status": "ok"}` |
| `/query` | POST | Query the RAG pipeline |

**POST /query** — Request body:
```json
{
  "question": "What is the main topic of the document?"
}
```

Response:
```json
{
  "answer": "The document discusses...",
  "sources": [
    {"filename": "paper.pdf", "page_number": 3, "chunk_index": 1}
  ],
  "scores": [0.89, 0.85, ...]
}
```

---

## Project Structure

```
DocuMind-RAG/
├── main.py              # FastAPI app
├── app_gradio.py        # Gradio UI
├── build_index.py       # Index build script
├── requirements.txt
├── .env.example
├── src/
│   ├── pipeline.py      # Full RAG pipeline (retrieve + generate)
│   ├── embeddings/      # MultilingualEmbedder
│   ├── ingestion/       # PDF loader, chunker
│   ├── vectorstore/     # FAISS store
│   ├── retrieval/       # Retriever, RAG retrieval pipeline
│   └── generation/      # OllamaGenerator, HFGenerator
├── data/raw_pdfs/       # PDF input directory
├── storage/             # FAISS index output
├── scripts/             # check_ollama.py, check_hf_connection.py
└── tests/
```

---

## LLM Backends

### Ollama (default, local)

- No API key required
- Runs entirely on your machine
- Test connectivity: `python scripts/check_ollama.py`

### HuggingFace Inference API

- Requires `HF_API_KEY` in `.env`
- Set `DOCUMIND_GENERATOR=hf`
- Test connectivity: `python scripts/check_hf_connection.py`

---

## Tests

```bash
pytest tests/ -v
```

---

## License

See project license file.
