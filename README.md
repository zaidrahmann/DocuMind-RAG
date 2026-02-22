# DocuMind-RAG

**RAG over your PDFs: ask questions, get answers with sources.**

Production-ready Retrieval-Augmented Generation pipeline. Point it at your documents, build an index once, then query via CLI or API—with cited sources every time.

---

## Proof it works

Run one command and get an answer from your indexed docs:

```bash
$ python -m documind ask "What is the conference format?"
```

**Example output:**

```
The conference format typically follows a standard academic structure: submissions are 
peer-reviewed, accepted papers are presented in oral or poster sessions, and proceedings 
are published. Specific guidelines (page limits, formatting, deadlines) are defined in 
the call for papers.

--- Sources ---
  1. AAAI conference format (1).pdf (page 1)
  2. AAAI conference format (1).pdf (page 2)
```

Same via API—one request, answer + sources:

```bash
$ curl -s -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d "{\"question\": \"What is the conference format?\"}"
```

```json
{"answer":"The conference format typically follows...","sources":[{"source":"AAAI conference format (1).pdf","page":1,...}]}
```

---

## How to run (5 steps)

1. **Clone and enter the repo**
   ```bash
   git clone https://github.com/your-org/DocuMind-RAG.git
   cd DocuMind-RAG
   ```

2. **Create a virtual environment and install**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # Windows
   # source .venv/bin/activate     # macOS/Linux
   pip install -r requirements.txt
   ```

3. **Configure (optional)**  
   Copy `.env.example` to `.env`. Default is **Ollama** (local, no API key). For HuggingFace, set `DOCUMIND_GENERATOR=hf` and `HF_API_KEY`.

4. **Build the index** (put your PDFs in `data/raw_pdfs/` first)
   ```bash
   python build_index.py --pdf-dir data/raw_pdfs --output storage/doc_index.index
   ```

5. **Ask a question**
   - **CLI:** `python -m documind ask "Your question here"`
   - **API:** Start the server with `python main.py`, then `curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"Your question here\"}"`

That’s it. No need to know Python or where the index lives—one command or one curl and you get an answer with sources.

---

## What you get

| Feature | Description |
|--------|-------------|
| **One-command CLI** | `python -m documind ask "..."` — answer and sources printed to the terminal |
| **One-request API** | `POST /ask` with `{"question": "..."}` → `{"answer": "...", "sources": [...]}` |
| **PDF ingestion** | Load from a directory, extract text per page, chunk with tiktoken |
| **Semantic search** | Multilingual embeddings (sentence-transformers) + FAISS vector store |
| **LLM flexibility** | Ollama (local, default) or HuggingFace Inference API |
| **Source citations** | Every answer includes document and page references |
| **REST + UI** | FastAPI with `/health`, `/query`, `/ask`; optional Gradio UI |
| **Reranking** | Retrieve top-20, rerank with a cross-encoder, pass top-5 to the LLM—better precision than “vector search only” |

**Beyond tutorial RAG:** retrieval is not just “embed and search.” We retrieve more candidates (e.g. 20), then rerank (query, chunk) pairs with a small cross-encoder and pass only the top-5 to the LLM. That improves answer quality without changing the rest of the stack.

---

## Architecture

```
PDFs → Load & extract → Chunk (tiktoken) → Embed → FAISS index
                                                      ↓
Answer + sources ← Generate (Ollama/HF) ← Rerank (cross-encoder) ← Retrieve top-20 ← Question
```

- **Build-time:** PDFs in `data/raw_pdfs/` → index in `storage/doc_index.index`
- **Query-time:** Question → embed → similarity search → context → LLM → answer + sources

---

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `DOCUMIND_GENERATOR` | `ollama` or `hf` | `ollama` |
| `OLLAMA_URL` | Ollama server | `http://localhost:11434` |
| `OLLAMA_MODEL` | Model name | `llama3` |
| `HF_API_KEY` | HuggingFace token (for `hf`) | — |
| `HF_MODEL` | HuggingFace model | `mistralai/Mistral-7B-Instruct-v0.2` |
| `DOCUMIND_USE_RERANKER` | Use cross-encoder reranking (`true` / `false`) | `true` |
| `DOCUMIND_RERANKER_MODEL` | Cross-encoder model for reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |

**Prerequisites:** Python 3.10+. For Ollama: [ollama.com](https://ollama.com), then `ollama run llama3`.

---

## API reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | `{"status": "ok"}` |
| `/ask` | POST | Body: `{"question": "..."}` → `{"answer": "...", "sources": [...]}` |
| `/query` | POST | Same as `/ask` plus similarity `scores` in the response |

Interactive docs: `http://localhost:8000/docs` when the server is running.

---

## Project structure

```
DocuMind-RAG/
├── main.py              # FastAPI app
├── app_gradio.py        # Gradio UI (optional)
├── build_index.py       # Index build script
├── documind/            # CLI: python -m documind ask "..."
├── src/
│   ├── pipeline.py      # Full RAG pipeline (retrieve + generate)
│   ├── embeddings/      # MultilingualEmbedder
│   ├── ingestion/      # PDF loader, chunker
│   ├── vectorstore/    # FAISS store
│   ├── retrieval/      # Retriever, Reranker, RAG pipeline
│   └── generation/     # OllamaGenerator, HFGenerator
├── data/raw_pdfs/       # PDF input
├── storage/             # FAISS index output
└── tests/
```

---

## Tests

```bash
pytest tests/ -v
```

---

## License

See project license file.
