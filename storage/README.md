# Storage Directory

This directory holds **generated indexes** and related artifacts. Its contents are not committed to version control.

## Structure

```
storage/
├── README.md             (this file — tracked)
├── .gitkeep              (keeps directory in git when empty)
├── doc_index.index       (FAISS binary index — gitignored)
├── doc_index.meta.json   (chunk metadata and text — gitignored)
└── doc_index.build.json  (build config: chunk_size, overlap — gitignored)
```

## Generated Files

| File | Description |
|------|-------------|
| `doc_index.index` | FAISS vector index (binary). Created by `build_index.py` or by the hot-reload watcher. |
| `doc_index.meta.json` | Chunk metadata (filename, page, chunk index, etc.) and chunk text. Same base path as the index with `.meta.json` suffix. |
| `doc_index.build.json` | Build configuration (chunk_size, overlap, strategy). Used by the `/knowledge-base` API and Gradio UI. Created on each index build. |

The default index path is controlled by `DOCUMIND_INDEX_PATH` (default: `storage/doc_index.index`). Both the `.index` and `.meta.json` files must exist for the RAG pipeline to load.

## Why These Are Gitignored

- **Size:** Indexes can be large and binary; they are environment-specific.
- **Reproducibility:** Indexes are rebuilt from source PDFs in `data/raw_pdfs/`. There is no need to version them.
- **Sensitivity:** Index content reflects your documents; keeping them out of the repo avoids accidental exposure.

To get a working index after cloning, add PDFs to `data/raw_pdfs/` and run:

```bash
python build_index.py --pdf-dir data/raw_pdfs --output storage/doc_index.index
```

Or start the server and rely on hot-reload after adding PDFs.
