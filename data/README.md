# Data Directory

This directory holds **input data** for DocuMind-RAG. Its contents are not committed to version control.

## Structure

```
data/
├── README.md          (this file — tracked)
└── raw_pdfs/          (PDF input — contents gitignored)
    └── .gitkeep       (keeps directory in git when empty)
```

## `raw_pdfs/`

Place your PDF documents here. They are used when:

- **Building the index:** `python build_index.py --pdf-dir data/raw_pdfs --output storage/doc_index.index`
- **Hot-reload:** When the API server is running (`python main.py`), it watches this directory. Adding, changing, or **deleting** a PDF triggers an automatic index rebuild (after a short debounce).

### Notes

- Only `.pdf` files are processed; other files are ignored.
- Corrupted or unreadable PDFs are skipped with a warning in the logs.
- **Do not commit PDFs to GitHub.** The path `data/raw_pdfs/*` is in `.gitignore` to avoid pushing document data. Use `.gitkeep` only to keep the empty directory in the repo.

## Other Data

If you add other input formats or directories (e.g. `data/documents/`), ensure sensitive or large files are listed in `.gitignore` and document the layout here.
