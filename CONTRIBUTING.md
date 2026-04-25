# Contributing to DocuMind-RAG

Thank you for your interest in contributing. This document explains how to propose changes and work with the codebase.

## Code of Conduct

By participating, you are expected to be respectful and constructive. Report unacceptable behavior via the project’s issue tracker or [SECURITY.md](SECURITY.md) if it involves security.

## How to Contribute

### Reporting Bugs and Suggesting Features

- Open an issue describing the bug or feature.
- For bugs: include steps to reproduce, expected vs actual behavior, and your environment (OS, Python version).
- For features: describe the use case and, if possible, a proposed design.

### Pull Requests

1. **Fork** the repository and clone your fork.
2. **Create a branch** from `main` (or `master`):  
   `git checkout -b fix/short-description` or `feature/short-description`.
3. **Make your changes.** Follow the project’s style (see below).
4. **Run tests and lint:**
   - `pytest tests/ -v`
   - `ruff check src documind main.py build_index.py app_gradio.py tests`
   - `ruff format src documind main.py build_index.py app_gradio.py tests`
   - `mypy src documind` (if mypy is set up).
5. **Commit** with clear, concise messages (e.g. “Add retry backoff for HF inference”, “Fix chunk metadata for empty pages”).
6. **Push** to your fork and open a **Pull Request** against the upstream `main` (or `master`).
7. **Address review feedback** if requested. CI must pass (lint, typecheck, tests).

### What to Document

- New or changed configuration: update `src/config.py`, [docs/configuration.md](docs/configuration.md), and `.env.example`.
- New endpoints or behavior: update [docs/api.md](docs/api.md) and the main [README.md](README.md) if it affects quick start or features.
- Architectural changes: update [docs/architecture.md](docs/architecture.md).

## Development Setup

See [docs/development.md](docs/development.md) for local setup, testing, and conventions.

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see the repository’s LICENSE file).
