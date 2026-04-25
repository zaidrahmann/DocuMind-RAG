# Security Policy

## Supported Versions

We release patches for security issues for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.1.x   | :white_check_mark:  |
| 1.0.x   | :white_check_mark:  |
| < 1.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it responsibly:

1. **Do not** open a public GitHub issue for security-sensitive bugs.
2. **Email** the maintainers (see repository contacts or GitHub org) with:
   - A clear description of the vulnerability
   - Steps to reproduce
   - Impact and suggested fix if you have one
3. We will acknowledge receipt and work with you to understand and address the issue. We may ask for more detail or suggest a patch.
4. After a fix is released, we can coordinate on a public disclosure (e.g. release notes, CVE) if appropriate.

## Best Practices for Users

- **Secrets:** Never commit `.env` or API keys. Use environment variables or a secrets manager in production.
- **Data:** PDFs and generated indexes in `data/` and `storage/` are gitignored; do not commit sensitive documents.
- **Network:** Run the API behind HTTPS and a reverse proxy in production; restrict access to the Gradio and API ports as needed.
- **Dependencies:** Keep `requirements.txt` dependencies up to date and run `pip install -r requirements.txt` in a clean environment to avoid supply-chain issues.

Thank you for helping keep DocuMind-RAG safe for everyone.
