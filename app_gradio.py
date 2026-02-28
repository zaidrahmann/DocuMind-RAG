"""Gradio web UI for DocuMind-RAG query interface."""

from __future__ import annotations

import os
from typing import Any

import gradio as gr
import requests

def _api_base() -> str:
    """API base URL from config (DOCUMIND_API_URL)."""
    try:
        from src.config import get_settings
        return get_settings().documind_api_url
    except Exception:
        return "http://localhost:8000"


API_BASE = _api_base()
QUERY_URL = f"{API_BASE.rstrip('/')}/query"
HEALTH_URL = f"{API_BASE.rstrip('/')}/health"
STATUS_URL = f"{API_BASE.rstrip('/')}/status"
KNOWLEDGE_BASE_URL = f"{API_BASE.rstrip('/')}/knowledge-base"


def check_backend_status() -> tuple[str, str]:
    """Check if the backend API is reachable. Returns (status_emoji, status_text)."""
    try:
        r = requests.get(HEALTH_URL, timeout=3)
        if r.ok:
            try:
                s = requests.get(STATUS_URL, timeout=2)
                if s.ok:
                    data = s.json()
                    st = data.get("status", "ready")
                    doc_count = data.get("doc_count", 0)
                    if st == "no_index":
                        return "🟡", "Ready (no index — add PDFs and build or drop a PDF)"
                    if st == "ready" and doc_count >= 0:
                        return "🟢", f"Ready ({doc_count} PDF{'s' if doc_count != 1 else ''})"
                    if st == "indexing":
                        return "🟡", "Indexing…"
                    if st == "error":
                        err = (data.get("last_error") or "")[:80]
                        return "🟡", f"Error: {err}" if err else "Backend error"
            except Exception:
                pass
            return "🟢", "Ready"
        return "🟡", "Backend returned an error"
    except requests.RequestException:
        return "🔴", (
            "Backend not running. Start it with:\n"
            "`python main.py` or `uvicorn main:app --reload --port 8000`"
        )


def get_status_markdown() -> str:
    """Return status bar markdown; used for initial load, manual refresh, and polling."""
    emoji, text = check_backend_status()
    return f"**Status:** {emoji} {text}"


def fetch_knowledge_base_details() -> str:
    """Fetch knowledge base stats from API and return formatted HTML."""
    try:
        r = requests.get(KNOWLEDGE_BASE_URL, timeout=5)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException:
        return (
            "<p style='color:var(--body-text-color-subdued);'>"
            "Could not load knowledge base details. Ensure the backend is running and has an index."
            "</p>"
        )

    total_docs = data.get("total_document_count", 0)
    total_chunks = data.get("total_chunk_count", 0)
    docs = data.get("documents", [])
    strategy = data.get("chunking_strategy", "—")
    chunk_size = data.get("chunk_size", "—")
    overlap = data.get("overlap", "—")

    if not docs:
        return (
            "<p style='color:var(--body-text-color-subdued);'>"
            "No documents in the knowledge base. Add PDFs to the data folder and build the index."
            "</p>"
        )

    rows = []
    for d in docs:
        rows.append(
            f"<tr><td style='padding:6px 12px;'>{d['filename']}</td>"
            f"<td style='padding:6px 12px; text-align:center;'>{d['page_count']}</td>"
            f"<td style='padding:6px 12px; text-align:center;'>{d['chunk_count']}</td></tr>"
        )
    table_body = "\n".join(rows)

    html = f"""
<div style="font-size:0.9rem;">
  <table style="border-collapse:collapse; width:100%; margin-bottom:1rem;">
    <thead>
      <tr style="border-bottom:2px solid var(--border-color-primary);">
        <th style="padding:8px 12px; text-align:left;">Document</th>
        <th style="padding:8px 12px; text-align:center;">Pages</th>
        <th style="padding:8px 12px; text-align:center;">Chunks</th>
      </tr>
    </thead>
    <tbody>
      {table_body}
    </tbody>
  </table>
  <div style="display:grid; gap:8px; margin-top:1rem;">
    <p><strong>Total documents:</strong> {total_docs}</p>
    <p><strong>Total chunks:</strong> {total_chunks}</p>
    <p><strong>Chunking strategy:</strong> {strategy}</p>
    <p><strong>Chunk size:</strong> {chunk_size} tokens</p>
    <p><strong>Overlap:</strong> {overlap} tokens</p>
  </div>
</div>
"""
    return html


def _relevance_badge(score: float) -> tuple[str, str]:
    """Return (emoji label, hex color) based on cross-encoder score."""
    if score >= 0.5:
        return "High relevance", "#155724"
    if score >= -1.0:
        return "Good match", "#856404"
    if score >= -3.0:
        return "Partial match", "#7a3e0a"
    return "Low relevance", "#721c24"


def _relevance_bg(score: float) -> str:
    if score >= 0.5:
        return "#d4edda"
    if score >= -1.0:
        return "#fff3cd"
    if score >= -3.0:
        return "#fde8d8"
    return "#f8d7da"


def format_sources(sources: list[dict[str, Any]], scores: list[float] | None = None) -> str:
    """Format sources as styled HTML cards."""
    if not sources:
        return "<p style='color:#888; font-style:italic;'>No sources retrieved.</p>"

    cards = []
    for i, src in enumerate(sources):
        filename = src.get("filename", "Unknown document")
        page = src.get("page_number")
        preview = (src.get("text") or "")[:220].strip()
        if preview and len(src.get("text", "")) > 220:
            preview += "…"

        location = f"Page {page}" if page is not None else ""

        # Relevance badge
        badge_html = ""
        if scores and i < len(scores):
            label, color = _relevance_badge(scores[i])
            bg = _relevance_bg(scores[i])
            badge_html = (
                f'<span style="background:{bg}; color:{color}; padding:2px 10px; '
                f'border-radius:12px; font-size:0.78em; font-weight:600;">{label}</span>'
            )

        preview_html = ""
        if preview:
            preview_html = (
                f'<div style="margin-top:8px; padding:6px 10px; border-left:3px solid #ccc; '
                f'color:#555; font-size:0.85em; font-style:italic; line-height:1.5;">'
                f'{preview}</div>'
            )

        card = f"""
<div style="border:1px solid #dde3ea; border-radius:10px; padding:12px 14px;
            background:#ffffff; box-shadow:0 1px 3px rgba(0,0,0,.06);">
  <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
    <span style="font-size:1em;">📄</span>
    <span style="font-weight:700; color:#1a1a1a; font-size:0.95em;">{filename}</span>
    {f'<span style="color:#888; font-size:0.85em;">·</span><span style="color:#555; font-size:0.85em;">{location}</span>' if location else ''}
    {badge_html}
  </div>
  {preview_html}
</div>"""
        cards.append(card)

    header = f'<p style="font-size:0.8em; color:#888; margin-bottom:8px;">{len(sources)} source(s) retrieved</p>'
    return header + '<div style="display:flex; flex-direction:column; gap:8px;">' + "".join(cards) + "</div>"


def ask(question: str) -> tuple[str, str]:
    """Call FastAPI /query endpoint and return (answer, sources)."""
    if not question or not question.strip():
        return "Please enter a question above.", "_No sources._"

    try:
        resp = requests.post(
            QUERY_URL,
            json={"question": question.strip()},
            timeout=60,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        return (
            "**Backend not available**\n\n"
            "The DocuMind API server is not running. Start it first:\n\n"
            "```\npython main.py\n```\n\n"
            "or:\n\n"
            "```\nuvicorn main:app --reload --port 8000\n```\n\n"
            "Then try again."
        ), "_No sources — backend is not running._"
    except requests.exceptions.Timeout:
        return (
            "**Request timed out.** The server took too long to respond. Please try again.",
            "_No sources retrieved._",
        )
    except requests.exceptions.RequestException as e:
        resp_obj = getattr(e, "response", None)
        status = getattr(resp_obj, "status_code", None) if resp_obj else None
        detail = ""
        if resp_obj is not None:
            try:
                d = resp_obj.json()
                detail = d.get("detail", str(d))
            except Exception:
                detail = resp_obj.text or str(e)
        else:
            detail = str(e)
        err = f"**Error ({status})**" if status else "**Error**"
        return f"{err}\n\n{detail}", "_No sources retrieved._"

    try:
        data = resp.json()
    except Exception:
        return (
            f"**Invalid response** from server.\n\nRaw: {resp.text[:500]}",
            "_No sources retrieved._",
        )

    answer = data.get("answer", "")
    sources = data.get("sources", [])
    scores = data.get("scores")
    sources_text = format_sources(sources, scores)

    return answer, sources_text


# --- Gradio UI ---

_CSS = """
/* ── Reset Gradio defaults ──────────────────────────────────── */
.gradio-container {
    max-width: min(1600px, 96vw) !important;
    width: 100% !important;
    margin: 0 auto !important;
    padding: 0 1.5rem !important;
    font-family: 'Inter', system-ui, sans-serif;
}
main { max-width: 100% !important; }
footer { display: none !important; }

/* ── Header ─────────────────────────────────────────────────── */
.app-header {
    text-align: center;
    padding: 2rem 0 1.2rem;
    border-bottom: 1px solid var(--border-color-primary);
    margin-bottom: 1.4rem;
}
.app-header h1 {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin: 0 0 0.3rem;
    background: linear-gradient(135deg, #3b82f6, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.app-header p {
    font-size: 0.97rem;
    color: var(--body-text-color-subdued);
    margin: 0;
}

/* ── Status pill ────────────────────────────────────────────── */
.status-row {
    display: flex !important;
    align-items: center !important;
    background: var(--block-background-fill);
    border: 1px solid var(--border-color-primary);
    border-radius: 999px;
    padding: 6px 8px 6px 16px;
    gap: 10px;
    margin-bottom: 1.2rem;
}
.status-row > div:first-child { flex: 1; min-width: 0; }
.status-row .status-bar,
.status-row .status-bar > div {
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    min-height: 0 !important;
}
.status-row .status-bar p { margin: 0; font-size: 0.88rem; }
.status-row .status-refresh-btn button {
    border-radius: 999px !important;
    padding: 4px 14px !important;
    font-size: 0.82rem !important;
    height: 30px !important;
    min-width: 0 !important;
    width: auto !important;
    white-space: nowrap;
    flex-shrink: 0;
}
.status-row .status-refresh-btn { flex-shrink: 0; width: auto !important; min-width: 0 !important; }

/* ── Question row ───────────────────────────────────────────── */
.question-row { gap: 12px !important; align-items: flex-end !important; }
.question-row .ask-btn button {
    height: 100% !important;
    min-height: 64px !important;
    border-radius: 10px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.2px;
}

/* ── Panels ─────────────────────────────────────────────────── */
.panel-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--body-text-color-subdued);
    margin-bottom: 6px;
}
.answer-box textarea {
    font-size: 0.95rem !important;
    line-height: 1.7 !important;
    border-radius: 10px !important;
}

/* ── Technical accordion ────────────────────────────────────── */
.tech-accordion { margin-top: 1.2rem; border-radius: 10px !important; }
.tech-accordion p { font-size: 0.82rem; color: var(--body-text-color-subdued); }
"""

with gr.Blocks(title="DocuMind – Enterprise RAG System") as demo:

    # ── Header ──────────────────────────────────────────────────────
    gr.HTML("""
    <div class="app-header">
      <h1>⚡ DocuMind</h1>
      <p>Enterprise RAG · Ask your documents anything · Answers with source citations</p>
    </div>
    """)

    # ── Status pill ─────────────────────────────────────────────────
    emoji, _ = check_backend_status()
    with gr.Row(elem_classes=["status-row"]):
        status_bar = gr.Markdown(
            get_status_markdown(),
            elem_classes="status-bar",
        )
        refresh_btn = gr.Button(
            "↻ Refresh",
            size="sm",
            variant="secondary",
            elem_classes="status-refresh-btn",
        )
    refresh_btn.click(fn=get_status_markdown, outputs=[status_bar])

    # ── Question input ───────────────────────────────────────────────
    with gr.Row(elem_classes=["question-row"]):
        question_box = gr.Textbox(
            label="Your question",
            placeholder="e.g.  What are the key findings?  ·  Summarise section 3  ·  Compare X vs Y…",
            lines=2,
            max_lines=6,
            scale=9,
        )
        ask_btn = gr.Button("Ask ➜", variant="primary", scale=1, elem_classes="ask-btn")

    # ── Answer + Sources ─────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=5):
            gr.HTML('<p class="panel-label">Answer</p>')
            answer_box = gr.Textbox(
                label="",
                placeholder="Your answer will appear here…",
                lines=12,
                max_lines=20,
                interactive=False,
                elem_classes="answer-box",
            )
        with gr.Column(scale=4):
            gr.HTML('<p class="panel-label">Sources</p>')
            sources_box = gr.HTML(
                "<p style='color:var(--body-text-color-subdued); font-size:0.88rem; "
                "font-style:italic; margin-top:4px;'>Retrieved sources will appear here after each query.</p>"
            )

    ask_btn.click(fn=ask, inputs=question_box, outputs=[answer_box, sources_box])
    question_box.submit(fn=ask, inputs=question_box, outputs=[answer_box, sources_box])

    # ── Knowledge base details ───────────────────────────────────────
    with gr.Accordion("📚 Knowledge base", open=False, elem_classes="tech-accordion"):
        kb_details = gr.HTML(
            "<p style='color:var(--body-text-color-subdued); font-size:0.88rem;'>"
            "Click <strong>View details</strong> to load document counts, chunking config, and more.</p>"
        )
        kb_btn = gr.Button("View details", size="sm")
    kb_btn.click(fn=fetch_knowledge_base_details, outputs=[kb_details])

    # ── Technical details ────────────────────────────────────────────
    with gr.Accordion("⚙ Technical details", open=False, elem_classes="tech-accordion"):
        gr.Markdown(
            f"**API endpoint:** `{QUERY_URL}`  \n"
            "Override the base URL with the `DOCUMIND_API_URL` environment variable."
        )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"), css=_CSS)
