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


def check_backend_status() -> tuple[str, str]:
    """Check if the backend API is reachable. Returns (status_emoji, status_text)."""
    try:
        r = requests.get(HEALTH_URL, timeout=3)
        if r.ok:
            return "🟢", "Ready"
        return "🟡", "Backend returned an error"
    except requests.RequestException:
        return "🔴", (
            "Backend not running. Start it with:\n"
            "`python main.py` or `uvicorn main:app --reload --port 8000`"
        )


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

with gr.Blocks(title="DocuMind – Enterprise RAG System") as demo:
    gr.Markdown(
        "# DocuMind – Enterprise RAG System\n"
        "Ask questions about your documents. Answers are generated using AI with source citations.",
        elem_classes="title-wrap",
    )

    emoji, status_text = check_backend_status()
    status_bar = gr.Markdown(
        f"**Status:** {emoji} {status_text}",
        elem_classes="status-bar ready" if emoji == "🟢" else "status-bar",
    )

    with gr.Row():
        question_box = gr.Textbox(
            label="Question",
            placeholder="e.g., What is Mamba? Summarize the main findings...",
            lines=2,
            max_lines=6,
            scale=7,
        )
        ask_btn = gr.Button("Ask", variant="primary", scale=1, size="lg")

    with gr.Row():
        with gr.Column(scale=5):
            gr.Markdown("### Answer")
            answer_box = gr.Textbox(
                label="",
                placeholder="Your answer will appear here...",
                lines=10,
                max_lines=15,
                interactive=False,
            )
        with gr.Column(scale=4):
            gr.Markdown("### Sources")
            sources_box = gr.HTML(
                "<p style='color:#888; font-style:italic;'>Retrieved sources will appear here after each query.</p>"
            )

    ask_btn.click(
        fn=ask,
        inputs=question_box,
        outputs=[answer_box, sources_box],
    )

    question_box.submit(
        fn=ask,
        inputs=question_box,
        outputs=[answer_box, sources_box],
    )

    with gr.Accordion("Technical details", open=False):
        gr.Markdown(
            f"API: `{QUERY_URL}`  \n"
            "Override with `DOCUMIND_API_URL` environment variable."
        )
        refresh_btn = gr.Button("Check status", size="sm")

        def refresh_status():
            emoji, text = check_backend_status()
            return f"**Status:** {emoji} {text}"

        refresh_btn.click(fn=refresh_status, outputs=[status_bar])

if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
        css="""
        .status-bar { padding: 8px 12px; border-radius: 8px; background: var(--block-background-fill); border: 1px solid var(--border-color-primary); }
        .title-wrap { margin-bottom: 1rem; }
        """,
    )
