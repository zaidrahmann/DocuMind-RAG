"""Gradio web UI for DocuMind-RAG query interface."""

from __future__ import annotations

import os
from typing import Any

import gradio as gr
import requests

# Default API base URL (override with DOCUMIND_API_URL env var)
API_BASE = os.environ.get("DOCUMIND_API_URL", "http://localhost:8000")
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


def format_sources(sources: list[dict[str, Any]], scores: list[float] | None = None) -> str:
    """Format sources as readable text."""
    if not sources:
        return "_No sources retrieved._"
    lines = []
    for i, src in enumerate(sources):
        score_str = f" (score: {scores[i]:.3f})" if scores and i < len(scores) else ""
        filename = src.get("filename", "Unknown")
        page = src.get("page_number", "?")
        chunk = src.get("chunk_index", "?")
        lines.append(f"• **{filename}** – page {page}, chunk {chunk}{score_str}")
    return "\n".join(lines)


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
        with gr.Column(scale=1):
            gr.Markdown("### Answer")
            answer_box = gr.Textbox(
                label="",
                placeholder="Your answer will appear here...",
                lines=10,
                max_lines=15,
                interactive=False,
            )
        with gr.Column(scale=1):
            gr.Markdown("### Sources")
            sources_box = gr.Markdown(
                "_Retrieved document chunks will appear here after each query._"
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
