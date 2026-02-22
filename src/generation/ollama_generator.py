"""Ollama-based generator for RAG answer synthesis. Runs locally — no API key needed."""

from __future__ import annotations

import logging
import os
from typing import Any

import requests

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

ENV_OLLAMA_URL = "OLLAMA_URL"
ENV_OLLAMA_MODEL = "OLLAMA_MODEL"

_PROMPT_TEMPLATE = """Answer the question using ONLY the context below.
If the answer is not in the context, say 'Not found in documents.'

Context:
{context}

Question:
{question}"""


class OllamaGenerator:
    """Local LLM generator using Ollama (ollama run llama3 / mistral / etc)."""

    DEFAULT_URL = "http://localhost:11434"
    DEFAULT_MODEL = "llama3"
    DEFAULT_TIMEOUT = 120

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get(ENV_OLLAMA_URL, "").strip() or self.DEFAULT_URL).rstrip("/")
        self.model = model or os.environ.get(ENV_OLLAMA_MODEL, "").strip() or self.DEFAULT_MODEL
        self.timeout = timeout if timeout is not None else self.DEFAULT_TIMEOUT

    def generate(self, question: str, context: str) -> str:
        if not question or not isinstance(question, str):
            raise ValueError("question must be a non-empty string")
        if context is None:
            context = ""
        if not isinstance(context, str):
            raise ValueError("context must be a string")

        prompt = _PROMPT_TEMPLATE.format(
            context=context.strip(), question=question.strip()
        )

        url = f"{self.base_url}/api/generate"
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 256},
        }

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.ConnectionError:
            return (
                "**Ollama is not running.** Start Ollama and pull a model:\n"
                "```\nollama run llama3\n```\n"
                "Then ensure the server is running (e.g. `ollama serve` or start Ollama app)."
            )
        except requests.Timeout:
            return "Ollama took too long to respond. Try a smaller model or increase timeout."
        except requests.RequestException as e:
            logger.warning("Ollama request failed: %s", e)
            return f"Ollama error: {e}"

        text = data.get("response", "")
        return (text.strip() if isinstance(text, str) else "").strip() or "No answer generated."
