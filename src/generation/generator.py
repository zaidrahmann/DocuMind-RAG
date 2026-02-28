"""HuggingFace-based generator for RAG answer synthesis."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from dotenv import load_dotenv

# Load .env from cwd or parent dirs (standard protocol)
load_dotenv()

logger = logging.getLogger(__name__)

ENV_HF_API_KEY = "HF_API_KEY"
ENV_HF_MODEL = "HF_MODEL"  # optional override; default uses Mistral-7B (flan-t5 deprecated on HF API)

_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly based on the provided context. "
    "If the answer is not present in the context, say 'Not found in documents.' "
    "Be concise and factual."
)

_USER_TEMPLATE = """Context:
{context}

Question: {question}

Answer using ONLY the context above."""


def _connection_error_message(exc: Exception) -> str:
    """Build a user-friendly message for connection/auth failures."""
    err_str = str(exc).strip()
    err_first_line = err_str.split("\n")[0][:120] if err_str else ""
    hint = f"\n\n_Technical note: {type(exc).__name__}: {err_first_line}_"

    if "401" in err_str or "Unauthorized" in err_str or "authentication" in err_str.lower():
        return (
            "**HuggingFace rejected your API key.** Please check:\n"
            "• HF_API_KEY in .env is correct — create or copy at https://huggingface.co/settings/tokens\n"
            "• Token type is **Read** or has **Inference** access (not only 'Write')\n"
            "• If you just created the token, wait a minute and try again"
            + hint
        )
    if "403" in err_str or "forbidden" in err_str.lower():
        return (
            "**Access forbidden.** The model or API may require you to accept terms or use a different token scope. "
            "Check https://huggingface.co/settings/tokens and try a token with Inference access."
            + hint
        )
    if "timeout" in err_str.lower() or "timed out" in err_str.lower():
        return "The AI service took too long to respond. Please try again." + hint
    if any(x in err_str.lower() for x in ("connection", "resolve", "network", "refused", "nodename", "getaddrinfo")):
        return (
            "**Could not reach the AI service.** Please check:\n"
            "• Your internet connection\n"
            "• Firewall/proxy allows https://api-inference.huggingface.co\n"
            "• If using a VPN or corporate network, try disabling or using a different network"
            + hint
        )
    return (
        "**We couldn't connect to the AI service.** Please check:\n"
        "• Your internet connection\n"
        "• Your HuggingFace API key (HF_API_KEY in .env) at https://huggingface.co/settings/tokens\n"
        "• That https://api-inference.huggingface.co is reachable in your browser"
        + hint
    )


class HFGenerator:
    """HuggingFace inference generator using InferenceClient (flan-t5 deprecated on classic API)."""

    DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
    DEFAULT_TIMEOUT = 90
    MAX_RETRIES = 3
    RETRY_DELAY = 5

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
    ) -> None:
        """Initialize the generator.

        Args:
            api_key: HuggingFace API key. Reads HF_API_KEY from environment if not set.
            model: Model ID. Uses HF_MODEL or Mistral-7B by default (flan-t5 deprecated).
            timeout: Request timeout in seconds. Uses 60 if not set.
        """
        self.api_key = api_key or os.environ.get(ENV_HF_API_KEY, "").strip()
        if not self.api_key:
            raise ValueError(
                f"HuggingFace API key required. Set {ENV_HF_API_KEY} in .env "
                "(see .env.example) or pass api_key to HFGenerator()."
            )
        self.model = (
            model
            or os.environ.get(ENV_HF_MODEL, "").strip()
            or self.DEFAULT_MODEL
        )
        self.timeout = timeout if timeout is not None else self.DEFAULT_TIMEOUT
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub")
        self._client = InferenceClient(
            token=self.api_key,
            timeout=self.timeout,
        )

    def generate(self, question: str, context: str) -> str:
        """Generate an answer from the question and retrieved context."""
        if not question or not isinstance(question, str):
            raise ValueError("question must be a non-empty string")
        if context is None:
            context = ""
        if not isinstance(context, str):
            raise ValueError("context must be a string")

        for attempt in range(self.MAX_RETRIES):
            try:
                logger.debug("HF chat_completion request (attempt %d)", attempt + 1)
                messages = [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": _USER_TEMPLATE.format(
                        context=context.strip(), question=question.strip()
                    )},
                ]
                out = self._client.chat_completion(
                    messages=messages,
                    model=self.model,
                    max_tokens=512,
                )
                text = out.choices[0].message.content or ""
                return text.strip() or "No answer generated."

            except Exception as e:
                err_str = str(e).lower()
                logger.warning("HF inference attempt %d failed: %s", attempt + 1, e)

                if "503" in str(e) or "loading" in err_str:
                    if attempt < self.MAX_RETRIES - 1:
                        time.sleep(self.RETRY_DELAY)
                        continue
                    return "Model is loading. Please try again in a minute."

                if attempt == self.MAX_RETRIES - 1 or any(
                    x in str(e) or x in err_str for x in ("401", "403", "connection", "timeout")
                ):
                    return _connection_error_message(e)

                time.sleep(self.RETRY_DELAY)

        return "Too many retries. Please try again later."
