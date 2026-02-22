"""HuggingFace-based generator for RAG answer synthesis."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import requests

from dotenv import load_dotenv

# Load .env from cwd or parent dirs (standard protocol)
load_dotenv()

logger = logging.getLogger(__name__)

ENV_HF_API_KEY = "HF_API_KEY"

_PROMPT_TEMPLATE = """Answer the question using ONLY the context below.
If the answer is not in the context, say 'Not found in documents.'

Context:
{context}

Question:
{question}"""


class HFGenerator:
    """Free HuggingFace inference generator using api-inference.huggingface.co."""

    DEFAULT_ENDPOINT = "https://api-inference.huggingface.co/models/google/flan-t5-base"
    DEFAULT_TIMEOUT = 30
    MAX_RETRIES = 3
    RETRY_DELAY = 5

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
        timeout: int | None = None,
    ) -> None:
        """Initialize the generator.

        Args:
            api_key: HuggingFace API key. Reads HF_API_KEY from environment if not set.
            endpoint: Model endpoint URL. Uses google/flan-t5-base if not set.
            timeout: Request timeout in seconds. Uses 30 if not set.
        """
        self.api_key = api_key or os.environ.get(ENV_HF_API_KEY, "").strip()
        if not self.api_key:
            raise ValueError(
                f"HuggingFace API key required. Set {ENV_HF_API_KEY} in .env "
                "(see .env.example) or pass api_key to HFGenerator()."
            )
        self.endpoint = endpoint or self.DEFAULT_ENDPOINT
        self.timeout = timeout if timeout is not None else self.DEFAULT_TIMEOUT
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

    def generate(self, question: str, context: str) -> str:
        """Generate an answer from the question and retrieved context.

        Args:
            question: User question.
            context: Retrieved context text (chunks from RAG).

        Returns:
            Clean string answer. Returns fallback message on errors.
        """
        if not question or not isinstance(question, str):
            raise ValueError("question must be a non-empty string")
        if context is None:
            context = ""
        if not isinstance(context, str):
            raise ValueError("context must be a string")

        prompt = _PROMPT_TEMPLATE.format(
            context=context.strip(), question=question.strip()
        )
        payload: dict[str, Any] = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 128, "return_full_text": False},
        }

        resp: requests.Response | None = None
        for attempt in range(self.MAX_RETRIES):
            try:
                logger.debug("HF inference request (attempt %d)", attempt + 1)
                resp = self._session.post(
                    self.endpoint,
                    json=payload,
                    timeout=self.timeout,
                )

                if resp.status_code == 429:
                    logger.warning("Rate limited (429). Retrying in %ds...", self.RETRY_DELAY)
                    time.sleep(self.RETRY_DELAY)
                    continue

                if resp.status_code == 503:
                    logger.warning(
                        "Model loading (503). Retrying in %ds...", self.RETRY_DELAY
                    )
                    time.sleep(self.RETRY_DELAY)
                    continue

                resp.raise_for_status()

                data = resp.json()
                err = self._check_error_response(data)
                if err:
                    return err
                return self._extract_generated_text(data)

            except requests.Timeout:
                logger.warning("Request timeout (attempt %d/%d)", attempt + 1, self.MAX_RETRIES)
                if attempt == self.MAX_RETRIES - 1:
                    return "Request timed out. Please try again."
                time.sleep(self.RETRY_DELAY)
            except requests.RequestException as e:
                logger.error("Request error: %s", e)
                return "Unable to reach the inference service."
            except (ValueError, KeyError, TypeError) as e:
                resp_preview = (resp.text[:200] if resp is not None else "")
                logger.error("JSON parsing error: %s (response: %s)", e, resp_preview)
                return "Invalid response from inference service."

        return "Too many retries. Please try again later."

    def _check_error_response(self, data: Any) -> str | None:
        """Return error message if response is an error, else None."""
        if isinstance(data, dict) and "error" in data:
            msg = str(data.get("error", "Unknown error"))
            logger.warning("HF API error response: %s", msg)
            if "loading" in msg.lower():
                return "Model is loading. Please try again shortly."
            return f"Inference error: {msg}"
        return None

    def _extract_generated_text(self, data: Any) -> str:
        """Extract generated text from HF API response. Handles multiple formats."""
        if isinstance(data, list) and len(data) > 0:
            item = data[0]
            if isinstance(item, dict):
                text = item.get("generated_text", "")
                if isinstance(text, str):
                    return text.strip()
        if isinstance(data, dict):
            text = data.get("generated_text", "")
            if isinstance(text, str):
                return text.strip()
        return "Invalid response format."
