"""Generation: answer synthesis from retrieved context."""

from .generator import HFGenerator
from .ollama_generator import OllamaGenerator

__all__ = ["HFGenerator", "OllamaGenerator"]
