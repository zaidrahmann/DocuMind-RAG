"""Test HuggingFace API key and connectivity. Run: python scripts/check_hf_connection.py"""

from __future__ import annotations

import os
import sys

if os.path.basename(os.getcwd()) == "scripts":
    os.chdir(os.path.dirname(os.getcwd()))

from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.environ.get("HF_API_KEY", "").strip()
if not HF_API_KEY:
    print("ERROR: HF_API_KEY is not set in .env")
    print("  Use Ollama instead: set DOCUMIND_GENERATOR=ollama and run 'ollama run llama3'")
    sys.exit(1)

print("HF_API_KEY is set (length %d)" % len(HF_API_KEY))
print("Note: HuggingFace Inference API has limited free model availability (404/410 common).")
print("  Use DOCUMIND_GENERATOR=ollama for local Ollama instead.")
print()

try:
    from huggingface_hub import InferenceClient
except ImportError:
    print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)

client = InferenceClient(token=HF_API_KEY, timeout=30, provider="hf-inference")
model = "mistralai/Mistral-7B-Instruct-v0.2"

print("Calling HuggingFace Inference API (model: %s)..." % model)
try:
    out = client.text_generation(
        "Say hello in one word.",
        model=model,
        max_new_tokens=5,
        return_full_text=False,
    )
    text = out if isinstance(out, str) else getattr(out, "generated_text", str(out))
    print("SUCCESS. Model replied:", repr(text))
except Exception as e:
    print("FAILED:", type(e).__name__, e)
    print()
    print("HuggingFace free Inference API often returns 404 for many models.")
    print("RECOMMENDED: Use Ollama (local, free, reliable):")
    print("  1. Install Ollama: https://ollama.com")
    print("  2. Run: ollama run llama3")
    print("  3. Set DOCUMIND_GENERATOR=ollama in .env")
    import traceback
    traceback.print_exc()
    sys.exit(1)
