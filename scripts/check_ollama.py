"""Test Ollama connectivity. Run: python scripts/check_ollama.py"""

from __future__ import annotations

import os
import sys

if os.path.basename(os.getcwd()) == "scripts":
    os.chdir(os.path.dirname(os.getcwd()))

from dotenv import load_dotenv

load_dotenv()

import requests

url = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
model = os.environ.get("OLLAMA_MODEL", "llama3")

print("Testing Ollama at %s (model: %s)..." % (url, model))
try:
    r = requests.post(
        url + "/api/generate",
        json={"model": model, "prompt": "Say hello in one word.", "stream": False},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    text = data.get("response", "")
    print("SUCCESS. Model replied:", repr(text[:80]))
except requests.ConnectionError:
    print("FAILED: Cannot connect to Ollama.")
    print("  Install Ollama: https://ollama.com")
    print("  Run: ollama run %s" % model)
    sys.exit(1)
except requests.RequestException as e:
    print("FAILED:", e)
    sys.exit(1)
