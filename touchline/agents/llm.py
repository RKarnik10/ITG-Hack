"""LLM provider chain, client factory and response cache.

Providers are tried in order; the default chain is
  GitHub Models (free, Microsoft-hosted)  ->  Azure AI Foundry (if configured)
  ->  local Ollama  ->  offline templates.
All online providers speak the OpenAI Chat Completions protocol, so one
Agent Framework client class covers them; only base_url / key / model change.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "cache" / "llm"


@dataclass(frozen=True)
class Provider:
    name: str          # github | foundry | ollama | offline
    model: str
    base_url: str | None = None
    api_key: str | None = None
    timeout: float = 60.0

    @property
    def label(self) -> str:
        return {"github": "GitHub Models", "foundry": "Azure AI Foundry", "ollama": "Ollama (local)",
                "offline": "Offline templates"}[self.name] + ("" if self.name == "offline" else f" · {self.model}")


def _ollama_up(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(base_url.replace("/v1", "") + "/api/tags", timeout=1.5) as r:
            return r.status == 200
    except Exception:
        return False


def available_providers() -> list[Provider]:
    """Provider chain from the environment, in priority order."""
    order = os.getenv("LLM_PROVIDERS", "github,foundry,ollama,offline").split(",")
    chain = []
    for name in (o.strip() for o in order):
        if name == "github" and os.getenv("GITHUB_TOKEN"):
            chain.append(Provider("github", os.getenv("GITHUB_MODEL", "openai/gpt-4.1-mini"),
                                  os.getenv("GITHUB_MODELS_ENDPOINT", "https://models.github.ai/inference"),
                                  os.getenv("GITHUB_TOKEN"), timeout=60))
        elif name == "foundry" and os.getenv("AZURE_AI_ENDPOINT") and os.getenv("AZURE_AI_KEY"):
            # Foundry / Azure OpenAI v1 endpoint, e.g. https://<resource>.openai.azure.com/openai/v1/
            chain.append(Provider("foundry", os.getenv("AZURE_AI_MODEL", "gpt-4.1-mini"),
                                  os.getenv("AZURE_AI_ENDPOINT"), os.getenv("AZURE_AI_KEY"), timeout=60))
        elif name == "ollama":
            base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            if _ollama_up(base):
                chain.append(Provider("ollama", os.getenv("OLLAMA_MODEL", "llama3.2"), base, "ollama", timeout=150))
        elif name == "offline":
            chain.append(Provider("offline", "templates"))
    if not chain or chain[-1].name != "offline":
        chain.append(Provider("offline", "templates"))
    return chain


def make_client(p: Provider):
    from agent_framework.openai import OpenAIChatCompletionClient
    return OpenAIChatCompletionClient(model=p.model, base_url=p.base_url, api_key=p.api_key)


class ProviderOutage(RuntimeError):
    """Raised for providers the user has switched off to demo failure recovery."""


# ---------------------------------------------------------------- cache
def cache_key(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:32]


def cache_get(key: str) -> dict | None:
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return None
    return None


def cache_put(key: str, value: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{key}.json").write_text(json.dumps(value, indent=1, default=str))
