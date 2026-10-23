"""LLM factory: one place that knows which provider/model to use.

``get_llm("router")`` → a small, fast model for classification / extraction.
``get_llm("answer")`` → the stronger model that writes user-facing explanations.
Provider (OpenAI default, Gemini alternative) comes from ``config.yaml → llm.provider``;
keys come from ``.env``. Agents never import a provider SDK directly, so switching provider is a
one-line config change and tests can inject a fake model.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from langchain_core.language_models import BaseChatModel

from src.core.config import get_secrets, get_settings

Role = Literal["router", "answer"]


class LLMUnavailableError(RuntimeError):
    """No API key for the configured provider."""


def _secret(value) -> str | None:
    text = value.get_secret_value() if value is not None else ""
    return text or None


def llm_available() -> bool:
    secrets, provider = get_secrets(), get_settings().llm.provider
    key = secrets.openai_api_key if provider == "openai" else secrets.google_api_key
    return _secret(key) is not None


@lru_cache(maxsize=4)
def get_llm(role: Role = "answer") -> BaseChatModel:
    settings, secrets = get_settings(), get_secrets()
    cfg = settings.llm
    if cfg.provider == "openai":
        key = _secret(secrets.openai_api_key)
        if key is None:
            raise LLMUnavailableError("OPENAI_API_KEY is not set in .env")
        from langchain_openai import ChatOpenAI

        model = cfg.openai.router_model if role == "router" else cfg.openai.answer_model
        return ChatOpenAI(
            model=model,
            api_key=key,
            temperature=0 if role == "router" else cfg.temperature,
            timeout=cfg.timeout_s,
            max_retries=2,
        )

    key = _secret(secrets.google_api_key)
    if key is None:
        raise LLMUnavailableError("GOOGLE_API_KEY is not set in .env")
    from langchain_google_genai import ChatGoogleGenerativeAI

    model = cfg.gemini.router_model if role == "router" else cfg.gemini.answer_model
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=key,
        temperature=0 if role == "router" else cfg.temperature,
        timeout=cfg.timeout_s,
        max_retries=2,
    )
