"""LLM factory: provider switch, key handling — no API calls (models are only constructed)."""

import pytest
from pydantic import SecretStr

from src.core import llm as llm_module
from src.core.config import Secrets, Settings


@pytest.fixture(autouse=True)
def clear_cache():
    llm_module.get_llm.cache_clear()
    yield
    llm_module.get_llm.cache_clear()


def _patch(monkeypatch, provider: str, **keys):
    monkeypatch.setattr(llm_module, "get_settings", lambda: Settings(llm={"provider": provider}))
    secrets = Secrets(_env_file=None, **{k: SecretStr(v) for k, v in keys.items()})
    monkeypatch.setattr(llm_module, "get_secrets", lambda: secrets)


def test_openai_roles_use_configured_models(monkeypatch):
    _patch(monkeypatch, "openai", openai_api_key="sk-test")
    router, answer = llm_module.get_llm("router"), llm_module.get_llm("answer")
    assert router.model_name == "gpt-4o-mini" and router.temperature == 0
    assert answer.model_name == "gpt-4o"
    assert llm_module.llm_available()


def test_gemini_provider(monkeypatch):
    _patch(monkeypatch, "gemini", google_api_key="g-test")
    model = llm_module.get_llm("answer")
    assert "gemini" in model.model


def test_missing_key_raises_clear_error(monkeypatch):
    _patch(monkeypatch, "openai", openai_api_key="")
    assert not llm_module.llm_available()
    with pytest.raises(llm_module.LLMUnavailableError, match="OPENAI_API_KEY"):
        llm_module.get_llm("answer")
    _patch(monkeypatch, "gemini")
    with pytest.raises(llm_module.LLMUnavailableError, match="GOOGLE_API_KEY"):
        llm_module.get_llm("answer")
