from src.core.config import PROJECT_ROOT, Secrets, Settings


def test_yaml_values_load():
    s = Settings()
    assert s.llm.provider in {"openai", "gemini"}
    assert s.market_data.cache_ttl_s == 1800
    assert "retirement_tax" in s.rag.categories
    assert s.goals.risk_profiles["moderate"].expected == 0.06


def test_env_var_overrides_yaml(monkeypatch):
    monkeypatch.setenv("FINNIE_LLM__PROVIDER", "gemini")
    monkeypatch.setenv("FINNIE_RAG__TOP_K", "9")
    s = Settings()
    assert s.llm.provider == "gemini"
    assert s.rag.top_k == 9


def test_resolve_makes_paths_absolute():
    s = Settings()
    assert s.resolve(s.rag.index_dir) == PROJECT_ROOT / "data" / "index"


def test_secrets_are_masked(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    secrets = Secrets(_env_file=None)
    assert secrets.openai_api_key.get_secret_value() == "sk-test-123"
    assert "sk-test-123" not in repr(secrets)


def test_missing_secrets_are_none(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert Secrets(_env_file=None).tavily_api_key is None
