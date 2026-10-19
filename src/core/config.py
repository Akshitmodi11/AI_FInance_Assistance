"""Application configuration.

Two sources, kept deliberately separate:

* ``config.yaml`` — non-secret settings (models, chunk sizes, cache TTLs). Committed to git.
  Any value can be overridden with an env var using the ``FINNIE_`` prefix and ``__`` for
  nesting, e.g. ``FINNIE_LLM__PROVIDER=gemini``.
* ``.env`` — API keys only. Never committed. Loaded into :class:`Secrets` as ``SecretStr`` so
  keys are masked if a settings object is ever printed or logged.

Use :func:`get_settings` / :func:`get_secrets` everywhere instead of reading ``os.environ``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"


class AppConfig(BaseModel):
    name: str = "Finnie"
    log_level: str = "INFO"


class ModelPair(BaseModel):
    router_model: str
    answer_model: str


class LLMConfig(BaseModel):
    provider: Literal["openai", "gemini"] = "openai"
    openai: ModelPair = ModelPair(router_model="gpt-4o-mini", answer_model="gpt-4o")
    gemini: ModelPair = ModelPair(
        router_model="gemini-2.0-flash", answer_model="gemini-2.0-flash"
    )
    temperature: float = 0.2
    timeout_s: int = 30


class RAGConfig(BaseModel):
    articles_dir: Path = Path("data/kb/articles")
    documents_dir: Path = Path("data/kb/documents")
    manifest_path: Path = Path("data/kb/manifest.yaml")
    index_dir: Path = Path("data/index")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 800
    chunk_overlap: int = 120
    retrieval_mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    top_k: int = 5
    candidate_k: int = 20
    rrf_k: int = 60
    min_score: float = 0.35
    categories: list[str] = []


class AlphaVantageConfig(BaseModel):
    requests_per_minute: int = 5
    requests_per_day: int = 25


class MarketDataConfig(BaseModel):
    providers: list[str] = ["alpha_vantage", "yfinance", "mock"]
    cache_dir: Path = Path(".cache/market")
    cache_ttl_s: int = 1800
    request_timeout_s: float = 10
    max_retries: int = 2
    alpha_vantage: AlphaVantageConfig = AlphaVantageConfig()


class NewsConfig(BaseModel):
    providers: list[str] = ["tavily", "mock"]
    max_results: int = 5


class ReturnBand(BaseModel):
    low: float
    expected: float
    high: float


class GoalsConfig(BaseModel):
    risk_profiles: dict[str, ReturnBand] = {}
    inflation: float = 0.025


class MemoryConfig(BaseModel):
    checkpoint_db: Path = Path(".cache/checkpoints.sqlite")


class Settings(BaseSettings):
    """Non-secret settings: defaults < config.yaml < FINNIE_* env vars."""

    model_config = SettingsConfigDict(
        yaml_file=CONFIG_PATH,
        env_prefix="FINNIE_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app: AppConfig = AppConfig()
    llm: LLMConfig = LLMConfig()
    rag: RAGConfig = RAGConfig()
    market_data: MarketDataConfig = MarketDataConfig()
    news: NewsConfig = NewsConfig()
    goals: GoalsConfig = GoalsConfig()
    memory: MemoryConfig = MemoryConfig()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Earlier sources win: explicit kwargs > env vars > config.yaml > class defaults.
        return (init_settings, env_settings, YamlConfigSettingsSource(settings_cls))

    def resolve(self, path: Path) -> Path:
        """Turn a config-relative path into an absolute one under the project root."""
        return path if path.is_absolute() else PROJECT_ROOT / path


class Secrets(BaseSettings):
    """API keys from .env / the environment. Missing keys are ``None``, never an error —
    callers decide whether to fall back (e.g. to mock market data)."""

    model_config = SettingsConfigDict(env_file=ENV_PATH, extra="ignore")

    openai_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    alpha_vantage_api_key: SecretStr | None = None
    tavily_api_key: SecretStr | None = None
    langsmith_api_key: SecretStr | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_secrets() -> Secrets:
    return Secrets()
