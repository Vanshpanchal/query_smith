from pydantic_settings import BaseSettings, SettingsConfigDict

SECRET_FIELDS = {
    "groq_api_key", "google_api_key", "langsmith_api_key", "api_keys", "admin_api_key",
    "app_database_url", "reader_database_url", "redis_url",
}  # fmt: skip


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_model: str = "groq:qwen/qwen3.8-27b"
    # Provider keys: LangChain clients read them from os.environ (see llm.export_provider_env)
    groq_api_key: str = ""
    google_api_key: str = ""
    embedding_model: str = "models/text-embedding-004"
    llm_max_retries: int = 2  # provider-level retries (429 / 5xx) with backoff; evals raise it

    app_database_url: str
    reader_database_url: str
    redis_url: str = "redis://localhost:6379/0"

    api_keys: str = "dev-key"
    admin_api_key: str = ""  # empty = admin endpoints disabled
    demo_api_key: str = ""  # public key the demo page pre-fills; must also be in API_KEYS
    rate_limit_per_minute: int = 20

    max_attempts: int = 3
    max_rows: int = 1000
    max_plan_cost: float = 1_000_000
    statement_timeout_ms: int = 5000
    answer_rows: int = 20  # rows the answer LLM sees (Architecture gotcha #4)
    stream_rows: int = 50  # rows sent in the SSE `rows` event

    # No dataset is named in code (ADR-009): the default comes from DEFAULT_DATASET in .env
    default_dataset: str | None = None
    datasets_dir: str = "datasets/"
    schema_retrieval_min_tables: int = 8  # datasets with more tables use pgvector retrieval
    schema_retrieval_top_k: int = 4
    history_messages: int = 8  # recent messages `contextualize` sees for follow-ups
    fewshots_k: int = 3  # similar few-shot examples in the SQL prompt (E3 sets 0)

    answer_cache_ttl_s: int = 3600  # Redis answer cache for first questions; 0 = off

    langsmith_tracing: bool = True
    langsmith_api_key: str = ""
    langsmith_project: str = "ask-data-dev"

    def __repr_args__(self):
        # repr()/str() show up in logs, tracebacks and pytest assertion messages: mask secrets
        for name, value in super().__repr_args__():
            yield name, ("***" if value and name in SECRET_FIELDS else value)

    @property
    def api_key_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}


settings = Settings()
