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

    app_database_url: str
    reader_database_url: str
    redis_url: str = "redis://localhost:6379/0"

    api_keys: str = "dev-key"
    admin_api_key: str = "dev-admin-key"
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
    schema_retrieval_min_tables: int = 8

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
