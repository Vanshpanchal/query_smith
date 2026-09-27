from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_model: str = "groq:qwen/qwen3.8-27b"
    embedding_model: str = "models/text-embedding-004"

    app_database_url: str
    reader_database_url: str
    redis_url: str = "redis://localhost:6379/0"

    api_keys: str = "dev-key"
    admin_api_key: str = "dev-admin-key"

    max_attempts: int = 3
    max_rows: int = 1000
    max_plan_cost: float = 1_000_000
    statement_timeout_ms: int = 5000

    default_dataset: str = "ipl"
    datasets_dir: str = "datasets/"
    schema_retrieval_min_tables: int = 8

    langsmith_tracing: bool = True
    langsmith_project: str = "ask-data-dev"


settings = Settings()
