"""Chat model + the four structured-output runnables the graph uses.

Structured output (`with_structured_output`) makes the model return a validated Pydantic
object. Groq models occasionally return invalid JSON for it (Architecture gotcha #6), so
`ainvoke_structured` retries once before giving up.
"""

import logging
import os
from typing import Any, Protocol

from ask_data.config import settings

log = logging.getLogger(__name__)


class StructuredLLM(Protocol):
    async def ainvoke(self, input: Any, /) -> Any: ...


PROVIDER_KEYS = {"groq": "GROQ_API_KEY", "google_genai": "GOOGLE_API_KEY"}


class LLMUnavailable(RuntimeError):
    pass


class UnavailableLLM:
    """Stands in when the model can't be built (e.g. no key), so the API still starts and
    every LLM node reports the reason instead of the app crashing at boot."""

    def __init__(self, reason: str):
        self.reason = reason

    async def ainvoke(self, input: Any, /) -> Any:
        raise LLMUnavailable(self.reason)


def export_provider_env() -> None:
    """pydantic-settings reads .env into `settings` but not into os.environ, where LangChain
    provider clients and LangSmith look. Copy over what is set; never overwrite real env vars."""
    values = {
        "GROQ_API_KEY": settings.groq_api_key,
        "GOOGLE_API_KEY": settings.google_api_key,
        "LANGSMITH_API_KEY": settings.langsmith_api_key,
        "LANGSMITH_PROJECT": settings.langsmith_project,
    }
    for name, value in values.items():
        value = value.strip()  # pasted keys often carry a trailing newline / non-breaking space
        if value and not os.environ.get(name):
            os.environ[name] = value
    # Tracing without a key only logs upload errors
    tracing = settings.langsmith_tracing and bool(os.environ.get("LANGSMITH_API_KEY"))
    os.environ.setdefault("LANGSMITH_TRACING", "true" if tracing else "false")


def build_structured_llms_or_unavailable() -> tuple[dict[str, StructuredLLM], str]:
    """(llms, status). Status is "ok" or why the model is unavailable."""
    provider = settings.llm_model.split(":", 1)[0]
    key = PROVIDER_KEYS.get(provider)
    value = os.environ.get(key, "") if key else ""
    if key and not value:
        reason = f"LLM not configured: {key} is empty (LLM_MODEL={settings.llm_model})"
    elif key and not value.isascii():
        # HTTP headers are ASCII: such a key fails every request with "'ascii' codec can't encode"
        positions = [i for i, c in enumerate(value) if not c.isascii()]
        reason = (
            f"LLM not configured: {key} contains non-ASCII characters at positions {positions} "
            "(usually invisible characters from copy-paste); retype the key in .env"
        )
    else:
        try:
            return build_structured_llms(), "ok"
        except Exception as e:  # noqa: BLE001
            reason = f"LLM could not be created: {type(e).__name__}: {e}"
    log.error(reason)
    stub = UnavailableLLM(reason)
    return {"intent": stub, "sql": stub, "answer": stub, "rewrite": stub}, reason


def build_structured_llms() -> dict[str, StructuredLLM]:
    """{"intent", "sql", "answer", "rewrite"} runnables for settings.llm_model."""
    from langchain.chat_models import init_chat_model

    from ask_data.graph.models import FinalAnswer, Intent, SqlDraft, StandaloneQuestion

    llm = init_chat_model(settings.llm_model, temperature=0)
    return {
        "intent": llm.with_structured_output(Intent),
        "sql": llm.with_structured_output(SqlDraft),
        "answer": llm.with_structured_output(FinalAnswer),
        "rewrite": llm.with_structured_output(StandaloneQuestion),
    }


async def ainvoke_structured(runnable: StructuredLLM, prompt: str, retries: int = 1) -> Any:
    """Call a structured-output runnable; retry on failure, then re-raise the last error."""
    for attempt in range(retries + 1):
        try:
            result = await runnable.ainvoke(prompt)
            if result is None:  # some providers return None instead of raising
                raise ValueError("model returned no structured output")
            return result
        except Exception as e:  # provider, parse and validation errors alike
            if attempt == retries:
                raise
            log.warning("structured output failed (attempt %d), retrying: %s", attempt + 1, e)
