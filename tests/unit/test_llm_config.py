"""Provider-key handling at startup (fake keys only)."""

import pytest

from ask_data import llm
from ask_data.config import settings


@pytest.fixture
def groq_model(monkeypatch):
    monkeypatch.setattr(settings, "llm_model", "groq:some-model")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    return monkeypatch


def test_missing_key_is_unavailable(groq_model):
    llms, status = llm.build_structured_llms_or_unavailable()
    assert "GROQ_API_KEY is empty" in status
    assert isinstance(llms["sql"], llm.UnavailableLLM)


def test_non_ascii_key_is_unavailable_with_positions(groq_model):
    # a pasted key with a combining mark: HTTP headers would fail with "'ascii' codec ..."
    groq_model.setenv("GROQ_API_KEY", "gsk_fakḛ")
    llms, status = llm.build_structured_llms_or_unavailable()
    assert "non-ASCII characters at positions [8]" in status
    assert isinstance(llms["intent"], llm.UnavailableLLM)


def test_export_strips_whitespace(groq_model):
    groq_model.setattr(settings, "groq_api_key", "gsk_fake \n")
    llm.export_provider_env()
    import os

    assert os.environ["GROQ_API_KEY"] == "gsk_fake"
