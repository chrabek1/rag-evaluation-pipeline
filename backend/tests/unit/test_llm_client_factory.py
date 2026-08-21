from unittest.mock import Mock

import pytest

from app.clients.llm_client_factory import (
    create_llm_client,
)


def test_create_llm_client_returns_gemini_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gemini_client = Mock()
    constructor = Mock(return_value=gemini_client)

    monkeypatch.setattr(
        "app.clients.llm_client_factory.GeminiClient",
        constructor,
    )

    result = create_llm_client(
        provider="gemini",
        model="gemini-test-model",
        temperature=0.0,
        gemini_api_key="test-api-key",
    )

    constructor.assert_called_once_with(
        api_key="test-api-key",
        model="gemini-test-model",
        temperature=0.0,
    )
    assert result is gemini_client


def test_create_llm_client_returns_ollama_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ollama_client = Mock()
    constructor = Mock(return_value=ollama_client)

    monkeypatch.setattr(
        "app.clients.llm_client_factory.OllamaClient",
        constructor,
    )

    result = create_llm_client(
        provider="ollama",
        model="llama3.2:3b",
        temperature=0.0,
        ollama_base_url=(
            "http://host.docker.internal:11434"
        ),
    )

    constructor.assert_called_once_with(
        base_url="http://host.docker.internal:11434",
        model="llama3.2:3b",
        temperature=0.0,
    )
    assert result is ollama_client


def test_create_llm_client_requires_gemini_api_key() -> None:
    with pytest.raises(
        ValueError,
        match="gemini_api_key is required for Gemini",
    ):
        create_llm_client(
            provider="gemini",
            model="gemini-test-model",
            temperature=0.0,
        )


def test_create_llm_client_requires_ollama_base_url() -> None:
    with pytest.raises(
        ValueError,
        match="ollama_base_url is required for Ollama",
    ):
        create_llm_client(
            provider="ollama",
            model="llama3.2:3b",
            temperature=0.0,
        )


def test_create_llm_client_rejects_unknown_provider() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported LLM provider: unknown",
    ):
        create_llm_client(
            provider="unknown",
            model="test-model",
            temperature=0.0,
        )