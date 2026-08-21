from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from app.clients.ollama_client import OllamaClient


def test_model_returns_configured_model() -> None:
    client = OllamaClient(
        base_url="http://ollama:11434",
        model="llama3.1:8b",
    )

    assert client.model == "llama3.1:8b"


@pytest.mark.asyncio
async def test_generate_returns_llm_response() -> None:
    client = OllamaClient(
        base_url="http://ollama:11434",
        model="llama3.1:8b",
    )

    response = Mock()
    response.json.return_value = {
        "model": "llama3.1:8b",
        "response": "Generated answer",
        "prompt_eval_count": 20,
        "eval_count": 8,
    }
    response.raise_for_status.return_value = None

    client._client.post = AsyncMock(
        return_value=response
    )

    result = await client.generate("Example prompt")

    client._client.post.assert_awaited_once_with(
        "/api/generate",
        json={
            "model": "llama3.1:8b",
            "prompt": "Example prompt",
            "stream": False,
            "options": {
                "temperature": 0.0,
            },
        },
    )
    response.raise_for_status.assert_called_once_with()

    assert result.text == "Generated answer"
    assert result.model == "llama3.1:8b"
    assert result.input_tokens == 20
    assert result.output_tokens == 8
    assert result.total_tokens == 28

    await client.close()


@pytest.mark.asyncio
async def test_generate_allows_missing_token_usage() -> None:
    client = OllamaClient(
        base_url="http://ollama:11434",
        model="llama3.1:8b",
    )

    response = Mock()
    response.json.return_value = {
        "response": "Generated answer",
    }
    response.raise_for_status.return_value = None

    client._client.post = AsyncMock(
        return_value=response
    )

    result = await client.generate("Example prompt")

    assert result.model == "llama3.1:8b"
    assert result.input_tokens is None
    assert result.output_tokens is None

    await client.close()


@pytest.mark.asyncio
async def test_generate_rejects_empty_prompt() -> None:
    client = OllamaClient(
        base_url="http://ollama:11434",
        model="llama3.1:8b",
    )
    client._client.post = AsyncMock()

    with pytest.raises(
        ValueError,
        match="prompt must not be empty",
    ):
        await client.generate(" ")

    client._client.post.assert_not_awaited()

    await client.close()


@pytest.mark.asyncio
async def test_generate_propagates_http_error() -> None:
    client = OllamaClient(
        base_url="http://ollama:11434",
        model="llama3.1:8b",
    )

    request = httpx.Request(
        "POST",
        "http://ollama:11434/api/generate",
    )
    response = httpx.Response(
        status_code=500,
        request=request,
    )

    client._client.post = AsyncMock(
        return_value=response
    )

    with pytest.raises(httpx.HTTPStatusError):
        await client.generate("Example prompt")

    await client.close()


@pytest.mark.asyncio
async def test_generate_rejects_empty_response_text() -> None:
    client = OllamaClient(
        base_url="http://ollama:11434",
        model="llama3.1:8b",
    )

    response = Mock()
    response.json.return_value = {"response": " "}
    response.raise_for_status.return_value = None
    client._client.post = AsyncMock(
        return_value=response
    )

    with pytest.raises(
        ValueError,
        match="text must not be empty",
    ):
        await client.generate("Example prompt")

    await client.close()


@pytest.mark.asyncio
async def test_close_closes_http_client() -> None:
    client = OllamaClient(
        base_url="http://ollama:11434",
        model="llama3.1:8b",
    )
    client._client.aclose = AsyncMock()

    await client.close()

    client._client.aclose.assert_awaited_once_with()


@pytest.mark.parametrize(
    ("base_url", "model", "message"),
    [
        (" ", "llama3.1:8b", "base_url must not be empty"),
        ("http://ollama:11434", " ", "model must not be empty"),
    ],
)
def test_init_rejects_empty_configuration(
    base_url: str,
    model: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        OllamaClient(
            base_url=base_url,
            model=model,
        )


def test_init_rejects_negative_temperature() -> None:
    with pytest.raises(
        ValueError,
        match="temperature must not be negative",
    ):
        OllamaClient(
            base_url="http://ollama:11434",
            model="llama3.1:8b",
            temperature=-0.1,
        )
