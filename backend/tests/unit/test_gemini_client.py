from unittest.mock import AsyncMock, Mock

import pytest

from app.clients.gemini_client import GeminiClient


def create_client(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[GeminiClient, Mock]:
    async_client = Mock()
    async_client.models.generate_content = AsyncMock()
    async_client.aclose = AsyncMock()

    google_client = Mock()
    google_client.aio = async_client

    client_factory = Mock(
        return_value=google_client
    )

    monkeypatch.setattr(
        "app.clients.gemini_client.genai.Client",
        client_factory,
    )

    client = GeminiClient(
        api_key="test-api-key",
        model="gemini-test-model",
    )

    client_factory.assert_called_once_with(
        api_key="test-api-key",
    )

    return client, async_client


def test_model_returns_configured_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = create_client(monkeypatch)

    assert client.model == "gemini-test-model"


@pytest.mark.asyncio
async def test_generate_returns_llm_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, async_client = create_client(monkeypatch)

    usage = Mock()
    usage.prompt_token_count = 20
    usage.candidates_token_count = 8

    response = Mock()
    response.text = "Generated answer"
    response.usage_metadata = usage

    async_client.models.generate_content.return_value = (
        response
    )

    result = await client.generate("Example prompt")

    call = (
        async_client.models.generate_content
        .await_args
    )

    assert call.kwargs["model"] == "gemini-test-model"
    assert call.kwargs["contents"] == "Example prompt"
    assert call.kwargs["config"].temperature == 0.0

    assert result.text == "Generated answer"
    assert result.model == "gemini-test-model"
    assert result.input_tokens == 20
    assert result.output_tokens == 8
    assert result.total_tokens == 28

    await client.close()
    async_client.aclose.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_generate_allows_missing_token_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, async_client = create_client(monkeypatch)

    response = Mock()
    response.text = "Generated answer"
    response.usage_metadata = None

    async_client.models.generate_content.return_value = (
        response
    )

    result = await client.generate("Example prompt")

    assert result.input_tokens is None
    assert result.output_tokens is None

    await client.close()


@pytest.mark.asyncio
async def test_generate_rejects_empty_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, async_client = create_client(monkeypatch)

    with pytest.raises(
        ValueError,
        match="prompt must not be empty",
    ):
        await client.generate(" ")

    (
        async_client.models.generate_content
        .assert_not_awaited()
    )

    await client.close()


@pytest.mark.asyncio
async def test_generate_rejects_response_without_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, async_client = create_client(monkeypatch)

    response = Mock()
    response.text = None
    response.usage_metadata = None

    async_client.models.generate_content.return_value = (
        response
    )

    with pytest.raises(
        ValueError,
        match="Gemini response does not contain text",
    ):
        await client.generate("Example prompt")

    await client.close()


@pytest.mark.asyncio
async def test_generate_propagates_sdk_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, async_client = create_client(monkeypatch)
    async_client.models.generate_content.side_effect = (
        RuntimeError("Gemini unavailable")
    )

    with pytest.raises(
        RuntimeError,
        match="Gemini unavailable",
    ):
        await client.generate("Example prompt")

    await client.close()


@pytest.mark.asyncio
async def test_close_closes_async_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, async_client = create_client(monkeypatch)

    await client.close()

    async_client.aclose.assert_awaited_once_with()


@pytest.mark.parametrize(
    ("api_key", "model", "message"),
    [
        (" ", "gemini-test-model", "api_key must not be empty"),
        ("test-api-key", " ", "model must not be empty"),
    ],
)
def test_init_rejects_empty_configuration(
    api_key: str,
    model: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        GeminiClient(
            api_key=api_key,
            model=model,
        )


def test_init_rejects_negative_temperature() -> None:
    with pytest.raises(
        ValueError,
        match="temperature must not be negative",
    ):
        GeminiClient(
            api_key="test-api-key",
            model="gemini-test-model",
            temperature=-0.1,
        )
