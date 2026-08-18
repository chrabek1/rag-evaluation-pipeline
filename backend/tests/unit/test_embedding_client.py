from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from app.clients.embedding_client import EmbeddingClient


@pytest.mark.asyncio
async def test_embed_sends_texts_and_returns_vectors() -> None:
    client = EmbeddingClient("http://embedding-service:8001")

    response = Mock()
    response.json.return_value = {
        "vectors": [
            [0.1, 0.2],
            [0.3, 0.4],
        ]
    }
    response.raise_for_status.return_value = None

    client._client.post = AsyncMock(return_value=response)

    vectors = await client.embed(["first text", "second text"])

    client._client.post.assert_awaited_once_with(
        "/embed",
        json={"texts": ["first text", "second text"]},
    )
    response.raise_for_status.assert_called_once_with()

    assert vectors == [
        [0.1, 0.2],
        [0.3, 0.4],
    ]

    await client.close()


@pytest.mark.asyncio
async def test_embed_with_empty_texts_returns_empty_list_without_request() -> None:
    client = EmbeddingClient("http://embedding-service:8001")
    client._client.post = AsyncMock()

    vectors = await client.embed([])

    assert vectors == []
    client._client.post.assert_not_awaited()

    await client.close()


@pytest.mark.asyncio
async def test_embed_propagates_http_error() -> None:
    client = EmbeddingClient("http://embedding-service:8001")

    request = httpx.Request(
        "POST",
        "http://embedding-service:8001/embed",
    )
    response = httpx.Response(
        status_code=500,
        request=request,
    )

    client._client.post = AsyncMock(return_value=response)

    with pytest.raises(httpx.HTTPStatusError):
        await client.embed(["example text"])

    await client.close()
    
@pytest.mark.asyncio
async def test_get_info_returns_embedding_model_info() -> None:
    client = EmbeddingClient("http://embedding-service:8001")

    response = Mock()
    response.json.return_value = {
        "model": "BAAI/bge-m3",
        "embedding_dimension": 1024,
    }
    response.raise_for_status.return_value = None

    client._client.get = AsyncMock(return_value=response)

    model_info = await client.get_info()

    client._client.get.assert_awaited_once_with("/info")
    response.raise_for_status.assert_called_once_with()

    assert model_info.model == "BAAI/bge-m3"
    assert model_info.embedding_dimension == 1024

    await client.close()