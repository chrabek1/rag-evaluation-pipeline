from unittest.mock import AsyncMock, Mock

import pytest

from app.evaluation.ragas_embedding_adapter import (
    RagasEmbeddingAdapter,
)


@pytest.mark.asyncio
async def test_aembed_text_returns_single_vector() -> None:
    embedding_client = Mock()
    embedding_client.embed = AsyncMock(
        return_value=[[0.1, 0.2, 0.3]]
    )

    adapter = RagasEmbeddingAdapter(
        embedding_client=embedding_client
    )

    result = await adapter.aembed_text("Example text")

    embedding_client.embed.assert_awaited_once_with(
        ["Example text"]
    )
    assert result == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_aembed_texts_returns_vectors() -> None:
    embedding_client = Mock()
    embedding_client.embed = AsyncMock(
        return_value=[
            [0.1, 0.2],
            [0.3, 0.4],
        ]
    )

    adapter = RagasEmbeddingAdapter(
        embedding_client=embedding_client
    )

    result = await adapter.aembed_texts(
        ["First text", "Second text"]
    )

    embedding_client.embed.assert_awaited_once_with(
        ["First text", "Second text"]
    )
    assert result == [
        [0.1, 0.2],
        [0.3, 0.4],
    ]


@pytest.mark.asyncio
async def test_aembed_texts_returns_empty_list_without_request() -> None:
    embedding_client = Mock()
    embedding_client.embed = AsyncMock()

    adapter = RagasEmbeddingAdapter(
        embedding_client=embedding_client
    )

    result = await adapter.aembed_texts([])

    assert result == []
    embedding_client.embed.assert_not_awaited()


@pytest.mark.asyncio
async def test_aembed_text_rejects_empty_text() -> None:
    embedding_client = Mock()
    embedding_client.embed = AsyncMock()

    adapter = RagasEmbeddingAdapter(
        embedding_client=embedding_client
    )

    with pytest.raises(
        ValueError,
        match="text must not be empty",
    ):
        await adapter.aembed_text(" ")

    embedding_client.embed.assert_not_awaited()


@pytest.mark.asyncio
async def test_aembed_texts_rejects_empty_text() -> None:
    embedding_client = Mock()
    embedding_client.embed = AsyncMock()

    adapter = RagasEmbeddingAdapter(
        embedding_client=embedding_client
    )

    with pytest.raises(
        ValueError,
        match="texts must not contain empty text",
    ):
        await adapter.aembed_texts(
            ["Valid text", " "]
        )

    embedding_client.embed.assert_not_awaited()


@pytest.mark.asyncio
async def test_aembed_text_rejects_invalid_vector_count() -> None:
    embedding_client = Mock()
    embedding_client.embed = AsyncMock(
        return_value=[]
    )

    adapter = RagasEmbeddingAdapter(
        embedding_client=embedding_client
    )

    with pytest.raises(
        ValueError,
        match="Embedding service must return one vector",
    ):
        await adapter.aembed_text("Example text")


@pytest.mark.asyncio
async def test_aembed_texts_rejects_invalid_vector_count() -> None:
    embedding_client = Mock()
    embedding_client.embed = AsyncMock(
        return_value=[[0.1, 0.2]]
    )

    adapter = RagasEmbeddingAdapter(
        embedding_client=embedding_client
    )

    with pytest.raises(
        ValueError,
        match="Embedding count must match text count",
    ):
        await adapter.aembed_texts(
            ["First text", "Second text"]
        )


@pytest.mark.asyncio
async def test_aembed_text_propagates_embedding_error() -> None:
    embedding_client = Mock()
    embedding_client.embed = AsyncMock(
        side_effect=RuntimeError(
            "Embedding service unavailable"
        )
    )
    adapter = RagasEmbeddingAdapter(
        embedding_client=embedding_client
    )

    with pytest.raises(
        RuntimeError,
        match="Embedding service unavailable",
    ):
        await adapter.aembed_text("Example text")


def test_embed_text_rejects_synchronous_usage() -> None:
    adapter = RagasEmbeddingAdapter(
        embedding_client=Mock()
    )

    with pytest.raises(
        RuntimeError,
        match="Synchronous embedding is not supported",
    ):
        adapter.embed_text("Example text")
