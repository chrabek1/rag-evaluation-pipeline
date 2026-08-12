from unittest.mock import AsyncMock

import pytest

from app.models.chunk import Chunk
from app.models.retrieved_chunk import RetrievedChunk
from app.services.retrieval_service import RetrievalService


@pytest.mark.asyncio
async def test_retrieve_embeds_query_and_returns_search_results() -> None:
    query_embedding = [0.1] * 1024

    expected_results = [
        RetrievedChunk(
            chunk_id="doc.pdf_0001",
            chunk=Chunk(
                filename="doc.pdf",
                content="Relevant content",
            ),
            score=0.9,
        )
    ]

    embedding_client = AsyncMock()
    embedding_client.embed.return_value = [query_embedding]

    chunk_repository = AsyncMock()
    chunk_repository.search.return_value = expected_results

    service = RetrievalService(
        embedding_client=embedding_client,
        chunk_repository=chunk_repository,
    )

    results = await service.retrieve(
        query="What is RAG?",
        top_k=5,
    )

    assert results == expected_results

    embedding_client.embed.assert_awaited_once_with(
        ["What is RAG?"]
    )

    chunk_repository.search.assert_awaited_once_with(
        query_embedding=query_embedding,
        top_k=5,
    )
    
@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["", " ", "\t", "\n"])
async def test_retrieve_rejects_empty_query(query: str) -> None:
    embedding_client = AsyncMock()
    chunk_repository = AsyncMock()

    service = RetrievalService(
        embedding_client=embedding_client,
        chunk_repository=chunk_repository,
    )

    with pytest.raises(ValueError, match="query must not be empty"):
        await service.retrieve(
            query=query,
            top_k=5,
        )

    embedding_client.embed.assert_not_awaited()
    chunk_repository.search.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("top_k", [0, -1])
async def test_retrieve_rejects_non_positive_top_k(top_k: int) -> None:
    embedding_client = AsyncMock()
    chunk_repository = AsyncMock()

    service = RetrievalService(
        embedding_client=embedding_client,
        chunk_repository=chunk_repository,
    )

    with pytest.raises(
        ValueError,
        match="top_k must be greater than 0",
    ):
        await service.retrieve(
            query="What is RAG?",
            top_k=top_k,
        )

    embedding_client.embed.assert_not_awaited()
    chunk_repository.search.assert_not_awaited()


@pytest.mark.asyncio
async def test_retrieve_rejects_invalid_embedding_count() -> None:
    embedding_client = AsyncMock()
    embedding_client.embed.return_value = []

    chunk_repository = AsyncMock()

    service = RetrievalService(
        embedding_client=embedding_client,
        chunk_repository=chunk_repository,
    )

    with pytest.raises(
        ValueError,
        match="Expected exactly one embedding for query",
    ):
        await service.retrieve(
            query="What is RAG?",
            top_k=5,
        )

    embedding_client.embed.assert_awaited_once_with(
        ["What is RAG?"]
    )
    chunk_repository.search.assert_not_awaited()