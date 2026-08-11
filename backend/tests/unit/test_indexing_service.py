from unittest.mock import AsyncMock

import pytest

from app.models.chunk import Chunk
from app.services.indexing_service import IndexingService


@pytest.mark.asyncio
async def test_index_creates_stable_chunk_ids_and_batches() -> None:
    embedding_client = AsyncMock()
    chunk_repository = AsyncMock()

    embedding_client.embed.side_effect = [
        [
            [0.1, 0.2],
            [0.3, 0.4],
        ],
        [
            [0.5, 0.6],
        ],
    ]

    chunks = [
        Chunk(filename="doc-a.pdf", content="chunk 1"),
        Chunk(filename="doc-b.pdf", content="chunk 2"),
        Chunk(filename="doc-a.pdf", content="chunk 3"),
    ]

    service = IndexingService(
        embedding_client=embedding_client,
        chunk_repository=chunk_repository,
        batch_size=2,
    )

    await service.index(chunks)

    assert embedding_client.embed.await_count == 2
    assert embedding_client.embed.await_args_list[0].args[0] == [
        "chunk 1",
        "chunk 2",
    ]
    assert embedding_client.embed.await_args_list[1].args[0] == [
        "chunk 3",
    ]

    assert chunk_repository.add_many.await_count == 2

    first_batch = chunk_repository.add_many.await_args_list[0].args[0]
    second_batch = chunk_repository.add_many.await_args_list[1].args[0]

    assert [chunk.chunk_id for chunk in first_batch] == [
        "doc-a.pdf_0001",
        "doc-b.pdf_0001",
    ]
    assert [chunk.chunk_id for chunk in second_batch] == [
        "doc-a.pdf_0002",
    ]

    assert [chunk.chunk.content for chunk in first_batch] == [
        "chunk 1",
        "chunk 2",
    ]
    assert [chunk.chunk.content for chunk in second_batch] == [
        "chunk 3",
    ]

    assert [chunk.embedding for chunk in first_batch] == [
        [0.1, 0.2],
        [0.3, 0.4],
    ]
    assert [chunk.embedding for chunk in second_batch] == [
        [0.5, 0.6],
    ]
    
@pytest.mark.asyncio
async def test_index_rejects_mismatched_embedding_count() -> None:
    embedding_client = AsyncMock()
    chunk_repository = AsyncMock()

    embedding_client.embed.return_value = [
        [0.1, 0.2],
    ]

    chunks = [
        Chunk(filename="doc-a.pdf", content="chunk 1"),
        Chunk(filename="doc-a.pdf", content="chunk 2"),
    ]

    service = IndexingService(
        embedding_client=embedding_client,
        chunk_repository=chunk_repository,
        batch_size=2,
    )

    with pytest.raises(
        ValueError,
        match="Number of embeddings does not match number of chunks",
    ):
        await service.index(chunks)

    chunk_repository.add_many.assert_not_awaited()
    
@pytest.mark.asyncio
async def test_index_with_empty_chunks_does_nothing() -> None:
    embedding_client = AsyncMock()
    chunk_repository = AsyncMock()

    service = IndexingService(
        embedding_client=embedding_client,
        chunk_repository=chunk_repository,
        batch_size=2,
    )

    await service.index([])

    embedding_client.embed.assert_not_awaited()
    chunk_repository.add_many.assert_not_awaited()
    
def test_init_rejects_non_positive_batch_size() -> None:
    embedding_client = AsyncMock()
    chunk_repository = AsyncMock()

    with pytest.raises(ValueError, match="batch_size must be greater than 0"):
        IndexingService(
            embedding_client=embedding_client,
            chunk_repository=chunk_repository,
            batch_size=0,
        )