from unittest.mock import AsyncMock

import asyncpg
import pytest

from app.models.chunk import Chunk
from app.models.embedded_chunk import EmbeddedChunk
from app.repositories.chunk_repository import ChunkRepository
from app.services.retrieval_service import RetrievalService


@pytest.mark.asyncio
async def test_retrieval_pipeline_returns_top_k_chunks_in_similarity_order(
    db_pool: asyncpg.Pool,
) -> None:
    repository = ChunkRepository(db_pool)

    chunks = [
        EmbeddedChunk(
            chunk_id="doc.pdf_0001",
            chunk=Chunk(
                filename="doc.pdf",
                content="Most similar chunk",
            ),
            embedding=[1.0, 0.0] + [0.0] * 1022,
        ),
        EmbeddedChunk(
            chunk_id="doc.pdf_0002",
            chunk=Chunk(
                filename="doc.pdf",
                content="Second most similar chunk",
            ),
            embedding=[0.8, 0.6] + [0.0] * 1022,
        ),
        EmbeddedChunk(
            chunk_id="doc.pdf_0003",
            chunk=Chunk(
                filename="doc.pdf",
                content="Least similar chunk",
            ),
            embedding=[0.0, 1.0] + [0.0] * 1022,
        ),
    ]

    await repository.add_many(chunks)

    query_embedding = [1.0, 0.0] + [0.0] * 1022

    embedding_client = AsyncMock()
    embedding_client.embed.return_value = [query_embedding]

    retrieval_service = RetrievalService(
        embedding_client=embedding_client,
        chunk_repository=repository,
    )

    results = await retrieval_service.retrieve(
        query="test query",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].chunk_id == "doc.pdf_0001"
    assert results[0].score == pytest.approx(1.0)
    assert results[1].chunk_id == "doc.pdf_0002"
    assert results[1].score == pytest.approx(0.8)

    embedding_client.embed.assert_awaited_once_with(["test query"])
    
@pytest.mark.asyncio
async def test_retrieval_pipeline_returns_all_chunks_when_top_k_exceeds_count(
    db_pool: asyncpg.Pool,
) -> None:
    repository = ChunkRepository(db_pool)

    chunks = [
        EmbeddedChunk(
            chunk_id="doc.pdf_0001",
            chunk=Chunk(
                filename="doc.pdf",
                content="First chunk",
            ),
            embedding=[1.0, 0.0] + [0.0] * 1022,
        ),
        EmbeddedChunk(
            chunk_id="doc.pdf_0002",
            chunk=Chunk(
                filename="doc.pdf",
                content="Second chunk",
            ),
            embedding=[0.0, 1.0] + [0.0] * 1022,
        ),
    ]

    await repository.add_many(chunks)

    query_embedding = [1.0, 0.0] + [0.0] * 1022

    embedding_client = AsyncMock()
    embedding_client.embed.return_value = [query_embedding]

    retrieval_service = RetrievalService(
        embedding_client=embedding_client,
        chunk_repository=repository,
    )

    results = await retrieval_service.retrieve(
        query="test query",
        top_k=5,
    )

    assert len(results) == 2
    assert results[0].chunk_id == "doc.pdf_0001"
    assert results[1].chunk_id == "doc.pdf_0002"


@pytest.mark.asyncio
async def test_retrieval_pipeline_returns_empty_list_when_database_is_empty(
    db_pool: asyncpg.Pool,
) -> None:
    repository = ChunkRepository(db_pool)

    query_embedding = [1.0, 0.0] + [0.0] * 1022

    embedding_client = AsyncMock()
    embedding_client.embed.return_value = [query_embedding]

    retrieval_service = RetrievalService(
        embedding_client=embedding_client,
        chunk_repository=repository,
    )

    results = await retrieval_service.retrieve(
        query="test query",
        top_k=5,
    )

    assert results == []