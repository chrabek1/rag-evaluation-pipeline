import asyncpg
import pytest

from app.clients.embedding_client import EmbeddingClient
from app.core.config import settings
from app.models.chunk import Chunk
from app.repositories.chunk_repository import ChunkRepository
from app.services.indexing_service import IndexingService


pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.asyncio,
]


async def test_indexing_uses_real_embedding_service(
    db_pool: asyncpg.Pool,
) -> None:
    embedding_client = EmbeddingClient(
        base_url=settings.embedding_service_url,
    )
    service = IndexingService(
        embedding_client=embedding_client,
        chunk_repository=ChunkRepository(db_pool),
        batch_size=2,
    )
    chunks = [
        Chunk(
            filename="cities.pdf",
            content="Paris is the capital of France.",
        ),
        Chunk(
            filename="cities.pdf",
            content="Berlin is the capital of Germany.",
        ),
    ]

    try:
        await service.index(chunks)
        await service.index(chunks)
    finally:
        await embedding_client.close()

    async with db_pool.acquire() as connection:
        rows = await connection.fetch(
            """
            SELECT chunk_id, vector_dims(embedding) AS dimension
            FROM chunks
            ORDER BY chunk_id
            """
        )

    assert [row["chunk_id"] for row in rows] == [
        "cities.pdf_0001",
        "cities.pdf_0002",
    ]
    assert all(row["dimension"] == 1024 for row in rows)
