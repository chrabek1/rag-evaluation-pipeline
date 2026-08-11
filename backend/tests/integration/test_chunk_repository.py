import asyncpg
import pytest

from app.models.chunk import Chunk
from app.models.embedded_chunk import EmbeddedChunk
from app.repositories.chunk_repository import ChunkRepository


@pytest.mark.asyncio
async def test_add_many_inserts_chunks_into_database(
    db_pool: asyncpg.Pool,
) -> None:
    repository = ChunkRepository(db_pool)

    chunks = [
        EmbeddedChunk(
            chunk_id="test-doc.pdf_0001",
            chunk=Chunk(
                filename="test-doc.pdf",
                content="First test chunk",
            ),
            embedding=[0.1] * 1024,
        ),
        EmbeddedChunk(
            chunk_id="test-doc.pdf_0002",
            chunk=Chunk(
                filename="test-doc.pdf",
                content="Second test chunk",
            ),
            embedding=[0.2] * 1024,
        ),
    ]

    await repository.add_many(chunks)

    async with db_pool.acquire() as connection:
        rows = await connection.fetch(
            """
            SELECT chunk_id, filename, content
            FROM chunks
            ORDER BY chunk_id
            """
        )

    assert len(rows) == 2

    assert rows[0]["chunk_id"] == "test-doc.pdf_0001"
    assert rows[0]["filename"] == "test-doc.pdf"
    assert rows[0]["content"] == "First test chunk"

    assert rows[1]["chunk_id"] == "test-doc.pdf_0002"
    assert rows[1]["filename"] == "test-doc.pdf"
    assert rows[1]["content"] == "Second test chunk"


@pytest.mark.asyncio
async def test_add_many_updates_existing_chunk_on_conflict(
    db_pool: asyncpg.Pool,
) -> None:
    repository = ChunkRepository(db_pool)

    original_chunk = EmbeddedChunk(
        chunk_id="test-doc.pdf_0001",
        chunk=Chunk(
            filename="test-doc.pdf",
            content="Original content",
        ),
        embedding=[0.1] * 1024,
    )

    updated_chunk = EmbeddedChunk(
        chunk_id="test-doc.pdf_0001",
        chunk=Chunk(
            filename="test-doc.pdf",
            content="Updated content",
        ),
        embedding=[0.9] * 1024,
    )

    await repository.add_many([original_chunk])
    await repository.add_many([updated_chunk])

    async with db_pool.acquire() as connection:
        row = await connection.fetchrow(
            """
            SELECT
                chunk_id,
                filename,
                content,
                embedding <=> $2::vector AS distance
            FROM chunks
            WHERE chunk_id = $1
            """,
            "test-doc.pdf_0001",
            [0.9] * 1024,
        )
        count = await connection.fetchval(
            "SELECT COUNT(*) FROM chunks"
        )

    assert count == 1
    assert row["chunk_id"] == "test-doc.pdf_0001"
    assert row["filename"] == "test-doc.pdf"
    assert row["content"] == "Updated content"
    assert row["distance"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_add_many_with_empty_list_does_not_modify_database(
    db_pool: asyncpg.Pool,
) -> None:
    repository = ChunkRepository(db_pool)

    await repository.add_many([])

    async with db_pool.acquire() as connection:
        count = await connection.fetchval(
            "SELECT COUNT(*) FROM chunks"
        )

    assert count == 0