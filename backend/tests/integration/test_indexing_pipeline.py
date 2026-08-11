from pathlib import Path
from unittest.mock import AsyncMock

import asyncpg
import pytest

from app.loaders.csv_chunk_loader import CsvChunkLoader
from app.repositories.chunk_repository import ChunkRepository
from app.services.indexing_service import IndexingService


@pytest.mark.asyncio
async def test_indexing_pipeline_indexes_csv_chunks_into_database(
    tmp_path: Path,
    db_pool: asyncpg.Pool,
) -> None:
    csv_path = tmp_path / "chunks.csv"
    csv_path.write_text(
        "filename,content\n"
        "doc-a.pdf,First chunk\n"
        "doc-a.pdf,Second chunk\n"
        "doc-b.pdf,Third chunk\n",
        encoding="utf-8",
    )

    loader = CsvChunkLoader()
    chunks = loader.load(csv_path)

    embedding_client = AsyncMock()
    embedding_client.embed.side_effect = [
        [
            [0.1] * 1024,
            [0.2] * 1024,
        ],
        [
            [0.3] * 1024,
        ],
    ]

    repository = ChunkRepository(db_pool)

    indexing_service = IndexingService(
        embedding_client=embedding_client,
        chunk_repository=repository,
        batch_size=2,
    )

    await indexing_service.index(chunks)

    async with db_pool.acquire() as connection:
        rows = await connection.fetch(
            """
            SELECT chunk_id, filename, content
            FROM chunks
            ORDER BY chunk_id
            """
        )

    assert len(rows) == 3

    assert rows[0]["chunk_id"] == "doc-a.pdf_0001"
    assert rows[0]["filename"] == "doc-a.pdf"
    assert rows[0]["content"] == "First chunk"

    assert rows[1]["chunk_id"] == "doc-a.pdf_0002"
    assert rows[1]["filename"] == "doc-a.pdf"
    assert rows[1]["content"] == "Second chunk"

    assert rows[2]["chunk_id"] == "doc-b.pdf_0001"
    assert rows[2]["filename"] == "doc-b.pdf"
    assert rows[2]["content"] == "Third chunk"

    assert embedding_client.embed.await_count == 2
    
@pytest.mark.asyncio
async def test_indexing_pipeline_is_idempotent(
    tmp_path: Path,
    db_pool: asyncpg.Pool,
) -> None:
    csv_path = tmp_path / "chunks.csv"
    csv_path.write_text(
        "filename,content\n"
        "doc-a.pdf,First chunk\n"
        "doc-a.pdf,Second chunk\n"
        "doc-b.pdf,Third chunk\n",
        encoding="utf-8",
    )

    loader = CsvChunkLoader()
    chunks = loader.load(csv_path)

    embedding_client = AsyncMock()
    embedding_client.embed.return_value = [
        [0.1] * 1024,
        [0.2] * 1024,
        [0.3] * 1024,
    ]

    repository = ChunkRepository(db_pool)

    indexing_service = IndexingService(
        embedding_client=embedding_client,
        chunk_repository=repository,
        batch_size=3,
    )

    await indexing_service.index(chunks)
    await indexing_service.index(chunks)

    async with db_pool.acquire() as connection:
        count = await connection.fetchval(
            "SELECT COUNT(*) FROM chunks"
        )

    assert count == 3