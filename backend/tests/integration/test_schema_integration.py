import os

import asyncpg
import pytest

from app.db.schema import initialize_schema


pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]

TEST_DATABASE_URL = os.environ["TEST_DATABASE_URL"]
EMBEDDING_DIMENSION = 1024


async def test_schema_uses_configured_vector_dimension(
    db_pool: asyncpg.Pool,
) -> None:
    await initialize_schema(
        TEST_DATABASE_URL,
        EMBEDDING_DIMENSION,
    )

    async with db_pool.acquire() as connection:
        embedding_type = await connection.fetchval(
            """
            SELECT format_type(atttypid, atttypmod)
            FROM pg_attribute
            WHERE attrelid = 'chunks'::regclass
                AND attname = 'embedding'
                AND NOT attisdropped
            """
        )

    assert embedding_type == "vector(1024)"


async def test_schema_rejects_real_dimension_mismatch(
    db_pool: asyncpg.Pool,
) -> None:
    with pytest.raises(
        RuntimeError,
        match="Embedding dimension mismatch",
    ):
        await initialize_schema(
            TEST_DATABASE_URL,
            EMBEDDING_DIMENSION - 1,
        )


async def test_postgres_rejects_vector_with_wrong_dimension(
    db_pool: asyncpg.Pool,
) -> None:
    async with db_pool.acquire() as connection:
        with pytest.raises(asyncpg.DataError):
            await connection.execute(
                """
                INSERT INTO chunks (
                    chunk_id,
                    filename,
                    content,
                    embedding
                )
                VALUES ($1, $2, $3, $4)
                """,
                "wrong-dimension_0001",
                "document.pdf",
                "Example content",
                [0.1] * (EMBEDDING_DIMENSION - 1),
            )
