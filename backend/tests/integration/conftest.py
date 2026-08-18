import os
from collections.abc import AsyncIterator

import asyncpg
import pytest_asyncio

from app.db.database import create_pool
from app.db.schema import initialize_schema


TEST_DATABASE_URL = os.environ["TEST_DATABASE_URL"]
TEST_EMBEDDING_DIMENSION = 1024


@pytest_asyncio.fixture
async def db_pool() -> AsyncIterator[asyncpg.Pool]:
    await initialize_schema(
        TEST_DATABASE_URL,
        TEST_EMBEDDING_DIMENSION,
    )
    pool = await create_pool(TEST_DATABASE_URL)

    try:
        async with pool.acquire() as connection:
            await connection.execute("DELETE FROM chunks")

        yield pool
    finally:
        await pool.close()