import os
from collections.abc import AsyncIterator
from urllib.parse import urlparse

import asyncpg
import pytest_asyncio

from app.db.database import create_pool
from app.db.schema import initialize_schema


TEST_DATABASE_URL = os.environ["TEST_DATABASE_URL"]
TEST_EMBEDDING_DIMENSION = 1024


def _validate_test_database_url() -> None:
    database_name = urlparse(
        TEST_DATABASE_URL
    ).path.lstrip("/")
    production_database_url = os.environ.get(
        "DATABASE_URL"
    )

    if not database_name.endswith("_test"):
        raise RuntimeError(
            "Integration tests require a database whose "
            "name ends with '_test'"
        )

    if (
        production_database_url is not None
        and TEST_DATABASE_URL == production_database_url
    ):
        raise RuntimeError(
            "TEST_DATABASE_URL must differ from DATABASE_URL"
        )


_validate_test_database_url()


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
