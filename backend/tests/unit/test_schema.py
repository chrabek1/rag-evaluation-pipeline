from unittest.mock import AsyncMock, patch

import pytest

from app.db.schema import initialize_schema


DATABASE_URL = "postgresql://user:password@localhost/test"


@pytest.mark.asyncio
async def test_initialize_schema_uses_requested_embedding_dimension() -> None:
    connection = AsyncMock()
    connection.fetchval.return_value = 768

    with patch(
        "app.db.schema.asyncpg.connect",
        new=AsyncMock(return_value=connection),
    ) as connect_mock:
        await initialize_schema(DATABASE_URL, 768)

    connect_mock.assert_awaited_once_with(DATABASE_URL)

    schema_query = connection.execute.await_args.args[0]
    assert "VECTOR(768)" in schema_query

    connection.fetchval.assert_awaited_once()
    connection.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_initialize_schema_rejects_dimension_mismatch() -> None:
    connection = AsyncMock()
    connection.fetchval.return_value = 1024

    with patch(
        "app.db.schema.asyncpg.connect",
        new=AsyncMock(return_value=connection),
    ):
        with pytest.raises(
            RuntimeError,
            match=(
                "model produces 768-dimensional vectors, "
                "but the database expects 1024"
            ),
        ):
            await initialize_schema(DATABASE_URL, 768)

    connection.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_initialize_schema_fails_when_dimension_cannot_be_determined() -> None:
    connection = AsyncMock()
    connection.fetchval.return_value = None

    with patch(
        "app.db.schema.asyncpg.connect",
        new=AsyncMock(return_value=connection),
    ):
        with pytest.raises(
            RuntimeError,
            match="Could not determine the dimension",
        ):
            await initialize_schema(DATABASE_URL, 768)

    connection.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_initialize_schema_rejects_non_positive_dimension() -> None:
    connect_mock = AsyncMock()

    with patch(
        "app.db.schema.asyncpg.connect",
        new=connect_mock,
    ):
        with pytest.raises(
            ValueError,
            match="embedding_dimension must be greater than 0",
        ):
            await initialize_schema(DATABASE_URL, 0)

    connect_mock.assert_not_awaited()