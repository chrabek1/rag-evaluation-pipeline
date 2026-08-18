from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.models.embedding_model_info import EmbeddingModelInfo
from scripts import index_chunks


@pytest.mark.asyncio
async def test_run_indexing_uses_embedding_service_dimension() -> None:
    chunks = [Mock()]

    loader = Mock()
    loader.load.return_value = chunks

    embedding_client = AsyncMock()
    embedding_client.get_info.return_value = EmbeddingModelInfo(
        model="test-model",
        embedding_dimension=768,
    )

    pool = AsyncMock()
    repository = Mock()
    indexing_service = AsyncMock()

    with (
        patch.object(
            index_chunks,
            "CsvChunkLoader",
            return_value=loader,
        ),
        patch.object(
            index_chunks,
            "EmbeddingClient",
            return_value=embedding_client,
        ),
        patch.object(
            index_chunks,
            "initialize_schema",
            new=AsyncMock(),
        ) as initialize_schema_mock,
        patch.object(
            index_chunks,
            "create_pool",
            new=AsyncMock(return_value=pool),
        ),
        patch.object(
            index_chunks,
            "ChunkRepository",
            return_value=repository,
        ),
        patch.object(
            index_chunks,
            "IndexingService",
            return_value=indexing_service,
        ),
    ):
        await index_chunks.run_indexing()

    embedding_client.get_info.assert_awaited_once_with()

    initialize_schema_mock.assert_awaited_once_with(
        index_chunks.settings.database_url,
        768,
    )

    indexing_service.index.assert_awaited_once_with(chunks)
    embedding_client.close.assert_awaited_once_with()
    pool.close.assert_awaited_once_with()