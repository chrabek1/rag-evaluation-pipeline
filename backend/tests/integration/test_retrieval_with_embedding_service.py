import asyncpg
import pytest

from app.clients.embedding_client import EmbeddingClient
from app.core.config import settings
from app.models.chunk import Chunk
from app.repositories.chunk_repository import ChunkRepository
from app.services.indexing_service import IndexingService
from app.services.retrieval_service import RetrievalService


pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.asyncio,
]


async def test_retrieval_uses_real_embedding_service(
    db_pool: asyncpg.Pool,
) -> None:
    embedding_client = EmbeddingClient(
        base_url=settings.embedding_service_url,
    )
    repository = ChunkRepository(db_pool)
    indexing_service = IndexingService(
        embedding_client=embedding_client,
        chunk_repository=repository,
    )
    retrieval_service = RetrievalService(
        embedding_client=embedding_client,
        chunk_repository=repository,
    )
    exact_query = "Paris is the capital of France."

    try:
        await indexing_service.index(
            [
                Chunk(
                    filename="cities.pdf",
                    content=exact_query,
                ),
                Chunk(
                    filename="science.pdf",
                    content=(
                        "Photosynthesis converts light "
                        "into chemical energy."
                    ),
                ),
            ]
        )
        results = await retrieval_service.retrieve(
            query=exact_query,
            top_k=2,
        )
    finally:
        await embedding_client.close()

    assert len(results) == 2
    assert results[0].chunk_id == "cities.pdf_0001"
    assert results[0].score >= results[1].score
