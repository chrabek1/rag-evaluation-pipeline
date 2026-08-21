import math

import pytest

from app.clients.embedding_client import EmbeddingClient
from app.core.config import settings


pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.asyncio,
]


async def test_embedding_client_matches_service_contract() -> None:
    client = EmbeddingClient(
        base_url=settings.embedding_service_url,
    )

    try:
        model_info = await client.get_info()
        vectors = await client.embed(
            ["First text", "Second text"]
        )
    finally:
        await client.close()

    assert model_info.model.strip()
    assert model_info.embedding_dimension > 0
    assert len(vectors) == 2

    for vector in vectors:
        assert len(vector) == model_info.embedding_dimension
        assert all(math.isfinite(value) for value in vector)
