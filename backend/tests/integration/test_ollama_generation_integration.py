import httpx
import pytest

from app.clients.ollama_client import OllamaClient
from app.core.config import settings
from app.models.chunk import Chunk
from app.models.retrieved_chunk import RetrievedChunk
from app.services.generation_service import GenerationService


pytestmark = [
    pytest.mark.external,
    pytest.mark.ollama,
    pytest.mark.asyncio,
]


async def test_generation_uses_local_ollama() -> None:
    if settings.generation_provider != "ollama":
        pytest.skip(
            "GENERATION_PROVIDER is not configured as ollama"
        )

    client = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.generation_model,
        temperature=settings.generation_temperature,
    )
    service = GenerationService(client)

    try:
        try:
            result = await service.generate(
                question="What is the capital of France?",
                retrieved_chunks=[
                    RetrievedChunk(
                        chunk_id="cities.pdf_0001",
                        chunk=Chunk(
                            filename="cities.pdf",
                            content=(
                                "Paris is the capital of France."
                            ),
                        ),
                        score=1.0,
                    )
                ],
            )
        except httpx.ConnectError:
            pytest.skip("Ollama server is not available")
    finally:
        await client.close()

    assert result.answer.strip()
    assert result.response.model == settings.generation_model
    assert result.latency_seconds >= 0.0

    if result.response.input_tokens is not None:
        assert result.response.input_tokens >= 0

    if result.response.output_tokens is not None:
        assert result.response.output_tokens >= 0
