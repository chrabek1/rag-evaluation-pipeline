import pytest

from app.clients.gemini_client import GeminiClient
from app.core.config import settings
from app.models.chunk import Chunk
from app.models.retrieved_chunk import RetrievedChunk
from app.services.generation_service import GenerationService


pytestmark = [
    pytest.mark.external,
    pytest.mark.asyncio,
]


async def test_generation_uses_gemini() -> None:
    if settings.gemini_api_key is None:
        pytest.skip("GEMINI_API_KEY is not configured")

    api_key = settings.gemini_api_key.get_secret_value()

    if not api_key.strip():
        pytest.skip("GEMINI_API_KEY is empty")

    client = GeminiClient(
        api_key=api_key,
        model=settings.evaluation_model,
        temperature=settings.evaluation_temperature,
    )
    service = GenerationService(client)

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
    finally:
        await client.close()

    assert result.answer.strip()
    assert result.response.model == settings.evaluation_model
    assert result.latency_seconds >= 0.0

    if result.response.input_tokens is not None:
        assert result.response.input_tokens >= 0

    if result.response.output_tokens is not None:
        assert result.response.output_tokens >= 0
