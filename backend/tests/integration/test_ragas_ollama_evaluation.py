import httpx
import pytest

from app.clients.embedding_client import EmbeddingClient
from app.core.config import settings
from app.evaluation.ragas_evaluator_factory import (
    create_ragas_evaluator,
)


pytestmark = [
    pytest.mark.external,
    pytest.mark.ollama,
    pytest.mark.asyncio,
]


async def test_ragas_uses_ollama_as_judge() -> None:
    if settings.generation_provider != "ollama":
        pytest.skip(
            "No local Ollama model is configured"
        )

    try:
        async with httpx.AsyncClient(
            base_url=settings.ollama_base_url,
            timeout=5.0,
        ) as client:
            response = await client.get("/api/tags")
            response.raise_for_status()
    except httpx.RequestError:
        pytest.skip("Ollama server is not available")

    embedding_client = EmbeddingClient(
        base_url=settings.embedding_service_url,
    )

    try:
        evaluator = create_ragas_evaluator(
            provider="ollama",
            model=settings.generation_model,
            embedding_client=embedding_client,
            temperature=0.0,
            answer_relevancy_strictness=1,
            ollama_base_url=settings.ollama_base_url,
        )

        result = await evaluator.evaluate(
            question="What is the capital of France?",
            answer="The capital of France is Paris.",
            contexts=[
                (
                    "Paris is the capital and largest "
                    "city of France."
                )
            ],
        )
    finally:
        await embedding_client.close()

    assert 0.0 <= result.faithfulness <= 1.0
    assert -1.0 <= result.answer_relevancy <= 1.0
