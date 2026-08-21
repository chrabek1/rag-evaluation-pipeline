import pytest

from app.clients.embedding_client import EmbeddingClient
from app.core.config import settings
from app.evaluation.ragas_evaluator_factory import (
    create_ragas_evaluator,
)


pytestmark = [
    pytest.mark.external,
    pytest.mark.asyncio,
]


async def test_ragas_evaluates_grounded_relevant_answer() -> None:
    if settings.gemini_api_key is None:
        pytest.skip("GEMINI_API_KEY is not configured")

    api_key = (
        settings.gemini_api_key.get_secret_value()
    )

    if not api_key.strip():
        pytest.skip("GEMINI_API_KEY is empty")

    embedding_client = EmbeddingClient(
        base_url=settings.embedding_service_url,
    )

    try:
        evaluator = create_ragas_evaluator(
            provider="gemini",
            model=settings.evaluation_model,
            embedding_client=embedding_client,
            temperature=settings.evaluation_temperature,
            answer_relevancy_strictness=1,
            gemini_api_key=api_key,
        )

        result = await evaluator.evaluate(
            question=(
                "What is the capital of France?"
            ),
            answer=(
                "The capital of France is Paris."
            ),
            contexts=[
                (
                    "Paris is the capital and "
                    "largest city of France."
                )
            ],
        )
    finally:
        await embedding_client.close()

    assert 0.0 <= result.faithfulness <= 1.0
    assert -1.0 <= result.answer_relevancy <= 1.0

    assert result.faithfulness >= 0.8
    assert result.answer_relevancy >= 0.8
