import pytest

from app.models.generation_result import GenerationResult
from app.models.llm_response import LLMResponse


def test_generation_result_stores_data() -> None:
    response = LLMResponse(
        text="Generated answer",
        model="test-model",
        input_tokens=20,
        output_tokens=8,
    )

    result = GenerationResult(
        response=response,
        latency_seconds=1.25,
    )

    assert result.response == response
    assert result.answer == "Generated answer"
    assert result.latency_seconds == 1.25


def test_generation_result_allows_zero_latency() -> None:
    result = GenerationResult(
        response=LLMResponse(
            text="Generated answer",
            model="test-model",
        ),
        latency_seconds=0.0,
    )

    assert result.latency_seconds == 0.0


def test_generation_result_rejects_negative_latency() -> None:
    with pytest.raises(
        ValueError,
        match="latency_seconds must not be negative",
    ):
        GenerationResult(
            response=LLMResponse(
                text="Generated answer",
                model="test-model",
            ),
            latency_seconds=-0.1,
        )