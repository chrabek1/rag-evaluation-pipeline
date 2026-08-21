import pytest

from app.models.llm_response import LLMResponse


def test_llm_response_stores_data() -> None:
    response = LLMResponse(
        text="Generated answer",
        model="test-model",
        input_tokens=10,
        output_tokens=5,
    )

    assert response.text == "Generated answer"
    assert response.model == "test-model"
    assert response.input_tokens == 10
    assert response.output_tokens == 5
    assert response.total_tokens == 15


def test_llm_response_allows_missing_token_usage() -> None:
    response = LLMResponse(
        text="Generated answer",
        model="test-model",
    )

    assert response.input_tokens is None
    assert response.output_tokens is None
    assert response.total_tokens is None


@pytest.mark.parametrize(
    "text",
    ["", " ", "\t", "\n"],
)
def test_llm_response_rejects_empty_text(
    text: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="text must not be empty",
    ):
        LLMResponse(
            text=text,
            model="test-model",
        )


@pytest.mark.parametrize(
    "model",
    ["", " ", "\t", "\n"],
)
def test_llm_response_rejects_empty_model(
    model: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="model must not be empty",
    ):
        LLMResponse(
            text="Generated answer",
            model=model,
        )


@pytest.mark.parametrize(
    ("input_tokens", "output_tokens", "message"),
    [
        (-1, 0, "input_tokens must not be negative"),
        (0, -1, "output_tokens must not be negative"),
    ],
)
def test_llm_response_rejects_negative_token_usage(
    input_tokens: int,
    output_tokens: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        LLMResponse(
            text="Generated answer",
            model="test-model",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )