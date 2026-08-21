from unittest.mock import Mock

import pytest
from instructor import Mode

from app.evaluation.ragas_evaluator_factory import (
    create_ragas_evaluator,
)


def test_create_ragas_evaluator_builds_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ragas_client = Mock()
    ragas_llm = Mock()
    embeddings = Mock()
    faithfulness = Mock()
    answer_relevancy = Mock()
    evaluator = Mock()
    embedding_client = Mock()

    client_constructor = Mock(
        return_value=ragas_client,
    )
    llm_constructor = Mock(
        return_value=ragas_llm,
    )
    embedding_constructor = Mock(
        return_value=embeddings,
    )
    faithfulness_constructor = Mock(
        return_value=faithfulness,
    )
    answer_relevancy_constructor = Mock(
        return_value=answer_relevancy,
    )
    evaluator_constructor = Mock(
        return_value=evaluator,
    )

    module = (
        "app.evaluation.ragas_evaluator_factory"
    )

    monkeypatch.setattr(
        f"{module}.from_litellm",
        client_constructor,
    )
    monkeypatch.setattr(
        f"{module}.llm_factory",
        llm_constructor,
    )
    monkeypatch.setattr(
        f"{module}.RagasEmbeddingAdapter",
        embedding_constructor,
    )
    monkeypatch.setattr(
        f"{module}.Faithfulness",
        faithfulness_constructor,
    )
    monkeypatch.setattr(
        f"{module}.AnswerRelevancy",
        answer_relevancy_constructor,
    )
    monkeypatch.setattr(
        f"{module}.GenerationEvaluator",
        evaluator_constructor,
    )

    result = create_ragas_evaluator(
        provider="gemini",
        model="gemini-test-model",
        embedding_client=embedding_client,
        temperature=0.2,
        gemini_api_key="test-api-key",
    )

    client_constructor.assert_called_once()
    assert (
        client_constructor.call_args.args[0]
        is not None
    )
    llm_constructor.assert_called_once_with(
        model="gemini/gemini-test-model",
        provider="google",
        client=ragas_client,
        adapter="litellm",
        api_key="test-api-key",
        temperature=0.2,
    )
    embedding_constructor.assert_called_once_with(
        embedding_client=embedding_client,
    )
    faithfulness_constructor.assert_called_once_with(
        llm=ragas_llm,
    )
    answer_relevancy_constructor.assert_called_once_with(
        llm=ragas_llm,
        embeddings=embeddings,
        strictness=3,
    )
    evaluator_constructor.assert_called_once_with(
        faithfulness=faithfulness,
        answer_relevancy=answer_relevancy,
    )
    assert result is evaluator


def test_create_ragas_evaluator_uses_custom_strictness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answer_relevancy_constructor = Mock(
        return_value=Mock()
    )
    module = "app.evaluation.ragas_evaluator_factory"

    monkeypatch.setattr(
        f"{module}.from_litellm",
        Mock(return_value=Mock()),
    )
    monkeypatch.setattr(
        f"{module}.llm_factory",
        Mock(return_value=Mock()),
    )
    monkeypatch.setattr(
        f"{module}.RagasEmbeddingAdapter",
        Mock(return_value=Mock()),
    )
    monkeypatch.setattr(
        f"{module}.Faithfulness",
        Mock(return_value=Mock()),
    )
    monkeypatch.setattr(
        f"{module}.AnswerRelevancy",
        answer_relevancy_constructor,
    )
    monkeypatch.setattr(
        f"{module}.GenerationEvaluator",
        Mock(return_value=Mock()),
    )

    create_ragas_evaluator(
        provider="gemini",
        model="gemini-test-model",
        embedding_client=Mock(),
        answer_relevancy_strictness=1,
        gemini_api_key="test-api-key",
    )

    assert (
        answer_relevancy_constructor.call_args.kwargs[
            "strictness"
        ]
        == 1
    )


@pytest.mark.parametrize(
    ("model", "message"),
    [
        ("", "model must not be empty"),
        (" ", "model must not be empty"),
    ],
)
def test_create_ragas_evaluator_rejects_empty_model(
    model: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        create_ragas_evaluator(
            provider="gemini",
            model=model,
            embedding_client=Mock(),
            gemini_api_key="test-api-key",
        )


def test_create_ragas_evaluator_rejects_negative_temperature() -> None:
    with pytest.raises(
        ValueError,
        match="temperature must not be negative",
    ):
        create_ragas_evaluator(
            provider="gemini",
            model="gemini-test-model",
            embedding_client=Mock(),
            temperature=-0.1,
            gemini_api_key="test-api-key",
        )


@pytest.mark.parametrize("strictness", [0, -1])
def test_create_ragas_evaluator_rejects_non_positive_strictness(
    strictness: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="answer_relevancy_strictness must be positive",
    ):
        create_ragas_evaluator(
            provider="gemini",
            model="gemini-test-model",
            embedding_client=Mock(),
            answer_relevancy_strictness=strictness,
            gemini_api_key="test-api-key",
        )


def test_create_ragas_evaluator_configures_ollama(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ragas_client = Mock()
    ragas_llm = Mock()
    llm_constructor = Mock(return_value=ragas_llm)
    module = "app.evaluation.ragas_evaluator_factory"

    client_constructor = Mock(return_value=ragas_client)
    monkeypatch.setattr(
        f"{module}.from_litellm",
        client_constructor,
    )
    monkeypatch.setattr(
        f"{module}.llm_factory",
        llm_constructor,
    )
    monkeypatch.setattr(
        f"{module}.RagasEmbeddingAdapter",
        Mock(return_value=Mock()),
    )
    monkeypatch.setattr(
        f"{module}.Faithfulness",
        Mock(return_value=Mock()),
    )
    monkeypatch.setattr(
        f"{module}.AnswerRelevancy",
        Mock(return_value=Mock()),
    )
    monkeypatch.setattr(
        f"{module}.GenerationEvaluator",
        Mock(return_value=Mock()),
    )

    create_ragas_evaluator(
        provider=" OLLAMA ",
        model="llama3.2:3b",
        embedding_client=Mock(),
        temperature=0.1,
        ollama_base_url=(
            "http://host.docker.internal:11434/"
        ),
    )

    llm_constructor.assert_called_once_with(
        model="ollama/llama3.2:3b",
        provider="ollama",
        client=ragas_client,
        adapter="litellm",
        temperature=0.1,
        api_base="http://host.docker.internal:11434",
    )
    client_constructor.assert_called_once()
    assert client_constructor.call_args.kwargs == {
        "mode": Mode.JSON_SCHEMA,
    }


@pytest.mark.parametrize(
    ("provider", "kwargs", "message"),
    [
        (
            "gemini",
            {},
            "gemini_api_key is required for Gemini",
        ),
        (
            "gemini",
            {"gemini_api_key": " "},
            "gemini_api_key is required for Gemini",
        ),
        (
            "ollama",
            {},
            "ollama_base_url is required for Ollama",
        ),
        (
            "ollama",
            {"ollama_base_url": " "},
            "ollama_base_url is required for Ollama",
        ),
    ],
)
def test_create_ragas_evaluator_requires_provider_configuration(
    provider: str,
    kwargs: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        create_ragas_evaluator(
            provider=provider,
            model="test-model",
            embedding_client=Mock(),
            **kwargs,
        )


def test_create_ragas_evaluator_rejects_unknown_provider() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported evaluation provider: unknown",
    ):
        create_ragas_evaluator(
            provider="unknown",
            model="test-model",
            embedding_client=Mock(),
        )
