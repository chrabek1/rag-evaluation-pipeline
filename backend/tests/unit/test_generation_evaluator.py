from unittest.mock import AsyncMock, Mock

import pytest

from app.evaluation.generation_evaluator import (
    GenerationEvaluator,
)


def create_evaluator() -> tuple[
    GenerationEvaluator,
    Mock,
    Mock,
]:
    faithfulness = Mock()
    faithfulness.ascore = AsyncMock(
        return_value=Mock(value=0.8)
    )

    answer_relevancy = Mock()
    answer_relevancy.ascore = AsyncMock(
        return_value=Mock(value=0.7)
    )

    evaluator = GenerationEvaluator(
        faithfulness=faithfulness,
        answer_relevancy=answer_relevancy,
    )

    return (
        evaluator,
        faithfulness,
        answer_relevancy,
    )


@pytest.mark.asyncio
async def test_evaluate_returns_ragas_scores() -> None:
    (
        evaluator,
        faithfulness,
        answer_relevancy,
    ) = create_evaluator()

    result = await evaluator.evaluate(
        question="Example question?",
        answer="Example answer.",
        contexts=[
            "First context.",
            "Second context.",
        ],
    )

    faithfulness.ascore.assert_awaited_once_with(
        user_input="Example question?",
        response="Example answer.",
        retrieved_contexts=[
            "First context.",
            "Second context.",
        ],
    )
    answer_relevancy.ascore.assert_awaited_once_with(
        user_input="Example question?",
        response="Example answer.",
    )

    assert result.faithfulness == 0.8
    assert result.answer_relevancy == 0.7


@pytest.mark.asyncio
async def test_evaluate_allows_negative_answer_relevancy() -> None:
    evaluator, _, answer_relevancy = create_evaluator()
    answer_relevancy.ascore.return_value = Mock(
        value=-0.2
    )

    result = await evaluator.evaluate(
        question="Example question?",
        answer="Example answer.",
        contexts=["Example context."],
    )

    assert result.answer_relevancy == -0.2


@pytest.mark.asyncio
async def test_evaluate_propagates_ragas_error() -> None:
    evaluator, faithfulness, answer_relevancy = (
        create_evaluator()
    )
    faithfulness.ascore.side_effect = RuntimeError(
        "RAGAS unavailable"
    )

    with pytest.raises(
        RuntimeError,
        match="RAGAS unavailable",
    ):
        await evaluator.evaluate(
            question="Example question?",
            answer="Example answer.",
            contexts=["Example context."],
        )

    answer_relevancy.ascore.assert_not_awaited()


@pytest.mark.asyncio
async def test_evaluate_rejects_ragas_score_outside_range() -> None:
    evaluator, faithfulness, _ = create_evaluator()
    faithfulness.ascore.return_value = Mock(value=1.1)

    with pytest.raises(
        ValueError,
        match="faithfulness must be between 0.0 and 1.0",
    ):
        await evaluator.evaluate(
            question="Example question?",
            answer="Example answer.",
            contexts=["Example context."],
        )


@pytest.mark.asyncio
async def test_evaluate_rejects_empty_question() -> None:
    evaluator, faithfulness, answer_relevancy = (
        create_evaluator()
    )

    with pytest.raises(
        ValueError,
        match="question must not be empty",
    ):
        await evaluator.evaluate(
            question=" ",
            answer="Example answer.",
            contexts=["Example context."],
        )

    faithfulness.ascore.assert_not_awaited()
    answer_relevancy.ascore.assert_not_awaited()


@pytest.mark.asyncio
async def test_evaluate_rejects_empty_answer() -> None:
    evaluator, faithfulness, answer_relevancy = (
        create_evaluator()
    )

    with pytest.raises(
        ValueError,
        match="answer must not be empty",
    ):
        await evaluator.evaluate(
            question="Example question?",
            answer=" ",
            contexts=["Example context."],
        )

    faithfulness.ascore.assert_not_awaited()
    answer_relevancy.ascore.assert_not_awaited()


@pytest.mark.asyncio
async def test_evaluate_rejects_empty_contexts() -> None:
    evaluator, faithfulness, answer_relevancy = (
        create_evaluator()
    )

    with pytest.raises(
        ValueError,
        match="contexts must not be empty",
    ):
        await evaluator.evaluate(
            question="Example question?",
            answer="Example answer.",
            contexts=[],
        )

    faithfulness.ascore.assert_not_awaited()
    answer_relevancy.ascore.assert_not_awaited()


@pytest.mark.asyncio
async def test_evaluate_rejects_empty_context_text() -> None:
    evaluator, faithfulness, answer_relevancy = (
        create_evaluator()
    )

    with pytest.raises(
        ValueError,
        match="contexts must not contain empty text",
    ):
        await evaluator.evaluate(
            question="Example question?",
            answer="Example answer.",
            contexts=["Valid context.", " "],
        )

    faithfulness.ascore.assert_not_awaited()
    answer_relevancy.ascore.assert_not_awaited()
