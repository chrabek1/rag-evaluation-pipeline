import pytest

from app.models.generation_metrics_result import (
    GenerationMetricsResult,
)


def test_generation_metrics_result_stores_scores() -> None:
    result = GenerationMetricsResult(
        faithfulness=0.8,
        answer_relevancy=0.7,
    )

    assert result.faithfulness == 0.8
    assert result.answer_relevancy == 0.7


def test_generation_metrics_result_allows_negative_answer_relevancy() -> None:
    result = GenerationMetricsResult(
        faithfulness=0.8,
        answer_relevancy=-0.2,
    )

    assert result.answer_relevancy == -0.2


@pytest.mark.parametrize(
    (
        "name",
        "faithfulness",
        "answer_relevancy",
        "minimum",
        "maximum",
    ),
    [
        ("faithfulness", -0.1, 0.5, 0.0, 1.0),
        ("faithfulness", 1.1, 0.5, 0.0, 1.0),
        (
            "answer_relevancy",
            0.5,
            -1.1,
            -1.0,
            1.0,
        ),
        (
            "answer_relevancy",
            0.5,
            1.1,
            -1.0,
            1.0,
        ),
    ],
)
def test_generation_metrics_result_rejects_score_outside_range(
    name: str,
    faithfulness: float,
    answer_relevancy: float,
    minimum: float,
    maximum: float,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            f"{name} must be between "
            f"{minimum} and {maximum}"
        ),
    ):
        GenerationMetricsResult(
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
        )


@pytest.mark.parametrize(
    ("name", "faithfulness", "answer_relevancy"),
    [
        ("faithfulness", float("nan"), 0.5),
        ("faithfulness", float("inf"), 0.5),
        ("answer_relevancy", 0.5, float("nan")),
        ("answer_relevancy", 0.5, float("inf")),
    ],
)
def test_generation_metrics_result_rejects_non_finite_score(
    name: str,
    faithfulness: float,
    answer_relevancy: float,
) -> None:
    with pytest.raises(
        ValueError,
        match=f"{name} must be finite",
    ):
        GenerationMetricsResult(
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
        )
