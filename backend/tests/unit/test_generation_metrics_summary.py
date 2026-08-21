import math

import pytest

from app.models.generation_metrics_summary import (
    GenerationMetricsSummary,
)


def test_generation_metrics_summary_stores_data() -> None:
    summary = GenerationMetricsSummary(
        query_count=5,
        mean_faithfulness=0.8,
        mean_answer_relevancy=0.7,
        mean_latency_seconds=1.5,
        total_input_tokens=100,
        total_output_tokens=50,
    )

    assert summary.query_count == 5
    assert summary.mean_faithfulness == 0.8
    assert summary.mean_answer_relevancy == 0.7
    assert summary.mean_latency_seconds == 1.5
    assert summary.total_input_tokens == 100
    assert summary.total_output_tokens == 50


def test_generation_metrics_summary_allows_missing_tokens() -> None:
    summary = GenerationMetricsSummary(
        query_count=1,
        mean_faithfulness=1.0,
        mean_answer_relevancy=0.9,
        mean_latency_seconds=0.0,
        total_input_tokens=None,
        total_output_tokens=None,
    )

    assert summary.total_input_tokens is None
    assert summary.total_output_tokens is None


@pytest.mark.parametrize("query_count", [0, -1])
def test_generation_metrics_summary_rejects_invalid_query_count(
    query_count: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="query_count must be greater than 0",
    ):
        GenerationMetricsSummary(
            query_count=query_count,
            mean_faithfulness=0.8,
            mean_answer_relevancy=0.7,
            mean_latency_seconds=1.0,
            total_input_tokens=10,
            total_output_tokens=5,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mean_faithfulness", -0.1),
        ("mean_faithfulness", 1.1),
        ("mean_answer_relevancy", -1.1),
        ("mean_answer_relevancy", 1.1),
    ],
)
def test_generation_metrics_summary_rejects_invalid_score(
    field: str,
    value: float,
) -> None:
    values = {
        "query_count": 1,
        "mean_faithfulness": 0.8,
        "mean_answer_relevancy": 0.7,
        "mean_latency_seconds": 1.0,
        "total_input_tokens": 10,
        "total_output_tokens": 5,
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        GenerationMetricsSummary(**values)


@pytest.mark.parametrize(
    "latency",
    [-0.1, math.nan, math.inf],
)
def test_generation_metrics_summary_rejects_invalid_latency(
    latency: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="mean_latency_seconds",
    ):
        GenerationMetricsSummary(
            query_count=1,
            mean_faithfulness=0.8,
            mean_answer_relevancy=0.7,
            mean_latency_seconds=latency,
            total_input_tokens=10,
            total_output_tokens=5,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("total_input_tokens", -1),
        ("total_output_tokens", -1),
    ],
)
def test_generation_metrics_summary_rejects_negative_tokens(
    field: str,
    value: int,
) -> None:
    values = {
        "query_count": 1,
        "mean_faithfulness": 0.8,
        "mean_answer_relevancy": 0.7,
        "mean_latency_seconds": 1.0,
        "total_input_tokens": 10,
        "total_output_tokens": 5,
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        GenerationMetricsSummary(**values)