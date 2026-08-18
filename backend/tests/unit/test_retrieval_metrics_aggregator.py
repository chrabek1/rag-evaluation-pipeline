import pytest

from app.evaluation.retrieval_metrics_aggregator import (
    RetrievalMetricsAggregator,
)
from app.models.retrieval_metrics_result import (
    RetrievalMetricsResult,
)
from app.models.retrieval_metrics_summary import (
    RetrievalMetricsSummary,
)


def test_aggregate_returns_mean_metrics() -> None:
    aggregator = RetrievalMetricsAggregator()

    results = [
        RetrievalMetricsResult(
            k=5,
            precision_at_k=0.2,
            recall_at_k=0.4,
            hit_rate_at_k=1.0,
            reciprocal_rank_at_k=0.5,
            ndcg_at_k=0.6,
            graded_ndcg_at_k=0.7,
            weighted_precision_at_k=0.3,
        ),
        RetrievalMetricsResult(
            k=5,
            precision_at_k=0.4,
            recall_at_k=0.8,
            hit_rate_at_k=0.0,
            reciprocal_rank_at_k=0.25,
            ndcg_at_k=0.2,
            graded_ndcg_at_k=0.5,
            weighted_precision_at_k=0.1,
        ),
    ]

    summary = aggregator.aggregate(results)

    assert isinstance(summary, RetrievalMetricsSummary)
    assert summary.k == 5
    assert summary.query_count == 2
    assert summary.mean_precision_at_k == pytest.approx(0.3)
    assert summary.mean_recall_at_k == pytest.approx(0.6)
    assert summary.mean_hit_rate_at_k == pytest.approx(0.5)
    assert summary.mrr_at_k == pytest.approx(0.375)
    assert summary.mean_ndcg_at_k == pytest.approx(0.4)
    assert summary.mean_graded_ndcg_at_k == pytest.approx(0.6)
    assert summary.mean_weighted_precision_at_k == pytest.approx(
        0.2
    )


def test_aggregate_rejects_empty_results() -> None:
    aggregator = RetrievalMetricsAggregator()

    with pytest.raises(
        ValueError,
        match="results must not be empty",
    ):
        aggregator.aggregate([])


def test_aggregate_rejects_results_with_different_k() -> None:
    aggregator = RetrievalMetricsAggregator()

    first_result = RetrievalMetricsResult(
        k=3,
        precision_at_k=1.0,
        recall_at_k=1.0,
        hit_rate_at_k=1.0,
        reciprocal_rank_at_k=1.0,
        ndcg_at_k=1.0,
        graded_ndcg_at_k=1.0,
        weighted_precision_at_k=1.0,
    )
    second_result = RetrievalMetricsResult(
        k=5,
        precision_at_k=1.0,
        recall_at_k=1.0,
        hit_rate_at_k=1.0,
        reciprocal_rank_at_k=1.0,
        ndcg_at_k=1.0,
        graded_ndcg_at_k=1.0,
        weighted_precision_at_k=1.0,
    )

    with pytest.raises(
        ValueError,
        match="all results must use the same k",
    ):
        aggregator.aggregate(
            [first_result, second_result]
        )