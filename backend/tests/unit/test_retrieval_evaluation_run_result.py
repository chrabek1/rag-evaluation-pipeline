import pytest

from app.models.retrieval_evaluation_result import (
    RetrievalEvaluationRunResult,
    RetrievalQueryEvaluationResult,
)
from app.models.retrieval_metrics_result import (
    RetrievalMetricsResult,
)
from app.models.retrieval_metrics_summary import (
    RetrievalMetricsSummary,
)


def build_metrics(
    k: int = 2,
) -> RetrievalMetricsResult:
    return RetrievalMetricsResult(
        k=k,
        precision_at_k=1.0,
        recall_at_k=1.0,
        hit_rate_at_k=1.0,
        reciprocal_rank_at_k=1.0,
        ndcg_at_k=1.0,
        graded_ndcg_at_k=1.0,
        weighted_precision_at_k=1.0,
        evidence_coverage_at_k=1.0,
    )


def build_query_result(
    query_id: str = "query-1",
    k: int = 2,
) -> RetrievalQueryEvaluationResult:
    return RetrievalQueryEvaluationResult(
        query_id=query_id,
        question="Example question?",
        retrieved_chunks=(),
        metrics=build_metrics(k=k),
    )


def build_summary(
    k: int = 2,
    query_count: int = 1,
) -> RetrievalMetricsSummary:
    return RetrievalMetricsSummary(
        k=k,
        query_count=query_count,
        mean_precision_at_k=1.0,
        mean_recall_at_k=1.0,
        mean_hit_rate_at_k=1.0,
        mrr_at_k=1.0,
        mean_ndcg_at_k=1.0,
        mean_graded_ndcg_at_k=1.0,
        mean_weighted_precision_at_k=1.0,
        mean_evidence_coverage_at_k=1.0,
    )


def test_retrieval_evaluation_run_result_stores_data() -> None:
    query_results = (build_query_result(),)
    summary = build_summary()

    result = RetrievalEvaluationRunResult(
        query_results=query_results,
        summary=summary,
    )

    assert result.query_results == query_results
    assert result.summary == summary


def test_retrieval_evaluation_run_result_rejects_empty_results() -> None:
    with pytest.raises(
        ValueError,
        match="query_results must not be empty",
    ):
        RetrievalEvaluationRunResult(
            query_results=(),
            summary=build_summary(query_count=0),
        )


def test_retrieval_evaluation_run_result_rejects_duplicate_queries() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "query_results must have unique query_id values"
        ),
    ):
        RetrievalEvaluationRunResult(
            query_results=(
                build_query_result(query_id="query-1"),
                build_query_result(query_id="query-1"),
            ),
            summary=build_summary(query_count=2),
        )


def test_retrieval_evaluation_run_result_rejects_wrong_query_count() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "summary query_count must match query_results"
        ),
    ):
        RetrievalEvaluationRunResult(
            query_results=(build_query_result(),),
            summary=build_summary(query_count=2),
        )


def test_retrieval_evaluation_run_result_rejects_different_k() -> None:
    with pytest.raises(
        ValueError,
        match="all query results must use summary k",
    ):
        RetrievalEvaluationRunResult(
            query_results=(
                build_query_result(k=3),
            ),
            summary=build_summary(k=2),
        )
