from math import log2

import pytest

from app.evaluation.retrieval_evaluator import (
    RetrievalEvaluator,
)
from app.models.retrieval_metrics_result import (
    RetrievalMetricsResult,
)


def test_evaluate_returns_all_retrieval_metrics() -> None:
    evaluator = RetrievalEvaluator()

    result = evaluator.evaluate(
        relevance_by_chunk_id={
            "chunk-2": 0.8,
            "chunk-4": 0.4,
        },
        retrieved_chunk_ids=[
            "chunk-1",
            "chunk-2",
            "chunk-3",
            "chunk-4",
        ],
        k=3,
    )

    expected_binary_ndcg = (
        1.0 / log2(3)
    ) / (
        1.0 + 1.0 / log2(3)
    )

    expected_graded_ndcg = (
        0.8 / log2(3)
    ) / (
        0.8 + 0.4 / log2(3)
    )

    assert isinstance(result, RetrievalMetricsResult)
    assert result.k == 3
    assert result.precision_at_k == pytest.approx(1 / 3)
    assert result.recall_at_k == pytest.approx(1 / 2)
    assert result.hit_rate_at_k == 1.0
    assert result.reciprocal_rank_at_k == pytest.approx(1 / 2)
    assert result.ndcg_at_k == pytest.approx(
        expected_binary_ndcg
    )
    assert result.graded_ndcg_at_k == pytest.approx(
        expected_graded_ndcg
    )
    assert result.weighted_precision_at_k == pytest.approx(
        0.8 / 3
    )


def test_evaluate_rejects_duplicate_retrieved_chunks() -> None:
    evaluator = RetrievalEvaluator()

    with pytest.raises(
        ValueError,
        match="retrieved_chunk_ids must not contain duplicates",
    ):
        evaluator.evaluate(
            relevance_by_chunk_id={"chunk-1": 1.0},
            retrieved_chunk_ids=[
                "chunk-1",
                "chunk-1",
            ],
            k=2,
        )