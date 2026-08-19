import pytest

from app.models.chunk import Chunk
from app.models.retrieval_metrics_result import (
    RetrievalMetricsResult,
)
from app.models.retrieval_evaluation_result import (
    RetrievalQueryEvaluationResult,
)
from app.models.retrieved_chunk import RetrievedChunk


def build_metrics() -> RetrievalMetricsResult:
    return RetrievalMetricsResult(
        k=2,
        precision_at_k=0.5,
        recall_at_k=1.0,
        hit_rate_at_k=1.0,
        reciprocal_rank_at_k=1.0,
        ndcg_at_k=1.0,
        graded_ndcg_at_k=1.0,
        weighted_precision_at_k=1.0,
        evidence_coverage_at_k=1.0,
    )


def build_retrieved_chunk(
    chunk_id: str,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        chunk=Chunk(
            filename="document.pdf",
            content="Retrieved content.",
        ),
        score=0.9,
    )


def test_retrieval_query_evaluation_result_stores_data() -> None:
    retrieved_chunks = (
        build_retrieved_chunk("chunk-1"),
        build_retrieved_chunk("chunk-2"),
    )
    metrics = build_metrics()

    result = RetrievalQueryEvaluationResult(
        query_id="query-1",
        question="Example question?",
        retrieved_chunks=retrieved_chunks,
        metrics=metrics,
    )

    assert result.query_id == "query-1"
    assert result.question == "Example question?"
    assert result.retrieved_chunks == retrieved_chunks
    assert result.metrics == metrics


@pytest.mark.parametrize("query_id", ["", " ", "\t", "\n"])
def test_retrieval_query_evaluation_result_rejects_empty_query_id(
    query_id: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="query_id must not be empty",
    ):
        RetrievalQueryEvaluationResult(
            query_id=query_id,
            question="Example question?",
            retrieved_chunks=(),
            metrics=build_metrics(),
        )


def test_retrieval_query_evaluation_result_rejects_empty_question() -> None:
    with pytest.raises(
        ValueError,
        match="question must not be empty",
    ):
        RetrievalQueryEvaluationResult(
            query_id="query-1",
            question=" ",
            retrieved_chunks=(),
            metrics=build_metrics(),
        )


def test_retrieval_query_evaluation_result_rejects_duplicate_chunks() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "retrieved_chunks must have unique chunk_id values"
        ),
    ):
        RetrievalQueryEvaluationResult(
            query_id="query-1",
            question="Example question?",
            retrieved_chunks=(
                build_retrieved_chunk("chunk-1"),
                build_retrieved_chunk("chunk-1"),
            ),
            metrics=build_metrics(),
        )
