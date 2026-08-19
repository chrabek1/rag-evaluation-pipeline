from unittest.mock import AsyncMock

import pytest

from app.evaluation.retrieval_evaluation_pipeline import (
    RetrievalEvaluationPipeline,
)
from app.evaluation.retrieval_evaluator import RetrievalEvaluator
from app.evaluation.retrieval_metrics_aggregator import (
    RetrievalMetricsAggregator,
)
from app.models.chunk import Chunk
from app.models.golden_dataset import (
    ChunkEvidenceIntervals,
    EvidenceInterval,
    GoldenDataset,
    GoldenDatasetMetadata,
    GoldenDatasetRecord,
    GoldenEvidence,
    GoldenRelevantChunk,
)
from app.models.retrieved_chunk import RetrievedChunk
from app.services.retrieval_service import RetrievalService


def build_relevant_chunk(
    chunk_id: str,
    evidence_coverage: float,
    start: int,
    end: int,
) -> GoldenRelevantChunk:
    return GoldenRelevantChunk(
        chunk_id=chunk_id,
        evidence_coverage=evidence_coverage,
        evidence_intervals=(
            ChunkEvidenceIntervals(
                evidence_index=0,
                intervals=(
                    EvidenceInterval(
                        start=start,
                        end=end,
                    ),
                ),
            ),
        ),
    )


def build_dataset() -> GoldenDataset:
    return GoldenDataset(
        metadata=GoldenDatasetMetadata(
            schema_version=1,
            evidence_interval_gap_tolerance=3,
        ),
        records=(
            GoldenDatasetRecord(
                query_id="query-1",
                question="Example question?",
                expected_answer="Example answer.",
                evidence=(
                    GoldenEvidence(
                        text="Relevant evidence fragment.",
                        normalized_length=100,
                    ),
                ),
                relevant_chunks=(
                    build_relevant_chunk(
                        chunk_id="chunk-1",
                        evidence_coverage=0.8,
                        start=0,
                        end=80,
                    ),
                    build_relevant_chunk(
                        chunk_id="chunk-2",
                        evidence_coverage=0.2,
                        start=80,
                        end=100,
                    ),
                ),
            ),
        ),
    )


def build_retrieved_chunk(
    chunk_id: str,
    score: float,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        chunk=Chunk(
            filename="document.pdf",
            content=f"Content of {chunk_id}.",
        ),
        score=score,
    )


@pytest.mark.asyncio
async def test_evaluate_returns_query_results_and_summary() -> None:
    retrieval_service = AsyncMock(spec=RetrievalService)
    retrieval_service.retrieve.return_value = [
        build_retrieved_chunk("chunk-1", 0.9),
        build_retrieved_chunk("chunk-2", 0.8),
    ]

    pipeline = RetrievalEvaluationPipeline(
        retrieval_service=retrieval_service,
        retrieval_evaluator=RetrievalEvaluator(),
        metrics_aggregator=RetrievalMetricsAggregator(),
    )

    result = await pipeline.evaluate(
        dataset=build_dataset(),
        top_k=2,
    )

    retrieval_service.retrieve.assert_awaited_once_with(
        query="Example question?",
        top_k=2,
    )

    assert len(result.query_results) == 1

    query_result = result.query_results[0]
    assert query_result.query_id == "query-1"
    assert query_result.question == "Example question?"
    assert [
        chunk.chunk_id
        for chunk in query_result.retrieved_chunks
    ] == ["chunk-1", "chunk-2"]

    assert query_result.metrics.precision_at_k == 1.0
    assert query_result.metrics.recall_at_k == 1.0
    assert query_result.metrics.hit_rate_at_k == 1.0
    assert query_result.metrics.reciprocal_rank_at_k == 1.0
    assert query_result.metrics.ndcg_at_k == 1.0
    assert query_result.metrics.graded_ndcg_at_k == 1.0
    assert query_result.metrics.weighted_precision_at_k == 1.0
    assert query_result.metrics.evidence_coverage_at_k == 1.0

    assert result.summary.k == 2
    assert result.summary.query_count == 1
    assert result.summary.mean_precision_at_k == 1.0
    assert result.summary.mean_recall_at_k == 1.0
    assert result.summary.mean_hit_rate_at_k == 1.0
    assert result.summary.mrr_at_k == 1.0
    assert result.summary.mean_ndcg_at_k == 1.0
    assert result.summary.mean_graded_ndcg_at_k == 1.0
    assert result.summary.mean_weighted_precision_at_k == 1.0
    assert result.summary.mean_evidence_coverage_at_k == 1.0


@pytest.mark.asyncio
async def test_evaluate_rejects_non_positive_top_k() -> None:
    retrieval_service = AsyncMock(spec=RetrievalService)

    pipeline = RetrievalEvaluationPipeline(
        retrieval_service=retrieval_service,
        retrieval_evaluator=RetrievalEvaluator(),
        metrics_aggregator=RetrievalMetricsAggregator(),
    )

    with pytest.raises(
        ValueError,
        match="top_k must be greater than 0",
    ):
        await pipeline.evaluate(
            dataset=build_dataset(),
            top_k=0,
        )

    retrieval_service.retrieve.assert_not_awaited()
