from unittest.mock import AsyncMock, Mock

import pytest

from app.evaluation.generation_metrics_aggregator import (
    GenerationMetricsAggregator,
)
from app.evaluation.rag_evaluation_pipeline import (
    RAGEvaluationPipeline,
)
from app.models.chunk import Chunk
from app.models.generation_metrics_result import (
    GenerationMetricsResult,
)
from app.models.generation_result import GenerationResult
from app.models.golden_dataset import (
    ChunkEvidenceIntervals,
    EvidenceInterval,
    GoldenDataset,
    GoldenDatasetMetadata,
    GoldenDatasetRecord,
    GoldenEvidence,
    GoldenRelevantChunk,
)
from app.models.llm_response import LLMResponse
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
from app.models.retrieved_chunk import RetrievedChunk


def create_dataset() -> GoldenDataset:
    return GoldenDataset(
        metadata=GoldenDatasetMetadata(
            schema_version=1,
            evidence_interval_gap_tolerance=3,
        ),
        records=(
            GoldenDatasetRecord(
                query_id="query-1",
                question="What is RAG?",
                expected_answer="RAG combines retrieval and generation.",
                evidence=(
                    GoldenEvidence(
                        text="RAG combines retrieval and generation.",
                        normalized_length=38,
                    ),
                ),
                relevant_chunks=(
                    GoldenRelevantChunk(
                        chunk_id="document.pdf_0001",
                        evidence_coverage=1.0,
                        evidence_intervals=(
                            ChunkEvidenceIntervals(
                                evidence_index=0,
                                intervals=(
                                    EvidenceInterval(
                                        start=0,
                                        end=38,
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


def create_retrieval_run() -> RetrievalEvaluationRunResult:
    retrieved_chunk = RetrievedChunk(
        chunk_id="document.pdf_0001",
        chunk=Chunk(
            filename="document.pdf",
            content=(
                "RAG combines retrieval and generation."
            ),
        ),
        score=0.9,
    )

    metrics = RetrievalMetricsResult(
        k=1,
        precision_at_k=1.0,
        recall_at_k=1.0,
        hit_rate_at_k=1.0,
        reciprocal_rank_at_k=1.0,
        ndcg_at_k=1.0,
        graded_ndcg_at_k=1.0,
        weighted_precision_at_k=1.0,
        evidence_coverage_at_k=1.0,
    )

    return RetrievalEvaluationRunResult(
        query_results=(
            RetrievalQueryEvaluationResult(
                query_id="query-1",
                question="What is RAG?",
                retrieved_chunks=(retrieved_chunk,),
                metrics=metrics,
            ),
        ),
        summary=RetrievalMetricsSummary(
            query_count=1,
            k=1,
            mean_precision_at_k=1.0,
            mean_recall_at_k=1.0,
            mean_hit_rate_at_k=1.0,
            mrr_at_k=1.0,
            mean_ndcg_at_k=1.0,
            mean_graded_ndcg_at_k=1.0,
            mean_weighted_precision_at_k=1.0,
            mean_evidence_coverage_at_k=1.0,
        ),
    )


@pytest.mark.asyncio
async def test_evaluate_combines_retrieval_and_generation() -> None:
    dataset = create_dataset()
    retrieval_run = create_retrieval_run()

    retrieval_pipeline = Mock()
    retrieval_pipeline.evaluate = AsyncMock(
        return_value=retrieval_run
    )

    generation = GenerationResult(
        response=LLMResponse(
            text="RAG combines retrieval and generation.",
            model="test-model",
            input_tokens=20,
            output_tokens=8,
        ),
        latency_seconds=2.0,
    )

    generation_service = Mock()
    generation_service.generate = AsyncMock(
        return_value=generation
    )

    generation_evaluator = Mock()
    generation_evaluator.evaluate = AsyncMock(
        return_value=GenerationMetricsResult(
            faithfulness=0.9,
            answer_relevancy=0.8,
        )
    )

    pipeline = RAGEvaluationPipeline(
        retrieval_pipeline=retrieval_pipeline,
        generation_service=generation_service,
        generation_evaluator=generation_evaluator,
        generation_metrics_aggregator=(
            GenerationMetricsAggregator()
        ),
    )

    result = await pipeline.evaluate(
        dataset=dataset,
        top_k=1,
    )

    assert len(result.query_results) == 1
    assert result.query_results[0].query_id == "query-1"
    assert result.query_results[0].expected_answer == (
        "RAG combines retrieval and generation."
    )
    assert result.query_results[0].generation == generation

    assert result.retrieval_summary == retrieval_run.summary
    assert result.generation_summary.query_count == 1
    assert result.generation_summary.mean_faithfulness == 0.9
    assert (
        result.generation_summary.mean_answer_relevancy
        == 0.8
    )
    assert result.generation_summary.total_input_tokens == 20
    assert result.generation_summary.total_output_tokens == 8

    generation_service.generate.assert_awaited_once_with(
        question="What is RAG?",
        retrieved_chunks=(
            retrieval_run.query_results[0].retrieved_chunks
        ),
    )

    generation_evaluator.evaluate.assert_awaited_once_with(
        question="What is RAG?",
        answer="RAG combines retrieval and generation.",
        contexts=[
            "RAG combines retrieval and generation."
        ],
    )


@pytest.mark.asyncio
async def test_evaluate_rejects_non_positive_top_k() -> None:
    pipeline = RAGEvaluationPipeline(
        retrieval_pipeline=Mock(),
        generation_service=Mock(),
        generation_evaluator=Mock(),
        generation_metrics_aggregator=Mock(),
    )

    with pytest.raises(
        ValueError,
        match="top_k must be greater than 0",
    ):
        await pipeline.evaluate(
            dataset=create_dataset(),
            top_k=0,
        )


@pytest.mark.asyncio
async def test_evaluate_rejects_unknown_retrieval_query() -> None:
    dataset = create_dataset()
    retrieval_run = create_retrieval_run()

    unknown_result = RetrievalQueryEvaluationResult(
        query_id="unknown-query",
        question=retrieval_run.query_results[0].question,
        retrieved_chunks=(
            retrieval_run.query_results[0].retrieved_chunks
        ),
        metrics=retrieval_run.query_results[0].metrics,
    )

    retrieval_pipeline = Mock()
    retrieval_pipeline.evaluate = AsyncMock(
        return_value=RetrievalEvaluationRunResult(
            query_results=(unknown_result,),
            summary=retrieval_run.summary,
        )
    )

    pipeline = RAGEvaluationPipeline(
        retrieval_pipeline=retrieval_pipeline,
        generation_service=Mock(),
        generation_evaluator=Mock(),
        generation_metrics_aggregator=Mock(),
    )

    with pytest.raises(
        ValueError,
        match=(
            "retrieval result contains an unknown query_id"
        ),
    ):
        await pipeline.evaluate(
            dataset=dataset,
            top_k=1,
        )