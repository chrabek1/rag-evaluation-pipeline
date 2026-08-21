import pytest

from app.evaluation.generation_metrics_aggregator import (
    GenerationMetricsAggregator,
)
from app.models.chunk import Chunk
from app.models.generation_metrics_result import (
    GenerationMetricsResult,
)
from app.models.generation_result import GenerationResult
from app.models.llm_response import LLMResponse
from app.models.rag_evaluation_result import (
    RAGQueryEvaluationResult,
)
from app.models.retrieval_evaluation_result import (
    RetrievalQueryEvaluationResult,
)
from app.models.retrieval_metrics_result import (
    RetrievalMetricsResult,
)
from app.models.retrieved_chunk import RetrievedChunk


def create_result(
    *,
    query_id: str,
    faithfulness: float,
    answer_relevancy: float,
    latency_seconds: float,
    input_tokens: int | None,
    output_tokens: int | None,
) -> RAGQueryEvaluationResult:
    chunk = RetrievedChunk(
        chunk_id=f"{query_id}-chunk",
        chunk=Chunk(
            content="Context",
            filename="document.pdf",
        ),
        score=0.9,
    )

    retrieval = RetrievalQueryEvaluationResult(
        query_id=query_id,
        question="Example question?",
        retrieved_chunks=(chunk,),
        metrics=RetrievalMetricsResult(
            k=1,
            precision_at_k=1.0,
            recall_at_k=1.0,
            hit_rate_at_k=1.0,
            reciprocal_rank_at_k=1.0,
            ndcg_at_k=1.0,
            graded_ndcg_at_k=1.0,
            weighted_precision_at_k=1.0,
            evidence_coverage_at_k=1.0,
        ),
    )

    generation = GenerationResult(
        response=LLMResponse(
            text="Generated answer",
            model="test-model",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
        latency_seconds=latency_seconds,
    )

    return RAGQueryEvaluationResult(
        retrieval=retrieval,
        expected_answer="Expected answer",
        generation=generation,
        generation_metrics=GenerationMetricsResult(
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
        ),
    )


def test_aggregate_returns_mean_metrics_and_token_totals() -> None:
    results = [
        create_result(
            query_id="query-1",
            faithfulness=0.8,
            answer_relevancy=0.6,
            latency_seconds=2.0,
            input_tokens=100,
            output_tokens=20,
        ),
        create_result(
            query_id="query-2",
            faithfulness=1.0,
            answer_relevancy=0.8,
            latency_seconds=4.0,
            input_tokens=120,
            output_tokens=30,
        ),
    ]

    summary = GenerationMetricsAggregator().aggregate(
        results
    )

    assert summary.query_count == 2
    assert summary.mean_faithfulness == pytest.approx(0.9)
    assert summary.mean_answer_relevancy == pytest.approx(0.7)
    assert summary.mean_latency_seconds == pytest.approx(3.0)
    assert summary.total_input_tokens == 220
    assert summary.total_output_tokens == 50


@pytest.mark.parametrize(
    ("missing_field", "expected_input", "expected_output"),
    [
        ("input", None, 50),
        ("output", 220, None),
    ],
)
def test_aggregate_returns_none_for_incomplete_token_usage(
    missing_field: str,
    expected_input: int | None,
    expected_output: int | None,
) -> None:
    first_input = None if missing_field == "input" else 100
    first_output = None if missing_field == "output" else 20

    results = [
        create_result(
            query_id="query-1",
            faithfulness=0.8,
            answer_relevancy=0.6,
            latency_seconds=2.0,
            input_tokens=first_input,
            output_tokens=first_output,
        ),
        create_result(
            query_id="query-2",
            faithfulness=1.0,
            answer_relevancy=0.8,
            latency_seconds=4.0,
            input_tokens=120,
            output_tokens=30,
        ),
    ]

    summary = GenerationMetricsAggregator().aggregate(
        results
    )

    assert summary.total_input_tokens == expected_input
    assert summary.total_output_tokens == expected_output


def test_aggregate_rejects_empty_results() -> None:
    with pytest.raises(
        ValueError,
        match="results must not be empty",
    ):
        GenerationMetricsAggregator().aggregate([])