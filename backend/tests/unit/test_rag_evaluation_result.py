import pytest

from app.models.chunk import Chunk
from app.models.generation_metrics_result import (
    GenerationMetricsResult,
)
from app.models.generation_metrics_summary import (
    GenerationMetricsSummary,
)
from app.models.generation_result import GenerationResult
from app.models.llm_response import LLMResponse
from app.models.rag_evaluation_result import (
    RAGEvaluationRunResult,
    RAGQueryEvaluationResult,
)
from app.models.retrieval_evaluation_result import (
    RetrievalQueryEvaluationResult,
)
from app.models.retrieval_metrics_result import (
    RetrievalMetricsResult,
)
from app.models.retrieval_metrics_summary import (
    RetrievalMetricsSummary,
)
from app.models.retrieved_chunk import RetrievedChunk


def create_retrieval_metrics(
    k: int = 1,
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


def create_retrieval_summary(
    query_count: int = 1,
    k: int = 1,
) -> RetrievalMetricsSummary:
    return RetrievalMetricsSummary(
        query_count=query_count,
        k=k,
        mean_precision_at_k=1.0,
        mean_recall_at_k=1.0,
        mean_hit_rate_at_k=1.0,
        mrr_at_k=1.0,
        mean_ndcg_at_k=1.0,
        mean_graded_ndcg_at_k=1.0,
        mean_weighted_precision_at_k=1.0,
        mean_evidence_coverage_at_k=1.0,
    )


def create_generation_summary(
    query_count: int = 1,
) -> GenerationMetricsSummary:
    return GenerationMetricsSummary(
        query_count=query_count,
        mean_faithfulness=1.0,
        mean_answer_relevancy=1.0,
        mean_latency_seconds=1.0,
        total_input_tokens=10,
        total_output_tokens=5,
    )


def create_query_result(
    query_id: str = "query-1",
    k: int = 1,
) -> RAGQueryEvaluationResult:
    retrieval = RetrievalQueryEvaluationResult(
        query_id=query_id,
        question="Example question?",
        retrieved_chunks=(
            RetrievedChunk(
                chunk_id="document.pdf_0001",
                chunk=Chunk(
                    filename="document.pdf",
                    content="Example context.",
                ),
                score=0.9,
            ),
        ),
        metrics=create_retrieval_metrics(k=k),
    )
    generation = GenerationResult(
        response=LLMResponse(
            text="Example answer.",
            model="test-model",
            input_tokens=10,
            output_tokens=5,
        ),
        latency_seconds=1.0,
    )

    return RAGQueryEvaluationResult(
        retrieval=retrieval,
        expected_answer="Expected answer.",
        generation=generation,
        generation_metrics=GenerationMetricsResult(
            faithfulness=1.0,
            answer_relevancy=1.0,
        ),
    )


def test_rag_query_evaluation_result_stores_data() -> None:
    result = create_query_result()

    assert result.query_id == "query-1"
    assert result.question == "Example question?"
    assert result.expected_answer == "Expected answer."
    assert result.generation.answer == "Example answer."
    assert result.generation_metrics.faithfulness == 1.0


def test_rag_query_evaluation_result_rejects_empty_expected_answer() -> None:
    result = create_query_result()

    with pytest.raises(
        ValueError,
        match="expected_answer must not be empty",
    ):
        RAGQueryEvaluationResult(
            retrieval=result.retrieval,
            expected_answer=" ",
            generation=result.generation,
            generation_metrics=result.generation_metrics,
        )


def test_rag_evaluation_run_result_stores_data() -> None:
    query_result = create_query_result()

    result = RAGEvaluationRunResult(
        query_results=(query_result,),
        retrieval_summary=create_retrieval_summary(),
        generation_summary=create_generation_summary(),
    )

    assert result.query_results == (query_result,)
    assert result.retrieval_summary.query_count == 1
    assert result.generation_summary.query_count == 1


def test_rag_evaluation_run_result_rejects_empty_results() -> None:
    with pytest.raises(
        ValueError,
        match="query_results must not be empty",
    ):
        RAGEvaluationRunResult(
            query_results=(),
            retrieval_summary=create_retrieval_summary(),
            generation_summary=create_generation_summary(),
        )


def test_rag_evaluation_run_result_rejects_duplicate_queries() -> None:
    first = create_query_result(query_id="query-1")
    second = create_query_result(query_id="query-1")

    with pytest.raises(
        ValueError,
        match="query_results must have unique query IDs",
    ):
        RAGEvaluationRunResult(
            query_results=(first, second),
            retrieval_summary=create_retrieval_summary(
                query_count=2
            ),
            generation_summary=create_generation_summary(
                query_count=2
            ),
        )


@pytest.mark.parametrize(
    ("retrieval_count", "generation_count", "message"),
    [
        (
            2,
            1,
            "retrieval summary query count must match",
        ),
        (
            1,
            2,
            "generation summary query count must match",
        ),
    ],
)
def test_rag_evaluation_run_result_rejects_wrong_query_count(
    retrieval_count: int,
    generation_count: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RAGEvaluationRunResult(
            query_results=(create_query_result(),),
            retrieval_summary=create_retrieval_summary(
                query_count=retrieval_count
            ),
            generation_summary=create_generation_summary(
                query_count=generation_count
            ),
        )


def test_rag_evaluation_run_result_rejects_different_k() -> None:
    with pytest.raises(
        ValueError,
        match="all retrieval results must use summary k",
    ):
        RAGEvaluationRunResult(
            query_results=(
                create_query_result(k=2),
            ),
            retrieval_summary=create_retrieval_summary(k=1),
            generation_summary=create_generation_summary(),
        )