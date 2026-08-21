import json
from pathlib import Path

import pytest

from app.evaluation.rag_result_writer import RAGResultWriter
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


def build_result() -> RAGEvaluationRunResult:
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

    retrieval = RetrievalQueryEvaluationResult(
        query_id="query-1",
        question="What is RAG?",
        retrieved_chunks=(
            RetrievedChunk(
                chunk_id="document.pdf_0001",
                chunk=Chunk(
                    filename="document.pdf",
                    content="RAG uses retrieved context.",
                ),
                score=0.91,
            ),
        ),
        metrics=metrics,
    )

    query_result = RAGQueryEvaluationResult(
        retrieval=retrieval,
        expected_answer="RAG uses retrieved information.",
        generation=GenerationResult(
            response=LLMResponse(
                text="RAG uses retrieved context.",
                model="llama3.2:3b",
                input_tokens=20,
                output_tokens=8,
            ),
            latency_seconds=2.5,
        ),
        generation_metrics=GenerationMetricsResult(
            faithfulness=0.9,
            answer_relevancy=0.8,
        ),
    )

    return RAGEvaluationRunResult(
        query_results=(query_result,),
        retrieval_summary=RetrievalMetricsSummary(
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
        generation_summary=GenerationMetricsSummary(
            query_count=1,
            mean_faithfulness=0.9,
            mean_answer_relevancy=0.8,
            mean_latency_seconds=2.5,
            total_input_tokens=20,
            total_output_tokens=8,
        ),
    )


def test_write_creates_structured_json(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "results" / "rag.json"

    RAGResultWriter().write(
        result=build_result(),
        output_path=output_path,
        embedding_model="BAAI/bge-m3",
        embedding_dimension=1024,
        generation_provider="ollama",
        generation_model="llama3.2:3b",
        evaluation_provider="gemini",
        evaluation_model="gemini-2.5-flash",
    )

    payload = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert payload["configuration"] == {
        "top_k": 1,
        "embedding_model": "BAAI/bge-m3",
        "embedding_dimension": 1024,
        "generation_provider": "ollama",
        "generation_model": "llama3.2:3b",
        "evaluation_provider": "gemini",
        "evaluation_model": "gemini-2.5-flash",
    }

    assert payload["summary"]["retrieval"]["query_count"] == 1
    assert payload["summary"]["generation"] == {
        "query_count": 1,
        "mean_faithfulness": 0.9,
        "mean_answer_relevancy": 0.8,
        "mean_latency_seconds": 2.5,
        "total_input_tokens": 20,
        "total_output_tokens": 8,
    }

    query = payload["queries"][0]

    assert query["query_id"] == "query-1"
    assert query["expected_answer"] == (
        "RAG uses retrieved information."
    )
    assert query["retrieval"]["retrieved_chunks"][0] == {
        "rank": 1,
        "chunk_id": "document.pdf_0001",
        "filename": "document.pdf",
        "content": "RAG uses retrieved context.",
        "score": 0.91,
    }
    assert query["generation"] == {
        "answer": "RAG uses retrieved context.",
        "model": "llama3.2:3b",
        "input_tokens": 20,
        "output_tokens": 8,
        "latency_seconds": 2.5,
        "metrics": {
            "faithfulness": 0.9,
            "answer_relevancy": 0.8,
        },
    }


@pytest.mark.parametrize(
    "field_name",
    [
        "embedding_model",
        "generation_provider",
        "generation_model",
        "evaluation_provider",
        "evaluation_model",
    ],
)
def test_write_rejects_empty_configuration_value(
    tmp_path: Path,
    field_name: str,
) -> None:
    arguments = {
        "embedding_model": "BAAI/bge-m3",
        "generation_provider": "ollama",
        "generation_model": "llama3.2:3b",
        "evaluation_provider": "gemini",
        "evaluation_model": "gemini-2.5-flash",
    }
    arguments[field_name] = " "

    with pytest.raises(
        ValueError,
        match=f"{field_name} must not be empty",
    ):
        RAGResultWriter().write(
            result=build_result(),
            output_path=tmp_path / "result.json",
            embedding_dimension=1024,
            **arguments,
        )


@pytest.mark.parametrize("dimension", [0, -1])
def test_write_rejects_invalid_embedding_dimension(
    tmp_path: Path,
    dimension: int,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "embedding_dimension must be greater than 0"
        ),
    ):
        RAGResultWriter().write(
            result=build_result(),
            output_path=tmp_path / "result.json",
            embedding_model="BAAI/bge-m3",
            embedding_dimension=dimension,
            generation_provider="ollama",
            generation_model="llama3.2:3b",
            evaluation_provider="gemini",
            evaluation_model="gemini-2.5-flash",
        )