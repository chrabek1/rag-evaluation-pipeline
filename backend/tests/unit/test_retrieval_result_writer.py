import json
from pathlib import Path

import pytest

from app.evaluation.retrieval_result_writer import (
    RetrievalResultWriter,
)
from app.models.chunk import Chunk
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


def build_metrics() -> RetrievalMetricsResult:
    return RetrievalMetricsResult(
        k=2,
        precision_at_k=0.5,
        recall_at_k=1.0,
        hit_rate_at_k=1.0,
        reciprocal_rank_at_k=1.0,
        ndcg_at_k=0.8,
        graded_ndcg_at_k=0.9,
        weighted_precision_at_k=0.75,
        evidence_coverage_at_k=1.0,
    )


def build_summary() -> RetrievalMetricsSummary:
    return RetrievalMetricsSummary(
        k=2,
        query_count=1,
        mean_precision_at_k=0.5,
        mean_recall_at_k=1.0,
        mean_hit_rate_at_k=1.0,
        mrr_at_k=1.0,
        mean_ndcg_at_k=0.8,
        mean_graded_ndcg_at_k=0.9,
        mean_weighted_precision_at_k=0.75,
        mean_evidence_coverage_at_k=1.0,
    )


def build_result() -> RetrievalEvaluationRunResult:
    retrieved_chunk = RetrievedChunk(
        chunk_id="document.pdf_0001",
        chunk=Chunk(
            filename="document.pdf",
            content="Retrieved content.",
        ),
        score=0.91,
    )

    query_result = RetrievalQueryEvaluationResult(
        query_id="query-1",
        question="Example question?",
        retrieved_chunks=(retrieved_chunk,),
        metrics=build_metrics(),
    )

    return RetrievalEvaluationRunResult(
        query_results=(query_result,),
        summary=build_summary(),
    )


def test_write_creates_structured_json(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "results" / "evaluation.json"

    RetrievalResultWriter().write(
        result=build_result(),
        output_path=output_path,
        embedding_model="BAAI/bge-m3",
        embedding_dimension=1024,
    )

    payload = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert payload["configuration"] == {
        "top_k": 2,
        "embedding_model": "BAAI/bge-m3",
        "embedding_dimension": 1024,
    }

    assert payload["summary"] == {
        "k": 2,
        "query_count": 1,
        "mean_precision_at_k": 0.5,
        "mean_recall_at_k": 1.0,
        "mean_hit_rate_at_k": 1.0,
        "mrr_at_k": 1.0,
        "mean_ndcg_at_k": 0.8,
        "mean_graded_ndcg_at_k": 0.9,
        "mean_weighted_precision_at_k": 0.75,
        "mean_evidence_coverage_at_k": 1.0,
    }

    query = payload["queries"][0]
    assert query["query_id"] == "query-1"
    assert query["question"] == "Example question?"
    assert query["retrieved_chunks"] == [
        {
            "rank": 1,
            "chunk_id": "document.pdf_0001",
            "filename": "document.pdf",
            "score": 0.91,
        }
    ]
    assert query["metrics"] == {
        "k": 2,
        "precision_at_k": 0.5,
        "recall_at_k": 1.0,
        "hit_rate_at_k": 1.0,
        "reciprocal_rank_at_k": 1.0,
        "ndcg_at_k": 0.8,
        "graded_ndcg_at_k": 0.9,
        "weighted_precision_at_k": 0.75,
        "evidence_coverage_at_k": 1.0,
    }


def test_write_rejects_empty_embedding_model(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="embedding_model must not be empty",
    ):
        RetrievalResultWriter().write(
            result=build_result(),
            output_path=tmp_path / "result.json",
            embedding_model=" ",
            embedding_dimension=1024,
        )


@pytest.mark.parametrize("embedding_dimension", [0, -1])
def test_write_rejects_invalid_embedding_dimension(
    tmp_path: Path,
    embedding_dimension: int,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "embedding_dimension must be greater than 0"
        ),
    ):
        RetrievalResultWriter().write(
            result=build_result(),
            output_path=tmp_path / "result.json",
            embedding_model="BAAI/bge-m3",
            embedding_dimension=embedding_dimension,
        )