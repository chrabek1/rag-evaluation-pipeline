from dataclasses import dataclass

from app.models.retrieval_metrics_result import (
    RetrievalMetricsResult,
)
from app.models.retrieval_metrics_summary import (
    RetrievalMetricsSummary,
)
from app.models.retrieved_chunk import RetrievedChunk


@dataclass(frozen=True, slots=True)
class RetrievalQueryEvaluationResult:
    query_id: str
    question: str
    retrieved_chunks: tuple[RetrievedChunk, ...]
    metrics: RetrievalMetricsResult

    def __post_init__(self) -> None:
        if not self.query_id.strip():
            raise ValueError("query_id must not be empty")

        if not self.question.strip():
            raise ValueError("question must not be empty")

        chunk_ids = [
            chunk.chunk_id
            for chunk in self.retrieved_chunks
        ]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError(
                "retrieved_chunks must have unique chunk_id values"
            )


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationRunResult:
    query_results: tuple[
        RetrievalQueryEvaluationResult,
        ...,
    ]
    summary: RetrievalMetricsSummary

    def __post_init__(self) -> None:
        if not self.query_results:
            raise ValueError(
                "query_results must not be empty"
            )

        query_ids = [
            result.query_id
            for result in self.query_results
        ]

        if len(query_ids) != len(set(query_ids)):
            raise ValueError(
                "query_results must have unique query_id values"
            )

        if self.summary.query_count != len(
            self.query_results
        ):
            raise ValueError(
                "summary query_count must match query_results"
            )

        if any(
            result.metrics.k != self.summary.k
            for result in self.query_results
        ):
            raise ValueError(
                "all query results must use summary k"
            )
