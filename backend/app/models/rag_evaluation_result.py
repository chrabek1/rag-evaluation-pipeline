from dataclasses import dataclass

from app.models.generation_metrics_result import GenerationMetricsResult
from app.models.generation_metrics_summary import GenerationMetricsSummary
from app.models.generation_result import GenerationResult
from app.models.retrieval_evaluation_result import RetrievalQueryEvaluationResult
from app.models.retrieval_metrics_summary import RetrievalMetricsSummary


@dataclass(frozen=True, slots=True)
class RAGQueryEvaluationResult:
    retrieval: RetrievalQueryEvaluationResult
    expected_answer: str
    generation: GenerationResult
    generation_metrics: GenerationMetricsResult

    def __post_init__(self) -> None:
        if not self.expected_answer.strip():
            raise ValueError("expected_answer must not be empty")

    @property
    def query_id(self) -> str:
        return self.retrieval.query_id

    @property
    def question(self) -> str:
        return self.retrieval.question


@dataclass(frozen=True, slots=True)
class RAGEvaluationRunResult:
    query_results: tuple[
        RAGQueryEvaluationResult,
        ...,
    ]
    retrieval_summary: RetrievalMetricsSummary
    generation_summary: GenerationMetricsResult

    def __post_init__(self) -> None:
        if not self.query_results:
            raise ValueError("query_results must not be empty")

        query_ids = [
            result.query_id
            for result in self.query_results
        ]

        if len(query_ids) != len(set(query_ids)):
            raise ValueError("query_results must have unique query IDs")

        query_count = len(self.query_results)

        if self.retrieval_summary.query_count != query_count:
            raise ValueError("retrieval summary query count must match query_results")

        if self.generation_summary.query_count != query_count:
            raise ValueError("generation summary query count must match query_results")

        if any(
            result.retrieval.metrics.k != self.retrieval_summary.k
            for result in self.query_results
        ):
            raise ValueError("all retrieval results must use summary k")