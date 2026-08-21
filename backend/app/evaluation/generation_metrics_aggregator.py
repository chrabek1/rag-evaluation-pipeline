from collections.abc import Sequence
from statistics import fmean

from app.models.generation_metrics_summary import GenerationMetricsSummary
from app.models.rag_evaluation_result import RAGEvaluationRunResult


class GenerationMetricsAggregator:
    def aggregate(
        self,
        results: Sequence[RAGEvaluationRunResult],
    ) -> GenerationMetricsSummary:
        if not results:
            raise ValueError("results must not be empty")

        input_tokens = [
            result.generation.response.input_tokens
            for result in results
        ]

        output_tokens = [
            result.generation.response.output_tokens
            for result in results
        ]

        return GenerationMetricsSummary(
            query_count=len(results),
            mean_faithfulness=fmean(
                result.generation_metrics.faithfulness
                for result in results
            ),
            mean_answer_relevancy=fmean(
                result.generation_metrics.answer_relevancy
                for result in results
            ),
            mean_latency_seconds=fmean(
                result.generation.latency_seconds
                for result in results
            ),
            total_input_tokens=self._sum_optional(input_tokens),
            total_output_tokens=self._sum_optional(output_tokens),
        )

    @staticmethod
    def _sum_optional(
        values: Sequence[int | None],
    ) -> int | None:
        if any(value is None for value in values):
            return None

        return sum(
            value
            for value in values
            if value is not None
        )