from statistics import fmean

from app.models.retrieval_metrics_result import RetrievalMetricsResult
from app.models.retrieval_metrics_summary import RetrievalMetricsSummary


class RetrievalMetricsAggregator:
    def aggregate(
        self, results: list[RetrievalMetricsResult],
    ) -> RetrievalMetricsSummary:
        if not results:
            raise ValueError("results must not be empty")
        
        k_values = {result.k for result in results}
        
        if len(k_values) != 1:
            raise ValueError(
                "all results must use the same k"
            )
        
        k = results[0].k
        
        return RetrievalMetricsSummary(
            k=k,
            query_count=len(results),
            mean_precision_at_k=fmean(
                result.precision_at_k
                for result in results
            ),
            mean_recall_at_k=fmean(
                result.recall_at_k
                for result in results
            ),
            mean_hit_rate_at_k=fmean(
                result.hit_rate_at_k
                for result in results
            ),
            mrr_at_k=fmean(
                result.reciprocal_rank_at_k
                for result in results
            ),
            mean_ndcg_at_k=fmean(
                result.ndcg_at_k
                for result in results
            ),
            mean_graded_ndcg_at_k=fmean(
                result.graded_ndcg_at_k
                for result in results
            ),
            mean_weighted_precision_at_k=fmean(
                result.weighted_precision_at_k
                for result in results
            ),
            mean_evidence_coverage_at_k=fmean(
                result.evidence_coverage_at_k
                for result in results
            ),
        )