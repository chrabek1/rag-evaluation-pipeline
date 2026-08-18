from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalMetricsSummary:
    k: int
    query_count: int
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_hit_rate_at_k: float
    mrr_at_k: float
    mean_ndcg_at_k: float
    mean_graded_ndcg_at_k: float
    mean_weighted_precision_at_k: float