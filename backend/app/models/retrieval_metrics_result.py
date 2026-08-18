from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalMetricsResult:
    k: int
    precision_at_k: float
    recall_at_k: float
    hit_rate_at_k: float
    reciprocal_rank_at_k: float
    ndcg_at_k: float
    graded_ndcg_at_k: float
    weighted_precision_at_k: float
    evidence_coverage_at_k: float