from app.evaluation.retrieval_metrics import (
    graded_ndcg_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
    weighted_precision_at_k,
)
from app.models.retrieval_metrics_result import RetrievalMetricsResult


class RetrievalEvaluator:
    def evaluate(
        self,
        relevance_by_chunk_id: dict[str, float],
        retrieved_chunk_ids: list[str],
        k: int,
    ) -> RetrievalMetricsResult:
        if len(retrieved_chunk_ids) != len(
            set(retrieved_chunk_ids)
        ):
            raise ValueError(
                "retrieved_chunk_ids must not contain duplicates"
            )
        
        relevant_chunk_ids = set(
            relevance_by_chunk_id
        )
        
        return RetrievalMetricsResult(
            k=k,
            precision_at_k=precision_at_k(
                relevant_chunk_ids,
                retrieved_chunk_ids,
                k,
            ),
            recall_at_k=recall_at_k(
                relevant_chunk_ids,
                retrieved_chunk_ids,
                k,
            ),
            hit_rate_at_k=hit_rate_at_k(
                relevant_chunk_ids,
                retrieved_chunk_ids,
                k,
            ),
            reciprocal_rank_at_k=reciprocal_rank_at_k(
                relevant_chunk_ids,
                retrieved_chunk_ids,
                k,
            ),
            ndcg_at_k=ndcg_at_k(
                relevant_chunk_ids,
                retrieved_chunk_ids,
                k,
            ),
            graded_ndcg_at_k=graded_ndcg_at_k(
                relevance_by_chunk_id,
                retrieved_chunk_ids,
                k,
            ),
            weighted_precision_at_k=weighted_precision_at_k(
                relevance_by_chunk_id,
                retrieved_chunk_ids,
                k,
            ),
        )