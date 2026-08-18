from math import log2


def precision_at_k(
    relevant_chunk_ids: set[str],
    retrieved_chunk_ids: list[str],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be greater than 0")
    
    top_k_chunk_ids = retrieved_chunk_ids[:k]
    
    relevant_retrieved_count = sum(
        chunk_id in relevant_chunk_ids
        for chunk_id in top_k_chunk_ids
    )
    
    return relevant_retrieved_count / k

def recall_at_k(
    relevant_chunk_ids: set[str],
    retrieved_chunk_ids: list[str],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be greater than 0")
    
    if not relevant_chunk_ids:
        raise ValueError("relevant_chunk_ids must not be empty")
    
    top_k_chunk_ids = set(retrieved_chunk_ids[:k])
    retrieved_relevant_chunk_ids = (
        relevant_chunk_ids & top_k_chunk_ids
    )
    
    return (
        len(retrieved_relevant_chunk_ids)
        / len(relevant_chunk_ids)
    )
    
def hit_rate_at_k(
    relevant_chunk_ids: set[str],
    retrieved_chunk_ids: list[str],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be greater than 0")
    
    if not relevant_chunk_ids:
        raise ValueError("relevant_chunk_ids must not be empty")
    
    top_k_chunk_ids = set(retrieved_chunk_ids[:k])
    has_relevant_result = bool(
        relevant_chunk_ids & top_k_chunk_ids
    )
    
    return float(has_relevant_result)

def reciprocal_rank_at_k(
    relevant_chunk_ids: set[str],
    retrieved_chunk_ids: list[str],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be greater than 0")
    
    if not relevant_chunk_ids:
        raise ValueError("relevant_chunk_ids must not be empty")
    
    for rank, chunk_id in enumerate(
        retrieved_chunk_ids[:k],
        start=1,
    ):
        if chunk_id in relevant_chunk_ids:
            return 1.0 / rank
        
    return 0.0

def _discounted_cumulative_gain(
    relevance_scores: list[float],
) -> float:
    return sum(
        relevance / log2(rank + 1)
        for rank, relevance in enumerate(
            relevance_scores,
            start=1,
        )
    )
    
def ndcg_at_k(
    relevant_chunk_ids: set[str],
    retrieved_chunk_ids: list[str],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be greater than 0")
    
    if not relevant_chunk_ids:
        raise ValueError("relevant_chunk_ids must not be empty")
    
    relevance_scores = [
        float(chunk_id in relevant_chunk_ids)
        for chunk_id in retrieved_chunk_ids[:k]
    ]
    
    dcg = _discounted_cumulative_gain(relevance_scores)
    
    ideal_relevant_count = min(
        len(relevant_chunk_ids),
        k,
    )
    ideal_relevance_scores = [1.0] * ideal_relevant_count
    idcg = _discounted_cumulative_gain(
        ideal_relevance_scores
    )
    
    return dcg / idcg

def graded_ndcg_at_k(
    relevance_by_chunk_id: dict[str, float],
    retrieved_chunk_ids: list[str],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be greater than 0")
    
    if not relevance_by_chunk_id:
        raise ValueError(
            "relevance_by_chunk_id must not be empty"
        )
        
    if any(
        score < 0.0 or score > 1.0
        for score in relevance_by_chunk_id.values()
    ):
        raise ValueError(
            "relevance scores must be between 0 and 1"
        )
    
    if not any(
        score > 0.0
        for score in relevance_by_chunk_id.values()
    ):
        raise ValueError(
            "relevance_by_chunk_id must contain at least one positive score"
        )
    
    retrieved_relevance_scores = [
        relevance_by_chunk_id.get(chunk_id, 0.0)
        for chunk_id in retrieved_chunk_ids[:k]
    ]
    
    dcg = _discounted_cumulative_gain(
        retrieved_relevance_scores
    )
    
    ideal_relevance_scores = sorted(
        relevance_by_chunk_id.values(),
        reverse=True,
    )[:k]
    
    idcg = _discounted_cumulative_gain(ideal_relevance_scores)
    
    return dcg / idcg

def weighted_precision_at_k(
    relevance_by_chunk_id: dict[str, float],
    retrieved_chunk_ids: list[str],
    k: int,
)-> float:
    if k <= 0:
        raise ValueError("k must be greater than 0")
    
    if not relevance_by_chunk_id:
        raise ValueError(
            "relevance_by_chunk_id must not be empty"
        )
    
    if any(
        score < 0.0 or score > 1.0
        for score in relevance_by_chunk_id.values()
    ):
        raise ValueError(
            "relevance scores must be between 0 and 1"
        )
    
    top_k_relevance_scores= [
        relevance_by_chunk_id.get(chunk_id, 0.0)
        for chunk_id in retrieved_chunk_ids[:k]
    ]
    
    return sum(top_k_relevance_scores) / k

def evidence_coverage_at_k(
    evidence_lengths: list[int],
    evidence_intervals_by_chunk_id: dict[
        str,
        dict[int, list[tuple[int, int]]],
    ],
    retrieved_chunk_ids: list[str],
    k: int,
    interval_gap_tolerance: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be greater than 0")
    
    if not evidence_lengths:
        raise ValueError("evidence_lengths must not be empty")
    
    if any(length <= 0 for length in evidence_lengths):
        raise ValueError(
            "evidence lengths must be greater than 0"
        )
    
    if interval_gap_tolerance < 0:
        raise ValueError(
            "interval_gap_tolerance must not be negative"
        )
    
    intervals_by_evidence_index = {
        index: []
        for index in range(len(evidence_lengths))
    }
    
    for chunk_id, intervals_by_index in (
        evidence_intervals_by_chunk_id.items()
    ):
        for evidence_index, intervals in intervals_by_index.items():
            if evidence_index not in intervals_by_evidence_index:
                raise ValueError(
                    f"invalid evidence index for chunk {chunk_id}"
                )
            
            evidence_length = evidence_lengths[evidence_index]
            
            for start, end in intervals:
                if not 0 <= start < end <= evidence_length:
                    raise ValueError(
                        f"invalid evidence interval for chunk {chunk_id}"
                    )
    for chunk_id in retrieved_chunk_ids[:k]:
        intervals_by_index = (
            evidence_intervals_by_chunk_id.get(chunk_id, {})
        )
        
        for evidence_index, intervals in intervals_by_index.items():
            intervals_by_evidence_index[
                evidence_index
            ].extend(intervals)
    
    covered_characters = 0
    
    for intervals in intervals_by_evidence_index.values():
        if not intervals:
            continue
        
        sorted_intervals = sorted(intervals)
        merged_start, merged_end = sorted_intervals[0]
        
        for start, end in sorted_intervals[1:]:
            if start <= merged_end + interval_gap_tolerance:
                merged_end = max(merged_end, end)
                continue
            
            covered_characters +=merged_end - merged_start
            merged_start, merged_end = start, end
            
        covered_characters += merged_end - merged_start
    
    total_evidence_characters =sum(evidence_lengths)
    
    return covered_characters / total_evidence_characters