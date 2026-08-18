from math import log2

import pytest

from app.evaluation.retrieval_metrics import (
    precision_at_k, 
    recall_at_k, 
    hit_rate_at_k,
    reciprocal_rank_at_k,
    ndcg_at_k,
    graded_ndcg_at_k,
    weighted_precision_at_k,
    
)

def test_precision_at_k_returns_fraction_of_relevant_results() -> None:
    relevant_chunk_ids = {"chunk-2", "chunk-4"}

    retrieved_chunk_ids = [
        "chunk-1",
        "chunk-2",
        "chunk-3",
        "chunk-4",
        "chunk-5",
    ]

    result = precision_at_k(
        relevant_chunk_ids,
        retrieved_chunk_ids,
        k=5,
    )

    assert result == pytest.approx(0.4)


def test_precision_at_k_uses_only_first_k_results() -> None:
    relevant_chunk_ids = {"chunk-3"}

    retrieved_chunk_ids = [
        "chunk-1",
        "chunk-2",
        "chunk-3",
    ]

    result = precision_at_k(
        relevant_chunk_ids,
        retrieved_chunk_ids,
        k=2,
    )

    assert result == 0.0


def test_precision_at_k_returns_zero_when_nothing_is_relevant() -> None:
    result = precision_at_k(
        relevant_chunk_ids={"chunk-10"},
        retrieved_chunk_ids=["chunk-1", "chunk-2"],
        k=2,
    )

    assert result == 0.0


def test_precision_at_k_rejects_non_positive_k() -> None:
    with pytest.raises(
        ValueError,
        match="k must be greater than 0",
    ):
        precision_at_k(
            relevant_chunk_ids={"chunk-1"},
            retrieved_chunk_ids=["chunk-1"],
            k=0,
        )
        
def test_recall_at_k_returns_fraction_of_all_relevant_chunks() -> None:
    relevant_chunk_ids = {
        "chunk-2",
        "chunk-4",
        "chunk-6",
        "chunk-8",
    }

    retrieved_chunk_ids = [
        "chunk-1",
        "chunk-2",
        "chunk-3",
        "chunk-4",
        "chunk-5",
    ]

    result = recall_at_k(
        relevant_chunk_ids,
        retrieved_chunk_ids,
        k=5,
    )

    assert result == pytest.approx(0.5)


def test_recall_at_k_uses_only_first_k_results() -> None:
    result = recall_at_k(
        relevant_chunk_ids={"chunk-3"},
        retrieved_chunk_ids=[
            "chunk-1",
            "chunk-2",
            "chunk-3",
        ],
        k=2,
    )

    assert result == 0.0


def test_recall_at_k_returns_one_when_all_relevant_chunks_are_found() -> None:
    result = recall_at_k(
        relevant_chunk_ids={"chunk-2", "chunk-4"},
        retrieved_chunk_ids=[
            "chunk-1",
            "chunk-2",
            "chunk-4",
        ],
        k=3,
    )

    assert result == 1.0


def test_recall_at_k_rejects_empty_relevant_chunks() -> None:
    with pytest.raises(
        ValueError,
        match="relevant_chunk_ids must not be empty",
    ):
        recall_at_k(
            relevant_chunk_ids=set(),
            retrieved_chunk_ids=["chunk-1"],
            k=1,
        )


def test_recall_at_k_rejects_non_positive_k() -> None:
    with pytest.raises(
        ValueError,
        match="k must be greater than 0",
    ):
        recall_at_k(
            relevant_chunk_ids={"chunk-1"},
            retrieved_chunk_ids=["chunk-1"],
            k=0,
        )
        
def test_hit_rate_at_k_returns_one_when_relevant_chunk_is_found() -> None:
    result = hit_rate_at_k(
        relevant_chunk_ids={"chunk-3"},
        retrieved_chunk_ids=[
            "chunk-1",
            "chunk-3",
            "chunk-5",
        ],
        k=3,
    )

    assert result == 1.0


def test_hit_rate_at_k_returns_zero_when_nothing_is_found() -> None:
    result = hit_rate_at_k(
        relevant_chunk_ids={"chunk-10"},
        retrieved_chunk_ids=[
            "chunk-1",
            "chunk-2",
            "chunk-3",
        ],
        k=3,
    )

    assert result == 0.0


def test_hit_rate_at_k_uses_only_first_k_results() -> None:
    result = hit_rate_at_k(
        relevant_chunk_ids={"chunk-3"},
        retrieved_chunk_ids=[
            "chunk-1",
            "chunk-2",
            "chunk-3",
        ],
        k=2,
    )

    assert result == 0.0


def test_hit_rate_at_k_rejects_empty_relevant_chunks() -> None:
    with pytest.raises(
        ValueError,
        match="relevant_chunk_ids must not be empty",
    ):
        hit_rate_at_k(
            relevant_chunk_ids=set(),
            retrieved_chunk_ids=["chunk-1"],
            k=1,
        )
        
def test_reciprocal_rank_at_k_uses_first_relevant_result() -> None:
    result = reciprocal_rank_at_k(
        relevant_chunk_ids={"chunk-2", "chunk-4"},
        retrieved_chunk_ids=[
            "chunk-1",
            "chunk-2",
            "chunk-3",
            "chunk-4",
        ],
        k=4,
    )

    assert result == pytest.approx(0.5)


def test_reciprocal_rank_at_k_returns_one_for_first_result() -> None:
    result = reciprocal_rank_at_k(
        relevant_chunk_ids={"chunk-1"},
        retrieved_chunk_ids=["chunk-1", "chunk-2"],
        k=2,
    )

    assert result == 1.0


def test_reciprocal_rank_at_k_returns_zero_without_relevant_result() -> None:
    result = reciprocal_rank_at_k(
        relevant_chunk_ids={"chunk-10"},
        retrieved_chunk_ids=["chunk-1", "chunk-2"],
        k=2,
    )

    assert result == 0.0


def test_reciprocal_rank_at_k_uses_only_first_k_results() -> None:
    result = reciprocal_rank_at_k(
        relevant_chunk_ids={"chunk-3"},
        retrieved_chunk_ids=[
            "chunk-1",
            "chunk-2",
            "chunk-3",
        ],
        k=2,
    )

    assert result == 0.0
    
def test_ndcg_at_k_returns_one_for_ideal_ranking() -> None:
    result = ndcg_at_k(
        relevant_chunk_ids={"chunk-1", "chunk-2"},
        retrieved_chunk_ids=[
            "chunk-1",
            "chunk-2",
            "chunk-3",
        ],
        k=3,
    )

    assert result == pytest.approx(1.0)


def test_ndcg_at_k_discounts_relevant_result_at_lower_rank() -> None:
    result = ndcg_at_k(
        relevant_chunk_ids={"chunk-2"},
        retrieved_chunk_ids=[
            "chunk-1",
            "chunk-2",
        ],
        k=2,
    )

    expected = 1.0 / log2(3)

    assert result == pytest.approx(expected)


def test_ndcg_at_k_returns_zero_without_relevant_results() -> None:
    result = ndcg_at_k(
        relevant_chunk_ids={"chunk-10"},
        retrieved_chunk_ids=[
            "chunk-1",
            "chunk-2",
        ],
        k=2,
    )

    assert result == 0.0


def test_ndcg_at_k_uses_only_first_k_results() -> None:
    result = ndcg_at_k(
        relevant_chunk_ids={"chunk-3"},
        retrieved_chunk_ids=[
            "chunk-1",
            "chunk-2",
            "chunk-3",
        ],
        k=2,
    )

    assert result == 0.0
    
    
def test_graded_ndcg_at_k_returns_one_for_ideal_ranking() -> None:
    result = graded_ndcg_at_k(
        relevance_by_chunk_id={
            "chunk-1": 0.9,
            "chunk-2": 0.6,
            "chunk-3": 0.3,
        },
        retrieved_chunk_ids=[
            "chunk-1",
            "chunk-2",
            "chunk-3",
        ],
        k=3,
    )

    assert result == pytest.approx(1.0)


def test_graded_ndcg_at_k_penalizes_incorrect_order() -> None:
    result = graded_ndcg_at_k(
        relevance_by_chunk_id={
            "chunk-1": 0.9,
            "chunk-2": 0.6,
            "chunk-3": 0.3,
        },
        retrieved_chunk_ids=[
            "chunk-3",
            "chunk-1",
            "chunk-2",
        ],
        k=3,
    )

    actual_dcg = (
        0.3
        + 0.9 / log2(3)
        + 0.6 / log2(4)
    )
    ideal_dcg = (
        0.9
        + 0.6 / log2(3)
        + 0.3 / log2(4)
    )

    assert result == pytest.approx(
        actual_dcg / ideal_dcg
    )
    assert result < 1.0


def test_graded_ndcg_at_k_assigns_zero_to_unknown_chunk() -> None:
    result = graded_ndcg_at_k(
        relevance_by_chunk_id={"chunk-1": 0.8},
        retrieved_chunk_ids=[
            "unknown-chunk",
            "chunk-1",
        ],
        k=2,
    )

    expected = (
        0.8 / log2(3)
    ) / 0.8

    assert result == pytest.approx(expected)


def test_graded_ndcg_at_k_uses_only_first_k_results() -> None:
    result = graded_ndcg_at_k(
        relevance_by_chunk_id={"chunk-3": 0.8},
        retrieved_chunk_ids=[
            "chunk-1",
            "chunk-2",
            "chunk-3",
        ],
        k=2,
    )

    assert result == 0.0


def test_graded_ndcg_at_k_rejects_score_outside_range() -> None:
    with pytest.raises(
        ValueError,
        match="relevance scores must be between 0 and 1",
    ):
        graded_ndcg_at_k(
            relevance_by_chunk_id={"chunk-1": 1.2},
            retrieved_chunk_ids=["chunk-1"],
            k=1,
        )
        
    def weighted_precision_at_k(
        relevance_by_chunk_id: dict[str, float],
        retrieved_chunk_ids: list[str],
        k: int,
    ) -> float:
        if k <= 0:
            raise ValueError("k must begreater than 0")
        
        if not relevance_by_chunk_id:
            raise ValueError(
                "relevance_by_chunk_id must not be empty"
            )
        
        if (
            score < 0.0 or score > 1.0
            for score in relevance_by_chunk_id.values()
        ):
            raise ValueError(
                "relevance scores must be between 0 and 1"
            )
        
        top_k_relevance_scores = [
            relevance_by_chunk_id.get(chunk_id, 0.0)
            for chunk_id in retrieved_chunk_ids[:k]
        ]
        
        return sum(top_k_relevance_scores) / k

def test_weighted_precision_at_k_returns_average_coverage() -> None:
    result = weighted_precision_at_k(
        relevance_by_chunk_id={
            "chunk-1": 0.9,
            "chunk-2": 0.3,
        },
        retrieved_chunk_ids=[
            "chunk-1",
            "chunk-2",
            "unknown-chunk",
        ],
        k=3,
    )

    assert result == pytest.approx(0.4)


def test_weighted_precision_at_k_assigns_zero_to_unknown_chunks() -> None:
    result = weighted_precision_at_k(
        relevance_by_chunk_id={"chunk-1": 0.8},
        retrieved_chunk_ids=[
            "unknown-1",
            "unknown-2",
        ],
        k=2,
    )

    assert result == 0.0


def test_weighted_precision_at_k_uses_only_first_k_results() -> None:
    result = weighted_precision_at_k(
        relevance_by_chunk_id={"chunk-3": 0.9},
        retrieved_chunk_ids=[
            "chunk-1",
            "chunk-2",
            "chunk-3",
        ],
        k=2,
    )

    assert result == 0.0


def test_weighted_precision_at_k_rejects_empty_relevance_mapping() -> None:
    with pytest.raises(
        ValueError,
        match="relevance_by_chunk_id must not be empty",
    ):
        weighted_precision_at_k(
            relevance_by_chunk_id={},
            retrieved_chunk_ids=["chunk-1"],
            k=1,
        )