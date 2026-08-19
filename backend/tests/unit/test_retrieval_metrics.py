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
    evidence_coverage_at_k,
    
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


def test_weighted_precision_at_k_returns_one_for_ideal_results() -> None:
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

    assert result == pytest.approx(1.0)


def test_weighted_precision_at_k_normalizes_against_ideal_results() -> None:
    result = weighted_precision_at_k(
        relevance_by_chunk_id={
            "chunk-1": 0.9,
            "chunk-2": 0.3,
        },
        retrieved_chunk_ids=[
            "chunk-1",
            "unknown-chunk",
        ],
        k=2,
    )

    assert result == pytest.approx(0.75)


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


def test_weighted_precision_at_k_rejects_only_zero_scores() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "relevance_by_chunk_id must contain "
            "at least one positive score"
        ),
    ):
        weighted_precision_at_k(
            relevance_by_chunk_id={"chunk-1": 0.0},
            retrieved_chunk_ids=["chunk-1"],
            k=1,
        )


def test_evidence_coverage_at_k_merges_overlapping_intervals() -> None:
    result = evidence_coverage_at_k(
        evidence_lengths=[100],
        evidence_intervals_by_chunk_id={
            "chunk-1": {
                0: [(0, 60)],
            },
            "chunk-2": {
                0: [(40, 100)],
            },
        },
        retrieved_chunk_ids=["chunk-1", "chunk-2"],
        k=2,
        interval_gap_tolerance=3,
    )

    assert result == 1.0


def test_evidence_coverage_at_k_counts_disjoint_intervals() -> None:
    result = evidence_coverage_at_k(
        evidence_lengths=[100],
        evidence_intervals_by_chunk_id={
            "chunk-1": {
                0: [(0, 30)],
            },
            "chunk-2": {
                0: [(60, 100)],
            },
        },
        retrieved_chunk_ids=["chunk-1", "chunk-2"],
        k=2,
        interval_gap_tolerance=3,
    )

    assert result == pytest.approx(0.7)


def test_evidence_coverage_at_k_handles_multiple_evidence() -> None:
    result = evidence_coverage_at_k(
        evidence_lengths=[100, 50],
        evidence_intervals_by_chunk_id={
            "chunk-1": {
                0: [(0, 50)],
            },
            "chunk-2": {
                1: [(0, 50)],
            },
        },
        retrieved_chunk_ids=["chunk-1", "chunk-2"],
        k=2,
        interval_gap_tolerance=3,
    )

    assert result == pytest.approx(100 / 150)


def test_evidence_coverage_at_k_uses_only_first_k_results() -> None:
    result = evidence_coverage_at_k(
        evidence_lengths=[100],
        evidence_intervals_by_chunk_id={
            "chunk-1": {
                0: [(0, 50)],
            },
            "chunk-2": {
                0: [(50, 100)],
            },
        },
        retrieved_chunk_ids=[
            "chunk-1",
            "unknown-chunk",
            "chunk-2",
        ],
        k=2,
        interval_gap_tolerance=3,
    )

    assert result == pytest.approx(0.5)


def test_evidence_coverage_at_k_merges_formatting_gap() -> None:
    result = evidence_coverage_at_k(
        evidence_lengths=[100],
        evidence_intervals_by_chunk_id={
            "chunk-1": {
                0: [(0, 49)],
            },
            "chunk-2": {
                0: [(52, 100)],
            },
        },
        retrieved_chunk_ids=["chunk-1", "chunk-2"],
        k=2,
        interval_gap_tolerance=3,
    )

    assert result == 1.0


def test_evidence_coverage_at_k_returns_zero_for_unknown_chunks() -> None:
    result = evidence_coverage_at_k(
        evidence_lengths=[100],
        evidence_intervals_by_chunk_id={
            "chunk-1": {
                0: [(0, 100)],
            },
        },
        retrieved_chunk_ids=["unknown-chunk"],
        k=1,
        interval_gap_tolerance=3,
    )

    assert result == 0.0


def test_evidence_coverage_at_k_rejects_invalid_interval() -> None:
    with pytest.raises(
        ValueError,
        match="invalid evidence interval",
    ):
        evidence_coverage_at_k(
            evidence_lengths=[100],
            evidence_intervals_by_chunk_id={
                "chunk-1": {
                    0: [(20, 101)],
                },
            },
            retrieved_chunk_ids=["chunk-1"],
            k=1,
            interval_gap_tolerance=3,
        )
