import pytest

from app.models.golden_dataset import (
    ChunkEvidenceIntervals,
    EvidenceInterval,
)


def test_chunk_evidence_intervals_stores_data() -> None:
    intervals = (
        EvidenceInterval(start=0, end=20),
        EvidenceInterval(start=30, end=50),
    )

    result = ChunkEvidenceIntervals(
        evidence_index=1,
        intervals=intervals,
    )

    assert result.evidence_index == 1
    assert result.intervals == intervals


def test_chunk_evidence_intervals_allows_zero_index() -> None:
    result = ChunkEvidenceIntervals(
        evidence_index=0,
        intervals=(
            EvidenceInterval(start=0, end=10),
        ),
    )

    assert result.evidence_index == 0


def test_chunk_evidence_intervals_rejects_negative_index() -> None:
    with pytest.raises(
        ValueError,
        match="evidence_index must not be negative",
    ):
        ChunkEvidenceIntervals(
            evidence_index=-1,
            intervals=(
                EvidenceInterval(start=0, end=10),
            ),
        )


def test_chunk_evidence_intervals_rejects_empty_intervals() -> None:
    with pytest.raises(
        ValueError,
        match="intervals must not be empty",
    ):
        ChunkEvidenceIntervals(
            evidence_index=0,
            intervals=(),
        )
