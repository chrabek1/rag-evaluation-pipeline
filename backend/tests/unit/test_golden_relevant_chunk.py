import pytest

from app.models.chunk_evidence_intervals import (
    ChunkEvidenceIntervals,
)
from app.models.evidence_interval import EvidenceInterval
from app.models.golden_relevant_chunk import (
    GoldenRelevantChunk,
)


def build_evidence_intervals(
    evidence_index: int = 0,
) -> ChunkEvidenceIntervals:
    return ChunkEvidenceIntervals(
        evidence_index=evidence_index,
        intervals=(
            EvidenceInterval(start=0, end=10),
        ),
    )


def test_golden_relevant_chunk_stores_data() -> None:
    evidence_intervals = (build_evidence_intervals(),)

    chunk = GoldenRelevantChunk(
        chunk_id="document.pdf_0001",
        evidence_coverage=0.75,
        evidence_intervals=evidence_intervals,
    )

    assert chunk.chunk_id == "document.pdf_0001"
    assert chunk.evidence_coverage == 0.75
    assert chunk.evidence_intervals == evidence_intervals


@pytest.mark.parametrize("chunk_id", ["", " ", "\t", "\n"])
def test_golden_relevant_chunk_rejects_empty_chunk_id(
    chunk_id: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="chunk_id must not be empty",
    ):
        GoldenRelevantChunk(
            chunk_id=chunk_id,
            evidence_coverage=0.5,
            evidence_intervals=(build_evidence_intervals(),),
        )


@pytest.mark.parametrize(
    "evidence_coverage",
    [0.0, -0.1, 1.1],
)
def test_golden_relevant_chunk_rejects_invalid_coverage(
    evidence_coverage: float,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "evidence_coverage must be greater than 0 "
            "and less than or equal to 1"
        ),
    ):
        GoldenRelevantChunk(
            chunk_id="document.pdf_0001",
            evidence_coverage=evidence_coverage,
            evidence_intervals=(build_evidence_intervals(),),
        )


def test_golden_relevant_chunk_rejects_empty_intervals() -> None:
    with pytest.raises(
        ValueError,
        match="evidence_intervals must not be empty",
    ):
        GoldenRelevantChunk(
            chunk_id="document.pdf_0001",
            evidence_coverage=0.5,
            evidence_intervals=(),
        )


def test_golden_relevant_chunk_rejects_duplicate_evidence_indices() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "evidence_intervals must have unique "
            "evidence_index values"
        ),
    ):
        GoldenRelevantChunk(
            chunk_id="document.pdf_0001",
            evidence_coverage=0.5,
            evidence_intervals=(
                build_evidence_intervals(evidence_index=0),
                build_evidence_intervals(evidence_index=0),
            ),
        )