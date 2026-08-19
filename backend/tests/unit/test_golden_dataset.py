import pytest

from app.models.chunk_evidence_intervals import (
    ChunkEvidenceIntervals,
)
from app.models.evidence_interval import EvidenceInterval
from app.models.golden_dataset import GoldenDataset
from app.models.golden_dataset_metadata import (
    GoldenDatasetMetadata,
)
from app.models.golden_dataset_record import (
    GoldenDatasetRecord,
)
from app.models.golden_evidence import GoldenEvidence
from app.models.golden_relevant_chunk import (
    GoldenRelevantChunk,
)


def build_metadata() -> GoldenDatasetMetadata:
    return GoldenDatasetMetadata(
        schema_version=1,
        evidence_interval_gap_tolerance=3,
    )


def build_record(
    query_id: str = "query-1",
) -> GoldenDatasetRecord:
    return GoldenDatasetRecord(
        query_id=query_id,
        question="Example question?",
        expected_answer="Example answer.",
        evidence=(
            GoldenEvidence(
                text="Relevant evidence.",
                normalized_length=18,
            ),
        ),
        relevant_chunks=(
            GoldenRelevantChunk(
                chunk_id=f"{query_id}-chunk",
                evidence_coverage=1.0,
                evidence_intervals=(
                    ChunkEvidenceIntervals(
                        evidence_index=0,
                        intervals=(
                            EvidenceInterval(
                                start=0,
                                end=18,
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


def test_golden_dataset_stores_metadata_and_records() -> None:
    metadata = build_metadata()
    records = (build_record(),)

    dataset = GoldenDataset(
        metadata=metadata,
        records=records,
    )

    assert dataset.metadata == metadata
    assert dataset.records == records


def test_golden_dataset_rejects_empty_records() -> None:
    with pytest.raises(
        ValueError,
        match="records must not be empty",
    ):
        GoldenDataset(
            metadata=build_metadata(),
            records=(),
        )


def test_golden_dataset_rejects_duplicate_query_ids() -> None:
    with pytest.raises(
        ValueError,
        match="records must have unique query_id values",
    ):
        GoldenDataset(
            metadata=build_metadata(),
            records=(
                build_record(query_id="query-1"),
                build_record(query_id="query-1"),
            ),
        )