import pytest

from app.models.golden_dataset import (
    GoldenDatasetMetadata,
)


def test_golden_dataset_metadata_stores_data() -> None:
    metadata = GoldenDatasetMetadata(
        schema_version=1,
        evidence_interval_gap_tolerance=3,
    )

    assert metadata.schema_version == 1
    assert metadata.evidence_interval_gap_tolerance == 3


def test_golden_dataset_metadata_allows_zero_gap_tolerance() -> None:
    metadata = GoldenDatasetMetadata(
        schema_version=1,
        evidence_interval_gap_tolerance=0,
    )

    assert metadata.evidence_interval_gap_tolerance == 0


@pytest.mark.parametrize("schema_version", [0, -1])
def test_golden_dataset_metadata_rejects_invalid_schema_version(
    schema_version: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="schema_version must be greater than 0",
    ):
        GoldenDatasetMetadata(
            schema_version=schema_version,
            evidence_interval_gap_tolerance=3,
        )


def test_golden_dataset_metadata_rejects_negative_gap_tolerance() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "evidence_interval_gap_tolerance "
            "must not be negative"
        ),
    ):
        GoldenDatasetMetadata(
            schema_version=1,
            evidence_interval_gap_tolerance=-1,
        )
