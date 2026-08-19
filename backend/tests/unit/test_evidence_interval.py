import pytest

from app.models.golden_dataset import EvidenceInterval


def test_evidence_interval_stores_boundaries() -> None:
    interval = EvidenceInterval(
        start=10,
        end=30,
    )

    assert interval.start == 10
    assert interval.end == 30


def test_evidence_interval_allows_zero_start() -> None:
    interval = EvidenceInterval(
        start=0,
        end=10,
    )

    assert interval.start == 0
    assert interval.end == 10


def test_evidence_interval_rejects_negative_start() -> None:
    with pytest.raises(
        ValueError,
        match="start must not be negative",
    ):
        EvidenceInterval(
            start=-1,
            end=10,
        )


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (10, 10),
        (10, 9),
    ],
)
def test_evidence_interval_rejects_invalid_end(
    start: int,
    end: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="end must be greater than start",
    ):
        EvidenceInterval(
            start=start,
            end=end,
        )
