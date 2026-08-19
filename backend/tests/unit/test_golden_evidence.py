import pytest

from app.models.golden_evidence import GoldenEvidence


def test_golden_evidence_stores_text_and_normalized_length() -> None:
    evidence = GoldenEvidence(
        text="Relevant evidence fragment.",
        normalized_length=27,
    )

    assert evidence.text == "Relevant evidence fragment."
    assert evidence.normalized_length == 27


@pytest.mark.parametrize("text", ["", " ", "\t", "\n"])
def test_golden_evidence_rejects_empty_text(text: str) -> None:
    with pytest.raises(
        ValueError,
        match="text must not be empty",
    ):
        GoldenEvidence(
            text=text,
            normalized_length=10,
        )


@pytest.mark.parametrize("normalized_length", [0, -1])
def test_golden_evidence_rejects_non_positive_length(
    normalized_length: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="normalized_length must be greater than 0",
    ):
        GoldenEvidence(
            text="Relevant evidence fragment.",
            normalized_length=normalized_length,
        )