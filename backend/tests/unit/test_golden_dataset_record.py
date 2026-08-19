import pytest

from app.models.chunk_evidence_intervals import (
    ChunkEvidenceIntervals,
)
from app.models.evidence_interval import EvidenceInterval
from app.models.golden_dataset_record import (
    GoldenDatasetRecord,
)
from app.models.golden_evidence import GoldenEvidence
from app.models.golden_relevant_chunk import (
    GoldenRelevantChunk,
)


def build_evidence(
    normalized_length: int = 20,
) -> GoldenEvidence:
    return GoldenEvidence(
        text="Relevant evidence fragment.",
        normalized_length=normalized_length,
    )


def build_relevant_chunk(
    chunk_id: str = "document.pdf_0001",
    evidence_index: int = 0,
    interval_end: int = 20,
) -> GoldenRelevantChunk:
    return GoldenRelevantChunk(
        chunk_id=chunk_id,
        evidence_coverage=1.0,
        evidence_intervals=(
            ChunkEvidenceIntervals(
                evidence_index=evidence_index,
                intervals=(
                    EvidenceInterval(
                        start=0,
                        end=interval_end,
                    ),
                ),
            ),
        ),
    )


def test_golden_dataset_record_stores_data() -> None:
    evidence = (build_evidence(),)
    relevant_chunks = (build_relevant_chunk(),)

    record = GoldenDatasetRecord(
        query_id="query-1",
        question="Example question?",
        expected_answer="Example answer.",
        evidence=evidence,
        relevant_chunks=relevant_chunks,
    )

    assert record.query_id == "query-1"
    assert record.question == "Example question?"
    assert record.expected_answer == "Example answer."
    assert record.evidence == evidence
    assert record.relevant_chunks == relevant_chunks


@pytest.mark.parametrize("query_id", ["", " ", "\t", "\n"])
def test_golden_dataset_record_rejects_empty_query_id(
    query_id: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="query_id must not be empty",
    ):
        GoldenDatasetRecord(
            query_id=query_id,
            question="Example question?",
            expected_answer="Example answer.",
            evidence=(build_evidence(),),
            relevant_chunks=(build_relevant_chunk(),),
        )


def test_golden_dataset_record_rejects_empty_question() -> None:
    with pytest.raises(
        ValueError,
        match="question must not be empty",
    ):
        GoldenDatasetRecord(
            query_id="query-1",
            question=" ",
            expected_answer="Example answer.",
            evidence=(build_evidence(),),
            relevant_chunks=(build_relevant_chunk(),),
        )


def test_golden_dataset_record_rejects_empty_expected_answer() -> None:
    with pytest.raises(
        ValueError,
        match="expected_answer must not be empty",
    ):
        GoldenDatasetRecord(
            query_id="query-1",
            question="Example question?",
            expected_answer=" ",
            evidence=(build_evidence(),),
            relevant_chunks=(build_relevant_chunk(),),
        )


def test_golden_dataset_record_rejects_empty_evidence() -> None:
    with pytest.raises(
        ValueError,
        match="evidence must not be empty",
    ):
        GoldenDatasetRecord(
            query_id="query-1",
            question="Example question?",
            expected_answer="Example answer.",
            evidence=(),
            relevant_chunks=(build_relevant_chunk(),),
        )


def test_golden_dataset_record_rejects_empty_relevant_chunks() -> None:
    with pytest.raises(
        ValueError,
        match="relevant_chunks must not be empty",
    ):
        GoldenDatasetRecord(
            query_id="query-1",
            question="Example question?",
            expected_answer="Example answer.",
            evidence=(build_evidence(),),
            relevant_chunks=(),
        )


def test_golden_dataset_record_rejects_duplicate_chunk_ids() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "relevant_chunks must have unique chunk_id values"
        ),
    ):
        GoldenDatasetRecord(
            query_id="query-1",
            question="Example question?",
            expected_answer="Example answer.",
            evidence=(build_evidence(),),
            relevant_chunks=(
                build_relevant_chunk(
                    chunk_id="document.pdf_0001"
                ),
                build_relevant_chunk(
                    chunk_id="document.pdf_0001"
                ),
            ),
        )


def test_golden_dataset_record_rejects_unknown_evidence_index() -> None:
    with pytest.raises(
        ValueError,
        match="evidence_index is outside evidence",
    ):
        GoldenDatasetRecord(
            query_id="query-1",
            question="Example question?",
            expected_answer="Example answer.",
            evidence=(build_evidence(),),
            relevant_chunks=(
                build_relevant_chunk(evidence_index=1),
            ),
        )


def test_golden_dataset_record_rejects_interval_past_evidence() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "evidence interval exceeds "
            "normalized evidence length"
        ),
    ):
        GoldenDatasetRecord(
            query_id="query-1",
            question="Example question?",
            expected_answer="Example answer.",
            evidence=(
                build_evidence(normalized_length=20),
            ),
            relevant_chunks=(
                build_relevant_chunk(interval_end=21),
            ),
        )