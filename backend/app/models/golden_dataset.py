from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvidenceInterval:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("start must not be negative")

        if self.end <= self.start:
            raise ValueError("end must be greater than start")


@dataclass(frozen=True, slots=True)
class ChunkEvidenceIntervals:
    evidence_index: int
    intervals: tuple[EvidenceInterval, ...]

    def __post_init__(self) -> None:
        if self.evidence_index < 0:
            raise ValueError(
                "evidence_index must not be negative"
            )

        if not self.intervals:
            raise ValueError("intervals must not be empty")


@dataclass(frozen=True, slots=True)
class GoldenEvidence:
    text: str
    normalized_length: int

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("text must not be empty")

        if self.normalized_length <= 0:
            raise ValueError(
                "normalized_length must be greater than 0"
            )


@dataclass(frozen=True, slots=True)
class GoldenRelevantChunk:
    chunk_id: str
    evidence_coverage: float
    evidence_intervals: tuple[
        ChunkEvidenceIntervals,
        ...,
    ]

    def __post_init__(self) -> None:
        if not self.chunk_id.strip():
            raise ValueError("chunk_id must not be empty")

        if not 0.0 < self.evidence_coverage <= 1.0:
            raise ValueError(
                "evidence_coverage must be greater than 0 "
                "and less than or equal to 1"
            )

        if not self.evidence_intervals:
            raise ValueError(
                "evidence_intervals must not be empty"
            )

        evidence_indices = [
            item.evidence_index
            for item in self.evidence_intervals
        ]

        if len(evidence_indices) != len(set(evidence_indices)):
            raise ValueError(
                "evidence_intervals must have unique "
                "evidence_index values"
            )


@dataclass(frozen=True, slots=True)
class GoldenDatasetRecord:
    query_id: str
    question: str
    expected_answer: str
    evidence: tuple[GoldenEvidence, ...]
    relevant_chunks: tuple[GoldenRelevantChunk, ...]

    def __post_init__(self) -> None:
        if not self.query_id.strip():
            raise ValueError("query_id must not be empty")

        if not self.question.strip():
            raise ValueError("question must not be empty")

        if not self.expected_answer.strip():
            raise ValueError(
                "expected_answer must not be empty"
            )

        if not self.evidence:
            raise ValueError("evidence must not be empty")

        if not self.relevant_chunks:
            raise ValueError(
                "relevant_chunks must not be empty"
            )

        chunk_ids = [
            chunk.chunk_id
            for chunk in self.relevant_chunks
        ]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError(
                "relevant_chunks must have unique chunk_id values"
            )

        for chunk in self.relevant_chunks:
            for group in chunk.evidence_intervals:
                if group.evidence_index >= len(self.evidence):
                    raise ValueError(
                        "evidence_index is outside evidence"
                    )

                evidence_length = self.evidence[
                    group.evidence_index
                ].normalized_length

                if any(
                    interval.end > evidence_length
                    for interval in group.intervals
                ):
                    raise ValueError(
                        "evidence interval exceeds "
                        "normalized evidence length"
                    )


@dataclass(frozen=True, slots=True)
class GoldenDatasetMetadata:
    schema_version: int
    evidence_interval_gap_tolerance: int

    def __post_init__(self) -> None:
        if self.schema_version <= 0:
            raise ValueError(
                "schema_version must be greater than 0"
            )

        if self.evidence_interval_gap_tolerance < 0:
            raise ValueError(
                "evidence_interval_gap_tolerance "
                "must not be negative"
            )


@dataclass(frozen=True, slots=True)
class GoldenDataset:
    metadata: GoldenDatasetMetadata
    records: tuple[GoldenDatasetRecord, ...]

    def __post_init__(self) -> None:
        if not self.records:
            raise ValueError("records must not be empty")

        query_ids = [
            record.query_id
            for record in self.records
        ]

        if len(query_ids) != len(set(query_ids)):
            raise ValueError(
                "records must have unique query_id values"
            )
