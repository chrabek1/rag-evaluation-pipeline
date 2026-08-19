from dataclasses import dataclass

from app.models.chunk_evidence_intervals import ChunkEvidenceIntervals


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
                "evidence_coverage must be greater than 0 and less than or equal to 1"
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
                "evidence_intervals must have unique evidence_index values"
            )