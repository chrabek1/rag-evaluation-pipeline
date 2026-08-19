from dataclasses import dataclass

from app.models.evidence_interval import EvidenceInterval


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