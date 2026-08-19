from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GoldenDatasetMetadata:
    schema_version: int
    evidence_interval_gap_tolerance: int
    
    def __post_init__(self) -> None:
        if self.schema_version <= 0:
            raise ValueError("schema_version must be greater than 0")
        
        if self.evidence_interval_gap_tolerance < 0:
            raise ValueError("evidence_interval_gap_tolerance must not be negative")