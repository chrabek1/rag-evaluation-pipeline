from dataclasses import dataclass

from app.models.golden_dataset_metadata import GoldenDatasetMetadata

from app.models.golden_dataset_record import GoldenDatasetRecord


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