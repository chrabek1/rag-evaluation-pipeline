import json
from pathlib import Path

from app.models.golden_dataset import (
    ChunkEvidenceIntervals,
    EvidenceInterval,
    GoldenDataset,
    GoldenDatasetMetadata,
    GoldenDatasetRecord,
    GoldenEvidence,
    GoldenRelevantChunk,
)


class GoldenDatasetLoader:
    SUPPORTED_SCHEMA_VERSION = 1
    
    def load(self, path: Path) -> GoldenDataset:
        try:
            with path.open("r", encoding="utf-8-sig") as file:
                payload = json.load(file)
        except json.JSONDecodeError as error:
            raise ValueError("golden dataset must contain valid JSON") from error
        
        if not isinstance(payload, dict):
            raise ValueError("golden dataset root must be an object")
        
        try:
            metadata = self._parse_metadata(
                payload["metadata"]
            )
            records_data = payload["records"]
        except KeyError as error:
            raise ValueError(
                f"missing required field: {error.args[0]}"
                ) from error
        
        if not isinstance(records_data, list):
            raise ValueError("records must be a list")
        
        records = tuple(
            self._parse_record(record_data)
            for record_data in records_data
        )
        
        return GoldenDataset(
            metadata=metadata,
            records=records,
        )
    
    def _parse_metadata(
        self,
        data: object,
    ) -> GoldenDatasetMetadata:
        if not isinstance(data, dict):
            raise ValueError("metadata must be an object")
        
        try:
            metadata = GoldenDatasetMetadata(
                schema_version=data["schema_version"],
                evidence_interval_gap_tolerance=data["evidence_interval_gap_tolerance"],
            )
        except KeyError as error:
            raise ValueError(f"missing required field: {error.args[0]}") from error
        
        if metadata.schema_version != self.SUPPORTED_SCHEMA_VERSION:
            raise ValueError(
                "unsupported golden dataset schema_version: "
                f"{metadata.schema_version}"
            )
        
        return metadata
    
    def _parse_record(
        self,
        data: object,
    ) -> GoldenDatasetRecord:
        if not isinstance(data, dict):
            raise ValueError("record must be an object")
        
        try:
            evidence_data = data["evidence"]
            relevant_chunks_data = data["relevant_chunks"]
            
            if not isinstance(evidence_data, list):
                raise ValueError("evidence must be a list")
            
            if not isinstance(relevant_chunks_data, list):
                raise ValueError("relevant_chunks must be a list")
            
            return GoldenDatasetRecord(
                query_id=data["query_id"],
                question=data["question"],
                expected_answer=data["expected_answer"],
                evidence=tuple(
                    self._parse_evidence(item)
                    for item in evidence_data
                ),
                relevant_chunks=tuple(
                    self._parse_relevant_chunk(item)
                    for item in relevant_chunks_data
                ),
            )
        except KeyError as error:
            raise ValueError(
                f"missing required field: {error.args[0]}"
            ) from error
    
    def _parse_evidence(
        self,
        data: object,
    ) -> GoldenEvidence:
        if not isinstance(data, dict):
            raise ValueError(
                "evidence item must be an object"
            )
        
        try:
            return GoldenEvidence(
                text=data["text"],
                normalized_length=data["normalized_length"],
            )
        except KeyError as error:
            raise ValueError(
                f"missing required field: {error.args[0]}"
            ) from error
            
    def _parse_relevant_chunk(
        self,
        data: object,
    ) -> GoldenRelevantChunk:
        if not isinstance(data, dict):
            raise ValueError(
                "relevant chunk must be an object"
            )
        
        try:
            evidence_intervals_data = data["evidence_intervals"]
            
            if not isinstance(evidence_intervals_data, list):
                raise ValueError("evidence_intervals must be a list")
            
            return GoldenRelevantChunk(
                chunk_id=data["chunk_id"],
                evidence_coverage=data["evidence_coverage"],
                evidence_intervals=tuple(
                    self._parse_chunk_evidence_intervals(item)
                    for item in evidence_intervals_data
                ),
            )
        except KeyError as error:
            raise ValueError(
                f"missing required field: {error.args[0]}"
            ) from error
            
    def _parse_chunk_evidence_intervals(
        self,
        data: object,
    ) -> ChunkEvidenceIntervals:
        if not isinstance(data, dict):
            raise ValueError("evidence intervals item must be an object")
        
        try:
            intervals_data = data["intervals"]
            
            if not isinstance(intervals_data, list):
                raise ValueError("intervals must be a list")
            
            return ChunkEvidenceIntervals(
                evidence_index=data["evidence_index"],
                intervals=tuple(
                    self._parse_interval(item)
                    for item in intervals_data
                ),
            )
        except KeyError as error:
            raise ValueError(f"missing required field: {error.args[0]}") from error
        
    def _parse_interval(
        self,
        data: object,
    ) -> EvidenceInterval:
        if (not isinstance(data, list) or len(data) != 2):
            raise ValueError("evidence interval must contain start and end")
        
        return EvidenceInterval(
            start=data[0],
            end=data[1],
        )
